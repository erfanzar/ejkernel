# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA Flash Attention public interface.

This module exposes :func:`flash_attention`, the top-level entry point for
running Flash Attention on NVIDIA GPUs through a CUTLASS-backed CUDA
kernel. The implementation supports:

* Forward and backward passes with JAX ``custom_vjp``.
* Both fixed-length and variable-length (ragged) batches via cumulative
  sequence lengths.
* Optional ALiBi positional bias, causal masking, sliding-window
  attention, logits soft-capping, and dropout.
* Autotuning of CUTLASS tile sizes via
  :func:`ejkernel.ops.execution.tuning.autotune`.

When the input configuration falls outside the capabilities of the CUDA
path (e.g., unsupported dtype, explicit attention masks, segment IDs, or
non-default precision), the function transparently falls back to the
Triton backend and, if that also fails, to the XLA backend.

Module-Level Constants:
    _PREC_TO_CODE: Mapping from :class:`jax.lax.Precision` enum members to
        integer codes understood by the kernel.
    _DTYPE_TO_CODE: Mapping from :class:`jnp.dtype` objects to integer
        codes for the kernel's logits computation dtype.
    PagedKVCache: Type alias for a paged KV-cache triple.
"""

from __future__ import annotations

import functools
import math
import warnings
from collections import OrderedDict

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib._ejit import ejit
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.ops.config.persistent import PersistentCache
from ejkernel.ops.execution.tuning import autotune
from ejkernel.ops.utils.fingerprint import device_fingerprint, short_hash

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl import flash_attention_cuda_bwd, flash_attention_cuda_fwd

_PREC_TO_CODE = {
    jax.lax.Precision.DEFAULT: 0,
    jax.lax.Precision.HIGHEST: 1,
    jax.lax.Precision.HIGH: 2,
}
_DTYPE_TO_CODE = {
    jnp.dtype("float16"): 0,
    jnp.dtype("bfloat16"): 1,
    jnp.dtype("float32"): 2,
    jnp.dtype("float64"): 3,
}

PagedKVCache = tuple[
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Int[Array, "batch max_blocks"],
]


def _precision_to_code(precision) -> int:
    """Convert a JAX precision specification to an integer code.

    Args:
        precision: A :class:`jax.lax.Precision` enum member or an integer
            code (``0``, ``1``, or ``2``).

    Returns:
        Integer code: ``0`` for ``DEFAULT``, ``1`` for ``HIGHEST``, ``2``
        for ``HIGH``.

    Raises:
        ValueError: If *precision* is not a recognised value.
    """
    if isinstance(precision, int):
        return int(precision)
    try:
        return _PREC_TO_CODE[precision]
    except KeyError as exc:
        raise ValueError("precision must be jax.lax.Precision.{DEFAULT|HIGHEST|HIGH} or an int code {0,1,2}.") from exc


def _dtype_to_code(dtype) -> int:
    """Convert a NumPy/JAX dtype to an integer code for the kernel.

    Args:
        dtype: A dtype-like object (string, :class:`jnp.dtype`, or NumPy
            dtype). Accepted values are ``float16``, ``bfloat16``,
            ``float32``, and ``float64``.

    Returns:
        Integer code: ``0`` (float16), ``1`` (bfloat16), ``2`` (float32),
        ``3`` (float64).

    Raises:
        ValueError: If *dtype* is not one of the four supported types.
    """
    d = jnp.dtype(dtype)
    try:
        return _DTYPE_TO_CODE[d]
    except KeyError as exc:
        raise ValueError("logits_dtype must be one of float16, bfloat16, float32, float64.") from exc


def _normalize_window(sliding_window: int | tuple[int, int] | None) -> tuple[int, int] | None:
    """Normalize the sliding-window argument to a ``(left, right)`` tuple.

    Args:
        sliding_window: Window specification. ``None`` disables windowed
            attention. A single ``int`` produces a symmetric window. A
            ``(left, right)`` tuple is accepted directly.

    Returns:
        A ``(left, right)`` integer tuple, or ``None`` when disabled.

    Raises:
        ValueError: If either window bound is negative.
    """
    if sliding_window is None:
        return None
    if isinstance(sliding_window, int):
        return int(sliding_window), int(sliding_window)
    left, right = sliding_window
    if left < 0 or right < 0:
        raise ValueError("Window bounds must be non-negative.")
    return int(left), int(right)


def _attention_mask_to_bias(
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Bool[Array, "batch seq_len_q seq_len_k"]
        | Int[Array, "batch seq_len_q seq_len_k"]
    ),
    *,
    num_heads: int,
    dtype: jnp.dtype = jnp.float32,
) -> Float[Array, "batch num_heads seq_len_q seq_len_k"]:
    """Convert a boolean or integer attention mask to an additive bias tensor.

    Positions that are ``True`` (or non-zero) receive a bias of ``0.0``;
    masked-out positions receive ``-inf`` (the dtype minimum).

    Args:
        attention_mask: Boolean or integer mask of rank 3 or 4.  A rank-3
            mask is interpreted as ``(batch, seq_len_q, seq_len_k)`` and is
            expanded along the head dimension.
        num_heads: Number of attention heads. Used for broadcasting a
            single-head mask to all heads.
        dtype: Output dtype for the bias tensor. Defaults to ``float32``.

    Returns:
        A ``float`` bias tensor of shape
        ``(batch, num_heads, seq_len_q, seq_len_k)``.

    Raises:
        ValueError: If *attention_mask* is not rank 3 or 4, or if its
            head dimension is incompatible with *num_heads*.
    """
    mask = attention_mask
    if mask.dtype != jnp.bool_:
        mask = mask != 0
    if mask.ndim == 3:
        mask = mask[:, None, :, :]
    if mask.ndim != 4:
        raise ValueError(f"attention_mask must be rank 3 or 4; got {mask.ndim}.")
    if mask.shape[1] not in (1, num_heads):
        raise ValueError(f"attention_mask head dim must be 1 or num_heads ({num_heads}); got {mask.shape[1]}.")
    if mask.shape[1] == 1 and num_heads != 1:
        mask = jnp.broadcast_to(mask, (mask.shape[0], num_heads, mask.shape[2], mask.shape[3]))
    neg_inf = jnp.finfo(dtype).min
    return jnp.where(mask, jnp.array(0.0, dtype=dtype), jnp.array(neg_inf, dtype=dtype))


def _sliding_window_bias(query: jax.Array, key: jax.Array, window: tuple[int, int]) -> jax.Array:
    """Create an additive attention bias that implements a sliding window.

    Positions outside the ``[q_idx - left, q_idx + right]`` range for each
    query index receive ``-inf``; positions inside receive ``0``.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``, used only to
            infer dimensions and dtype.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        window: ``(left, right)`` tuple specifying how many key positions
            to the left and right of each query position are visible.

    Returns:
        Bias tensor of shape
        ``(batch, num_heads, seq_len_q, seq_len_k)`` with ``0`` for
        visible positions and ``-inf`` for masked positions.
    """
    left, right = window
    seq_len_q = int(query.shape[1])
    seq_len_k = int(key.shape[1])
    q_idx = jnp.arange(seq_len_q, dtype=jnp.int32)[:, None]
    k_idx = jnp.arange(seq_len_k, dtype=jnp.int32)[None, :]
    mask = (k_idx >= (q_idx - left)) & (k_idx <= (q_idx + right))
    zero = jnp.array(0, dtype=query.dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=query.dtype)
    bias_2d = jnp.where(mask, zero, neg_inf)
    bias = jnp.tile(bias_2d[None, None, :, :], (query.shape[0], query.shape[2], 1, 1))
    return bias


def _can_autotune_cutlass(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
) -> bool:
    """Determine whether CUTLASS tile autotuning is applicable.

    Autotuning is only enabled for the "simple" fast path: GPU backend,
    no masks or segment IDs, no sliding window, no logits soft-cap, half-
    precision inputs (``float16`` or ``bfloat16``), head dimension 64 or
    128, and output normalization enabled.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Optional attention mask (must be ``None``).
        bias: Optional additive bias (must be ``None``).
        softmax_aux: Optional softmax auxiliary (must be ``None``).
        q_segment_ids: Optional query segment IDs (must be ``None``).
        kv_segment_ids: Optional key/value segment IDs (must be ``None``).
        sliding_window: Sliding-window spec (must be ``None``).
        logits_soft_cap: Soft-cap value (must be ``None`` or ``<= 0``).
        normalize_output: Must be ``True``.

    Returns:
        ``True`` if CUTLASS autotuning can be used, ``False`` otherwise.
    """
    if jax.default_backend() != "gpu":
        return False
    if attention_mask is not None or bias is not None:
        return False
    if softmax_aux is not None or q_segment_ids is not None or kv_segment_ids is not None:
        return False
    if sliding_window is not None:
        return False
    if logits_soft_cap is not None and logits_soft_cap > 0.0:
        return False
    if query.dtype not in (jnp.float16, jnp.bfloat16):
        return False
    if value.dtype != query.dtype or key.dtype != query.dtype:
        return False
    if query.shape[-1] not in (64, 128):
        return False
    if not normalize_output:
        return False
    return True


@autotune(
    hyperparams={
        "cutlass_block_m": [128, 64],
        "cutlass_block_n": [128, 256],
        "cutlass_force_aligned": [1, 0],
    },
    timing_warmup_iterations=1,
    timing_rounds=3,
    calls_per_round=3,
    cache_key="cuda_flash_attention_cutlass_tiles_h128",
)
def _flash_attention_cuda_autotuned_h128(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    *,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    cutlass_block_m: int = 128,
    cutlass_block_n: int = 128,
    cutlass_force_aligned: int = 1,
) -> jax.Array:
    """Autotuned CUDA Flash Attention dispatch for head dimension 128.

    Wraps :func:`flash_attention_cuda` with CUTLASS tile-size autotuning
    via the ``@autotune`` decorator. The search space covers
    ``cutlass_block_m`` in ``{128, 64}``, ``cutlass_block_n`` in
    ``{128, 256}``, and ``cutlass_force_aligned`` in ``{1, 0}``.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Unused (must be ``None`` for autotuning).
        bias: Unused (must be ``None`` for autotuning).
        softmax_aux: Unused.
        q_segment_ids: Unused.
        kv_segment_ids: Unused.
        softmax_scale: Softmax scaling factor, or ``None`` for default.
        causal: Whether to apply causal masking.
        sliding_window: Sliding-window spec (must be ``None``).
        logits_soft_cap: Logits soft-cap (must be ``None``).
        normalize_output: Whether to normalize the output.
        cutlass_block_m: CUTLASS tile height (autotuned).
        cutlass_block_n: CUTLASS tile width (autotuned).
        cutlass_force_aligned: Whether to force aligned memory accesses
            (autotuned).

    Returns:
        Attention output with the same shape and dtype as *query*.
    """
    return flash_attention_cuda(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        normalize_output=normalize_output,
        cutlass_block_m=cutlass_block_m,
        cutlass_block_n=cutlass_block_n,
        cutlass_force_aligned=cutlass_force_aligned,
    )


@autotune(
    hyperparams={
        "cutlass_block_m": [64, 128],
        "cutlass_block_n": [128, 64],
        "cutlass_force_aligned": [1, 0],
    },
    timing_warmup_iterations=1,
    timing_rounds=3,
    calls_per_round=3,
    cache_key="cuda_flash_attention_cutlass_tiles_h64",
)
def _flash_attention_cuda_autotuned_h64(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    *,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    cutlass_block_m: int = 64,
    cutlass_block_n: int = 128,
    cutlass_force_aligned: int = 1,
) -> jax.Array:
    """Autotuned CUDA Flash Attention dispatch for head dimension 64.

    Identical to :func:`_flash_attention_cuda_autotuned_h128` but targets
    head dimension 64. The search space covers ``cutlass_block_m`` in
    ``{64, 128}``, ``cutlass_block_n`` in ``{128, 64}``, and
    ``cutlass_force_aligned`` in ``{1, 0}``.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Unused (must be ``None`` for autotuning).
        bias: Unused (must be ``None`` for autotuning).
        softmax_aux: Unused.
        q_segment_ids: Unused.
        kv_segment_ids: Unused.
        softmax_scale: Softmax scaling factor, or ``None`` for default.
        causal: Whether to apply causal masking.
        sliding_window: Sliding-window spec (must be ``None``).
        logits_soft_cap: Logits soft-cap (must be ``None``).
        normalize_output: Whether to normalize the output.
        cutlass_block_m: CUTLASS tile height (autotuned).
        cutlass_block_n: CUTLASS tile width (autotuned).
        cutlass_force_aligned: Whether to force aligned memory accesses
            (autotuned).

    Returns:
        Attention output with the same shape and dtype as *query*.
    """
    return flash_attention_cuda(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        normalize_output=normalize_output,
        cutlass_block_m=cutlass_block_m,
        cutlass_block_n=cutlass_block_n,
        cutlass_force_aligned=cutlass_force_aligned,
    )


_CUTLASS_TILE_CACHE: OrderedDict[tuple[object, ...], dict[str, int]] = OrderedDict()
_CUTLASS_TILE_CACHE_LIMIT = 64
_CUTLASS_TILE_PERSIST = PersistentCache("flash-attn-cutlass-tiles", cfg_type=dict)
_CUTLASS_TILE_PERSIST_OP = "cutlass_tiles_v1"


def _cutlass_cache_key(
    query: jax.Array,
    key: jax.Array,
    softmax_scale: float | None,
    causal: bool,
) -> tuple[object, ...] | None:
    """Build a hashable cache key for CUTLASS tile lookup.

    Returns ``None`` during JAX abstract tracing (when shapes are not
    yet concrete), preventing cache lookups that would fail.

    Args:
        query: Query tensor (must be concrete for a valid key).
        key: Key tensor (must be concrete for a valid key).
        softmax_scale: Softmax scaling factor, or ``None``.
        causal: Whether causal masking is enabled.

    Returns:
        A hashable tuple encoding shape, dtype, scale, and causality, or
        ``None`` if the arrays are abstract.
    """
    if not jax.core.is_concrete(query) or not jax.core.is_concrete(key):
        return None
    return (
        tuple(query.shape),
        tuple(key.shape),
        query.dtype,
        softmax_scale if softmax_scale is not None else "default",
        bool(causal),
    )


def _cutlass_cache_get(key: tuple[object, ...] | None) -> dict[str, int] | None:
    """Look up previously autotuned CUTLASS tile parameters.

    Checks the in-memory LRU cache first, then falls back to the
    persistent on-disk cache. A hit is promoted to the most-recently-used
    position in memory.

    Args:
        key: Cache key produced by :func:`_cutlass_cache_key`, or ``None``
            to skip the lookup.

    Returns:
        A dictionary of CUTLASS tile parameters (e.g.,
        ``{"cutlass_block_m": 128, ...}``), or ``None`` on a miss.
    """
    if key is None:
        return None
    try:
        dev = device_fingerprint()
    except Exception:
        dev = "unknown"
    mem_key = (dev, *key)
    hit = _CUTLASS_TILE_CACHE.get(mem_key)
    if hit is not None:
        _CUTLASS_TILE_CACHE.move_to_end(mem_key)
        return hit
    try:
        persisted = _CUTLASS_TILE_PERSIST.get(dev, _CUTLASS_TILE_PERSIST_OP, short_hash(key))
    except Exception:
        persisted = None
    if isinstance(persisted, dict):
        _CUTLASS_TILE_CACHE[mem_key] = persisted
        _CUTLASS_TILE_CACHE.move_to_end(mem_key)
        if len(_CUTLASS_TILE_CACHE) > _CUTLASS_TILE_CACHE_LIMIT:
            _CUTLASS_TILE_CACHE.popitem(last=False)
        return persisted
    return None


def _cutlass_cache_put(key: tuple[object, ...] | None, value: dict[str, int]) -> None:
    """Store autotuned CUTLASS tile parameters in both memory and disk caches.

    The in-memory :data:`_CUTLASS_TILE_CACHE` is an LRU
    :class:`~collections.OrderedDict` with a capacity of
    :data:`_CUTLASS_TILE_CACHE_LIMIT` entries. Eviction follows
    least-recently-used order. Errors during disk persistence are silently
    ignored.

    Args:
        key: Cache key produced by :func:`_cutlass_cache_key`, or ``None``
            to skip the store.
        value: Dictionary of CUTLASS tile parameters to cache.
    """
    if key is None:
        return
    try:
        dev = device_fingerprint()
    except Exception:
        dev = "unknown"
    mem_key = (dev, *key)
    _CUTLASS_TILE_CACHE[mem_key] = value
    _CUTLASS_TILE_CACHE.move_to_end(mem_key)
    if len(_CUTLASS_TILE_CACHE) > _CUTLASS_TILE_CACHE_LIMIT:
        _CUTLASS_TILE_CACHE.popitem(last=False)
    try:
        _CUTLASS_TILE_PERSIST.put(dev, _CUTLASS_TILE_PERSIST_OP, short_hash(key), value)
    except Exception:
        pass


def _pick_chunk_sizes(
    query: jax.Array,
    key: jax.Array,
    fwd_params: FwdParams | None,
    logits_soft_cap: float | None,
    softmax_aux: jax.Array | None,
) -> tuple[int, int]:
    """Select query and key/value block sizes for the attention computation.

    When :class:`FwdParams` are not supplied, the block sizes default to
    ``min(128, seq_len)``. Logits soft-capping raises the minimum block
    size to avoid numerical issues.

    Args:
        query: Query tensor, used to infer ``seq_len_q``.
        key: Key tensor, used to infer ``seq_len_k``.
        fwd_params: Optional forward-pass parameters carrying explicit
            block-size overrides.
        logits_soft_cap: Soft-cap value; when set, minimum block sizes
            are raised.
        softmax_aux: Auxiliary softmax tensor; when provided together with
            *logits_soft_cap*, the minimum block size is raised to 32.

    Returns:
        ``(q_block, kv_block)`` integer pair.
    """
    if fwd_params is None:
        q_block = min(128, int(query.shape[1]))
        kv_block = min(128, int(key.shape[1]))
    else:
        q_block = min(128, int(query.shape[1])) if fwd_params.q_blocksize is None else int(fwd_params.q_blocksize)
        kv_block = min(128, int(key.shape[1])) if fwd_params.kv_blocksize is None else int(fwd_params.kv_blocksize)

    q_block = max(1, q_block)
    kv_block = max(1, kv_block)

    if logits_soft_cap is not None:
        min_block = 32 if softmax_aux is not None else 16
        q_block = max(min_block, q_block)
        kv_block = max(min_block, kv_block)
    return q_block, kv_block


def _is_alibi_slopes(bias: jax.Array | None, num_heads: int) -> bool:
    """Check whether *bias* represents a per-head ALiBi slopes vector.

    ALiBi slopes are identified by shape: either a 1-D tensor of length
    ``num_heads`` or a 2-D tensor whose second dimension equals
    ``num_heads``.

    Args:
        bias: Bias tensor to inspect, or ``None``.
        num_heads: Number of attention heads.

    Returns:
        ``True`` if *bias* is an ALiBi slopes tensor, ``False`` otherwise.
    """
    if bias is None:
        return False
    if bias.ndim == 1 and int(bias.shape[0]) == num_heads:
        return True
    if bias.ndim == 2 and int(bias.shape[1]) == num_heads:
        return True
    return False


def flash_attention_cuda(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None = None,
    bias: jax.Array | None = None,
    softmax_aux: jax.Array | None = None,
    q_segment_ids: jax.Array | None = None,
    kv_segment_ids: jax.Array | None = None,
    block_tables: jax.Array | None = None,
    *,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    **_: int,
) -> jax.Array:
    """Run the CUDA Flash Attention forward pass (inference-only, no grad).

    This is a simplified entry point that calls
    :func:`._cuda_impl.flash_attention_cuda_fwd` with fixed-length
    tensors and no dropout. It is called by the autotuned wrappers and
    by :func:`flash_attention_call`.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        value: Value tensor of the same shape as *key*.
        attention_mask: Ignored (unsupported by the CUDA kernel).
        bias: Optional ALiBi slopes tensor. Full attention bias tensors
            are not supported and are silently ignored.
        softmax_aux: Ignored.
        q_segment_ids: Ignored.
        kv_segment_ids: Ignored.
        block_tables: Optional paged-KV block table. When set, *key* and
            *value* are interpreted as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        softmax_scale: Softmax scaling factor, or ``None`` for the
            default ``head_dim ** -0.5``.
        causal: Whether to apply a causal mask.
        sliding_window: Sliding-window specification, or ``None``.
        logits_soft_cap: Soft-cap for attention logits, or ``None``.
        normalize_output: Whether to normalize the output by the softmax
            denominator.
        **_: Extra keyword arguments (e.g., CUTLASS tile parameters) are
            accepted and ignored.

    Returns:
        Attention output of the same shape and dtype as *query*.
    """
    del attention_mask, softmax_aux, q_segment_ids, kv_segment_ids
    alibi_slopes = bias if _is_alibi_slopes(bias, int(query.shape[-2])) else None
    out, _, _ = flash_attention_cuda_fwd(
        query=query,
        key=key,
        value=value,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        block_tables=block_tables,
        softmax_scale=softmax_scale,
        dropout_prob=0.0,
        dropout_seed=0,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        normalize_output=normalize_output,
        max_seqlen_q=int(query.shape[1]),
        max_seqlen_k=int(key.shape[1]),
        is_varlen=False,
    )
    return out


def _should_fallback(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    dropout_prob: float,
    cum_seqlens_q: jax.Array | None,
    cum_seqlens_k: jax.Array | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    block_tables: jax.Array | None,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
) -> bool:
    """Decide whether the request must fall back to a non-CUDA backend.

    The CUDA kernel requires all of the following to hold:

    * Running on a GPU backend.
    * No explicit ``attention_mask``, ``softmax_aux``, or segment IDs.
    * ``cum_seqlens_q`` and ``cum_seqlens_k`` must either both be present
      or both absent; when present, inputs must be rank 3.
    * Matching dtypes across Q, K, V (``float16`` or ``bfloat16``).
    * Bias, if provided, must be an ALiBi slopes tensor.
    * Output normalization must be enabled.
    * Precision must be ``DEFAULT`` and logits dtype must be ``float32``.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Explicit mask tensor (triggers fallback if set).
        bias: Additive bias or ALiBi slopes.
        dropout_prob: Dropout probability (not used for the check, but
            passed through).
        cum_seqlens_q: Cumulative query sequence lengths.
        cum_seqlens_k: Cumulative key sequence lengths.
        softmax_aux: Auxiliary softmax tensor (triggers fallback if set).
        q_segment_ids: Query segment IDs (triggers fallback if set).
        kv_segment_ids: Key/value segment IDs (triggers fallback if set).
        block_tables: Optional paged-KV block table (CUDA-only feature).
        normalize_output: Must be ``True`` for the CUDA path.
        precision: JAX precision enum or int code.
        logits_dtype: Dtype for logits computation.

    Returns:
        ``True`` if the request cannot be handled by the CUDA kernel and
        should be routed to a fallback backend.
    """
    if jax.default_backend() != "gpu":
        return True
    if attention_mask is not None:
        return True
    if softmax_aux is not None:
        return True
    if q_segment_ids is not None or kv_segment_ids is not None:
        return True
    if block_tables is not None:
        if cum_seqlens_q is not None or cum_seqlens_k is not None:
            return True
    if (cum_seqlens_q is None) != (cum_seqlens_k is None):
        return True
    if cum_seqlens_q is not None:
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            return True
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return True
    if query.dtype not in (jnp.float16, jnp.bfloat16):
        return True
    if bias is not None and not _is_alibi_slopes(bias, int(query.shape[-2])):
        return True
    if not normalize_output:
        return True
    if isinstance(precision, int):
        if int(precision) != 0:
            return True
    elif precision != lax.Precision.DEFAULT:
        return True
    if jnp.dtype(logits_dtype) != jnp.float32:
        return True
    return False


def _fallback_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    dropout_seed: int | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    cum_seqlens_q: jax.Array | None,
    cum_seqlens_k: jax.Array | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    softmax_aux: jax.Array | None,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
):
    """Dispatch to the Triton or XLA Flash Attention backend as a fallback.

    When running on a GPU the function first attempts the Triton backend.
    If the import fails or the Triton call raises an exception, it falls
    through to the XLA backend, which is always available.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Optional boolean/integer attention mask.
        bias: Optional additive attention bias.
        softmax_scale: Softmax scaling factor, or ``None``.
        dropout_prob: Dropout probability.
        causal: Whether to apply causal masking.
        dropout_seed: RNG seed for dropout, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Optional backward-pass tuning parameters.
        cum_seqlens_q: Cumulative query sequence lengths, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window specification, or ``None``.
        logits_soft_cap: Logits soft-cap value, or ``None``.
        softmax_aux: Auxiliary softmax tensor, or ``None``.
        q_segment_ids: Query segment IDs, or ``None``.
        kv_segment_ids: Key/value segment IDs, or ``None``.
        normalize_output: Whether to normalize the output.
        precision: JAX precision enum.
        logits_dtype: Dtype for internal logits computation.

    Returns:
        Attention output tensor from the Triton or XLA backend.
    """
    if jax.default_backend() == "gpu":
        try:
            from ejkernel.kernels._triton.flash_attention import flash_attention as triton_flash_attention

            return triton_flash_attention(
                query,
                key,
                value,
                attention_mask=attention_mask,
                bias=bias,
                softmax_scale=softmax_scale,
                dropout_prob=dropout_prob,
                causal=causal,
                dropout_seed=dropout_seed,
                cum_seqlens_q=cum_seqlens_q,
                cum_seqlens_k=cum_seqlens_k,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                softmax_aux=softmax_aux,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                fwd_params=fwd_params,
                bwd_params=bwd_params,
                normalize_output=normalize_output,
                precision=precision,
                logits_dtype=logits_dtype,
            )
        except Exception:
            pass

    from ejkernel.kernels._xla.flash_attention import flash_attention as xla_flash_attention

    return xla_flash_attention(
        query,
        key,
        value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
    )


def _jax_fwd_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    block_tables: Int[Array, "batch max_blocks"] | None = None,
):
    """Custom VJP forward rule for CUDA Flash Attention.

    Computes the forward pass and returns the output together with a
    residual tuple that is consumed by :func:`_jax_bwd_attention_call`.
    Handles causal/sliding-window interaction, variable-length batches,
    and ALiBi slope detection.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attention_mask: Optional attention mask (unused by CUDA kernel).
        bias: Optional additive bias or ALiBi slopes tensor.
        softmax_scale: Softmax scaling factor, or ``None``.
        dropout_prob: Dropout probability.
        causal: Whether to apply causal masking.
        dropout_seed: RNG seed for dropout, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Ignored (backward parameters are captured in the
            residual).
        cum_seqlens_q: Cumulative query sequence lengths for
            variable-length batches, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window size, or ``None``.
        logits_soft_cap: Logits soft-cap value, or ``None``.
        softmax_aux: Auxiliary softmax tensor, or ``None``.
        q_segment_ids: Query segment IDs, or ``None``.
        kv_segment_ids: Key/value segment IDs, or ``None``.
        normalize_output: Whether to normalize the output.
        precision: JAX precision enum.
        logits_dtype: Dtype for logits computation.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When set, *key* and *value* are
            treated as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.

    Returns:
        A 2-tuple ``(out, residual)`` where *out* is the attention output
        and *residual* is a tuple of tensors and scalars needed by the
        backward pass.
    """
    del bwd_params

    window_tuple = _normalize_window(sliding_window)
    cuda_causal = bool(causal)
    cuda_window = window_tuple
    if window_tuple is not None and cuda_causal:
        left, _ = window_tuple
        cuda_window = (left, 0)
        cuda_causal = False

    if softmax_scale is None:
        scale_val = float(1.0 / math.sqrt(query.shape[-1]))
    else:
        scale_val = float(softmax_scale)

    is_varlen = cum_seqlens_q is not None and cum_seqlens_k is not None
    if block_tables is not None and is_varlen:
        raise ValueError("paged_kv does not support varlen inputs for CUDA flash attention.")
    if is_varlen:
        max_seqlen_q = int(query.shape[0])
        max_seqlen_k = int(key.shape[0])
    else:
        max_seqlen_q = int(query.shape[1])
        if block_tables is not None:
            max_seqlen_k = int(block_tables.shape[1]) * int(key.shape[1])
        else:
            max_seqlen_k = int(key.shape[1])

    alibi_slopes = bias if _is_alibi_slopes(bias, int(query.shape[-2])) else None
    if dropout_seed is None:
        dropout_seed = 0

    out, softmax_lse, rng_state = flash_attention_cuda_fwd(
        query=query,
        key=key,
        value=value,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cum_seqlens_q,
        cu_seqlens_k=cum_seqlens_k,
        block_tables=block_tables,
        softmax_scale=scale_val,
        dropout_prob=dropout_prob,
        dropout_seed=dropout_seed,
        causal=cuda_causal,
        sliding_window=cuda_window,
        logits_soft_cap=logits_soft_cap,
        normalize_output=normalize_output,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        is_varlen=is_varlen,
    )

    residual = (
        query,
        key,
        value,
        out,
        softmax_lse,
        rng_state,
        alibi_slopes,
        cum_seqlens_q,
        cum_seqlens_k,
        cuda_window,
        cuda_causal,
        scale_val,
        logits_soft_cap,
        dropout_prob,
        normalize_output,
        max_seqlen_q,
        max_seqlen_k,
        is_varlen,
        block_tables,
    )
    return out, residual


def _jax_bwd_attention_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
    residual: tuple,
    dO: Float[Array, "batch seq_len num_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len_q num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Custom VJP backward rule for CUDA Flash Attention.

    Unpacks the *residual* tuple saved by :func:`_jax_fwd_attention_call`
    and invokes :func:`._cuda_impl.flash_attention_cuda_bwd` to compute
    gradients with respect to query, key, and value. Gradients for all
    other arguments are returned as ``None``.

    Args:
        softmax_scale: Ignored (already captured in residual).
        dropout_prob: Ignored (already captured in residual).
        causal: Ignored (already captured in residual).
        fwd_params: Ignored.
        bwd_params: Ignored.
        sliding_window: Ignored (already captured in residual).
        logits_soft_cap: Ignored (already captured in residual).
        normalize_output: Ignored (already captured in residual).
        precision: Ignored.
        logits_dtype: Ignored.
        residual: Tuple of tensors and scalars saved during the forward
            pass.
        dO: Upstream gradient of the attention output.

    Returns:
        An 11-tuple where the first three elements are ``dq``, ``dk``,
        ``dv`` (gradients w.r.t. query, key, value) and the remaining
        eight are ``None`` (no gradient for the corresponding inputs).
    """
    del softmax_scale, fwd_params, bwd_params, sliding_window, logits_soft_cap, normalize_output, precision, logits_dtype

    (
        query,
        key,
        value,
        out,
        softmax_lse,
        rng_state,
        alibi_slopes,
        cu_seqlens_q,
        cu_seqlens_k,
        window_tuple,
        causal_val,
        scale_val,
        logits_soft_cap_val,
        dropout_prob_val,
        normalize_output_val,
        max_seqlen_q,
        max_seqlen_k,
        is_varlen,
        block_tables,
    ) = residual

    if block_tables is not None:
        raise ValueError("paged_kv is not supported for CUDA flash attention backward.")

    dq, dk, dv = flash_attention_cuda_bwd(
        query=query,
        key=key,
        value=value,
        out=out,
        d_out=dO,
        softmax_lse=softmax_lse,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
        rng_state=rng_state,
        softmax_scale=float(scale_val),
        dropout_prob=float(dropout_prob_val),
        causal=bool(causal_val),
        sliding_window=window_tuple,
        logits_soft_cap=logits_soft_cap_val,
        normalize_output=bool(normalize_output_val),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        is_varlen=bool(is_varlen),
    )

    return (
        dq,
        dk,
        dv,
        None,  # attention_mask
        None,  # bias
        None,  # dropout_seed
        None,  # cum_seqlens_q
        None,  # cum_seqlens_k
        None,  # softmax_aux
        None,  # q_segment_ids
        None,  # kv_segment_ids
        None,  # block_tables
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9, 10, 13, 14, 18, 19, 20))
@ejit(static_argnums=(5, 6, 7, 9, 10, 13, 14, 18, 19, 20))
def flash_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | Float[Array, "num_heads"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    block_tables: Int[Array, "batch max_blocks"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Differentiable CUDA Flash Attention call with custom VJP.

    This function is decorated with :func:`jax.custom_vjp` so that
    :func:`_jax_fwd_attention_call` and :func:`_jax_bwd_attention_call`
    are used for forward and backward passes respectively. It is also JIT-
    compiled via ``@ejit``.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        value: Value tensor with the same shape as *key*.
        attention_mask: Optional boolean/integer mask.
        bias: Optional additive bias or ALiBi slopes.
        softmax_scale: Softmax scaling factor, or ``None``.
        dropout_prob: Dropout probability in ``[0, 1)``.
        causal: Whether to apply causal masking.
        dropout_seed: RNG seed for dropout, or ``None``.
        fwd_params: Optional forward-pass tuning parameters.
        bwd_params: Optional backward-pass tuning parameters.
        cum_seqlens_q: Cumulative query sequence lengths for
            variable-length batches, or ``None``.
        cum_seqlens_k: Cumulative key sequence lengths, or ``None``.
        sliding_window: Sliding-window specification, or ``None``.
        logits_soft_cap: Logits soft-cap, or ``None``.
        softmax_aux: Auxiliary softmax tensor, or ``None``.
        q_segment_ids: Query segment IDs, or ``None``.
        kv_segment_ids: Key/value segment IDs, or ``None``.
        normalize_output: Whether to normalize the output.
        precision: JAX precision enum.
        logits_dtype: Dtype for internal logits computation.
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            expected to be paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.

    Returns:
        Attention output tensor of shape
        ``(batch, seq_len_q, num_heads, head_dim)``.
    """
    return _jax_fwd_attention_call(
        query,
        key,
        value,
        attention_mask,
        bias,
        softmax_scale,
        dropout_prob,
        causal,
        dropout_seed,
        fwd_params,
        bwd_params,
        cum_seqlens_q,
        cum_seqlens_k,
        sliding_window,
        logits_soft_cap,
        softmax_aux,
        q_segment_ids,
        kv_segment_ids,
        normalize_output,
        precision,
        logits_dtype,
        block_tables,
    )[0]


flash_attention_call.defvjp(_jax_fwd_attention_call, _jax_bwd_attention_call)


@kernel_registry.register("flash_attention", Platform.CUDA, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: (
        Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
        | None
    ) = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    block_tables: Int[Array, "batch max_blocks"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Compute scaled dot-product attention using the CUDA Flash Attention kernel.

    This is the top-level public entry point registered with the ejKernel
    kernel registry under ``("flash_attention", Platform.CUDA, Backend.GPU)``.
    It validates inputs, converts unsupported attention masks to additive
    biases where possible, and either dispatches to the native CUDA path
    or transparently falls back to the Triton / XLA backends.

    The CUDA path requires ``float16`` or ``bfloat16`` inputs with
    matching dtypes, no explicit mask or segment IDs, ``DEFAULT``
    precision, and ``float32`` logits dtype. Any deviation triggers the
    fallback.

    Args:
        query: Query tensor of shape
            ``(batch, seq_len_q, num_heads, head_dim)``.
        key: Key tensor of shape
            ``(batch, seq_len_k, num_kv_heads, head_dim)``.
        value: Value tensor with the same shape as *key*.
        attention_mask: Optional boolean or integer mask of shape
            ``(batch, [num_heads,] seq_len_q, seq_len_k)``. When provided,
            the CUDA kernel cannot consume it directly; the mask is
            converted to an additive bias or dropped with a warning.
        bias: Optional additive bias of shape
            ``(batch, num_heads, seq_len_q, seq_len_k)`` or an ALiBi slopes
            vector of shape ``(num_heads,)``.
        softmax_scale: Multiplicative scaling applied to QK^T before
            softmax. Defaults to ``head_dim ** -0.5`` when ``None``.
        dropout_prob: Dropout probability in ``[0, 1)``. Defaults to
            ``0.0`` (no dropout).
        causal: If ``True``, apply a causal (lower-triangular) mask so
            that each query position only attends to earlier key positions.
        dropout_seed: Integer seed for dropout. ``None`` defaults to ``0``.
        cum_seqlens_q: Cumulative query sequence lengths of shape
            ``(batch + 1,)`` for variable-length (ragged) batches. Pass
            ``None`` for fixed-length batches.
        cum_seqlens_k: Cumulative key sequence lengths, analogous to
            *cum_seqlens_q*.
        sliding_window: Sliding-window size as a single ``int`` (symmetric),
            a ``(left, right)`` tuple, or ``None`` to disable.
        fwd_params: Optional :class:`~ejkernel.ops.FwdParams` for
            forward-pass block-size tuning.
        bwd_params: Optional :class:`~ejkernel.ops.BwdParams` for
            backward-pass tuning.
        logits_soft_cap: Soft-capping value applied to raw attention logits
            before softmax. ``None`` disables capping.
        softmax_aux: Auxiliary softmax tensor (e.g., sink tokens). ``None``
            disables.
        normalize_output: If ``True`` (default), divide the output by the
            softmax normalizer as in standard attention.
        precision: JAX arithmetic precision for the fallback backends.
            Defaults to :attr:`jax.lax.Precision.DEFAULT`.
        logits_dtype: Dtype for internal logits computation. Defaults to
            ``jnp.float32``.
        q_segment_ids: Optional query segment IDs of shape
            ``(batch, seq_len_q)`` for document-aware masking (keyword-only).
        kv_segment_ids: Optional key/value segment IDs of shape
            ``(batch, seq_len_k)`` (keyword-only).
        block_tables: Optional paged-KV block table of shape
            ``(batch, max_blocks)``. When provided, *key* and *value* are
            treated as paged KV caches with shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``
            (keyword-only).

    Returns:
        Attention output tensor of shape
        ``(batch, seq_len_q, num_heads, head_dim)`` with the same dtype
        as *query*.

    Warns:
        UserWarning: When *attention_mask* is provided, since the CUDA
            kernel does not natively support masks. The mask is either
            converted to a bias or silently dropped.
    """
    if attention_mask is not None:
        if bias is None:
            warnings.warn(
                "CUDA flash_attention does not support attention_mask; "
                "converting mask to bias and falling back when needed.",
                stacklevel=2,
            )
            bias = _attention_mask_to_bias(attention_mask, num_heads=int(query.shape[2]))
        else:
            warnings.warn(
                "CUDA flash_attention does not support attention_mask; dropping attention_mask and using bias only.",
                stacklevel=2,
            )
        attention_mask = None

    if _should_fallback(
        query,
        key,
        value,
        attention_mask,
        bias,
        dropout_prob,
        cum_seqlens_q,
        cum_seqlens_k,
        softmax_aux,
        q_segment_ids,
        kv_segment_ids,
        block_tables,
        normalize_output,
        precision,
        logits_dtype,
    ):
        if block_tables is not None:
            raise ValueError("paged_kv is only supported by the CUDA backend without masks/varlen/segments.")
        return _fallback_flash_attention(
            query,
            key,
            value,
            attention_mask,
            bias,
            softmax_scale,
            dropout_prob,
            causal,
            dropout_seed,
            fwd_params,
            bwd_params,
            cum_seqlens_q,
            cum_seqlens_k,
            sliding_window,
            logits_soft_cap,
            softmax_aux,
            q_segment_ids,
            kv_segment_ids,
            normalize_output,
            precision,
            logits_dtype,
        )

    return flash_attention_call(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        block_tables=block_tables,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
    )
