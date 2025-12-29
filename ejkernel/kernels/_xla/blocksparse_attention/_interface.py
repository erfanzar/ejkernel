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

"""Block-sparse attention interface for XLA fallback computation.

This module provides the public API for block-sparse attention that
handles packed multi-sequence inputs with segment IDs and positions.
Acts as a correctness fallback when specialized kernels are unavailable.
"""

from __future__ import annotations

import typing as tp

import jax
import jaxtyping
from beartype import beartype
from beartype.typing import Callable
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ..attention import attention as dense_attention

if tp.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


def _normalize_segment_ids(ids: Int[Array, "..."] | None, *, which: str) -> Int[Array, "batch seqlen"] | None:
    if ids is None:
        return None
    ids = jnp.asarray(ids, jnp.int32)
    if ids.ndim == 2:
        return ids
    if ids.ndim == 3:
        return ids[:, 0, :]
    raise ValueError(f"{which}_segment_ids must be 2D or 3D, got shape {ids.shape}")


def _normalize_positions(
    pos: Int[Array, "..."] | None, *, batch: int, seqlen: int, fill: int
) -> Int[Array, "batch seqlen"]:
    if pos is None:
        return jnp.broadcast_to(jnp.arange(seqlen, dtype=jnp.int32)[None, :], (batch, seqlen))
    pos = jnp.asarray(pos, jnp.int32)
    if pos.shape != (batch, seqlen):
        raise ValueError(f"positions must have shape {(batch, seqlen)}, got {pos.shape}")
    return jnp.where(jnp.isnan(pos), fill, pos).astype(jnp.int32) if jnp.issubdtype(pos.dtype, jnp.floating) else pos


def _normalize_attention_mask(
    attention_mask: Bool[Array, "..."] | Int[Array, "..."] | None,
    *,
    batch: int,
    q_len: int,
    kv_len: int,
) -> Bool[Array, "batch q kv"] | None:
    if attention_mask is None:
        return None
    m = attention_mask
    if m.dtype != jnp.bool_:
        m = m != 0

    if m.ndim == 4:
        if m.shape[0] != batch or m.shape[2] != q_len or m.shape[3] != kv_len:
            raise ValueError(f"attention_mask must have shape (B, H/1, Q, K); got {m.shape}")
        # Head-specific masks cannot be encoded in a single (B,Q,K) in general; use head 0 for determinism.
        return m[:, 0, :, :]
    if m.ndim == 3:
        if m.shape != (batch, q_len, kv_len):
            raise ValueError(f"attention_mask must have shape (B, Q, K); got {m.shape}")
        return m
    if m.ndim == 2:
        if m.shape != (batch, kv_len):
            raise ValueError(f"2D attention_mask is treated as KV padding mask with shape (B, K); got {m.shape}")
        return jnp.broadcast_to(m[:, None, :], (batch, q_len, kv_len))

    raise ValueError(f"Unsupported attention_mask rank {m.ndim} with shape {m.shape}")


def _normalize_softmax_aux(
    softmax_aux: Float[Array, "..."] | None,
    *,
    num_heads: int,
    num_kv_heads: int,
    dtype: jnp.dtype,
) -> Float[Array, "num_heads num_sinks"] | None:
    if softmax_aux is None:
        return None
    aux = jnp.asarray(softmax_aux, dtype=dtype)
    if aux.ndim == 1:
        # For block-sparse (Splash) attention, `softmax_aux` acts like a per-head
        # attention sink logit (i.e. one extra "sink" entry per head).
        if aux.shape[0] == num_heads:
            return aux[:, None]
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)[:, None]
        return jnp.broadcast_to(aux[None, :], (num_heads, aux.shape[0]))
    if aux.ndim == 2:
        if aux.shape[0] == num_heads:
            return aux
        if aux.shape[0] == num_kv_heads:
            reps = num_heads // num_kv_heads
            return jnp.repeat(aux, repeats=reps, axis=0)
        raise ValueError(
            f"softmax_aux first dim must be num_kv_heads ({num_kv_heads}) or num_heads ({num_heads}); got {aux.shape[0]}"
        )
    raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")


@kernel_registry.register("blocksparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"]
    | Int[Array, "batch num_heads_or_1 seq_len kv_len"]
    | None = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """XLA fallback for block-sparse attention with packed (multi-sequence) support.

    This implementation is a correctness fallback: it materializes the token-level
    mask implied by segment IDs, positions, causal/sliding-window settings (and an
    optional attention_mask), then computes dense attention in JAX/XLA.
    """
    del (
        fused_backward,
        qkv_layouts,
        fwd_params,
        bwd_params,
        mask_builder,
        chunk_size,
        sequence_parallelism_mesh_axis_name,
    )

    if bias is not None:
        raise NotImplementedError("Bias is not supported in blocksparse_attention (XLA fallback)")

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query/key/value must be rank-4 tensors (B, H, T, D)")

    batch, num_heads, q_len, head_dim = query.shape
    _b2, num_kv_heads, kv_len, _d2 = key.shape
    if _b2 != batch:
        raise ValueError(f"batch mismatch: query batch {batch}, key batch {_b2}")
    if value.shape[:3] != (batch, num_kv_heads, kv_len):
        raise ValueError(f"value must have shape (B, Hkv, K, Vd); got {value.shape}")
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    if sliding_window is None:
        window_left = window_right = None
    elif isinstance(sliding_window, int):
        window_left = window_right = int(sliding_window)
    else:
        window_left, window_right = int(sliding_window[0]), int(sliding_window[1])

    q_ids = _normalize_segment_ids(q_segment_ids, which="q")
    kv_ids = _normalize_segment_ids(kv_segment_ids, which="kv")

    if kv_ids is None and q_ids is not None and kv_len == q_len:
        kv_ids = q_ids
    if q_ids is None and kv_ids is not None and kv_len == q_len:
        q_ids = kv_ids

    if q_ids is None:
        q_ids = jnp.ones((batch, q_len), dtype=jnp.int32)
    if kv_ids is None:
        kv_ids = jnp.ones((batch, kv_len), dtype=jnp.int32)

    q_pos = _normalize_positions(q_positions, batch=batch, seqlen=q_len, fill=-1)
    kv_pos = _normalize_positions(kv_positions, batch=batch, seqlen=kv_len, fill=jnp.iinfo(jnp.int32).max)

    q_valid = q_ids >= 0
    kv_valid = kv_ids >= 0

    mask = (q_ids[:, :, None] == kv_ids[:, None, :]) & q_valid[:, :, None] & kv_valid[:, None, :]

    if causal:
        mask = mask & (q_pos[:, :, None] >= kv_pos[:, None, :])

    if window_left is not None or window_right is not None:
        wl = window_left if window_left is not None else jnp.iinfo(jnp.int32).max
        wr = window_right if window_right is not None else jnp.iinfo(jnp.int32).max
        mask = mask & (kv_pos[:, None, :] >= (q_pos[:, :, None] - wl)) & (kv_pos[:, None, :] <= (q_pos[:, :, None] + wr))

    attn_mask = _normalize_attention_mask(attention_mask, batch=batch, q_len=q_len, kv_len=kv_len)
    if attn_mask is not None:
        mask = mask & attn_mask

    row_has_any = jnp.any(mask, axis=-1)

    if softmax_aux is None:
        q_bthd = jnp.transpose(query, (0, 2, 1, 3))
        k_bthd = jnp.transpose(key, (0, 2, 1, 3))
        v_bthd = jnp.transpose(value, (0, 2, 1, 3))
        mask_4d = mask[:, None, :, :]

        out_bthd, _ = dense_attention(
            query=q_bthd,
            key=k_bthd,
            value=v_bthd,
            attention_mask=mask_4d,
            softmax_aux=None,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            dtype=q_bthd.dtype,
            softmax_dtype=None,
            dropout_prob=0.0,
            deterministic=True,
            dropout_rng=None,
            causal=causal,
            sliding_window=None,
            bias=None,
            init_bias=None,
        )

        out_bthd = out_bthd * (row_has_any & q_valid).astype(out_bthd.dtype)[:, :, None, None]
        return jnp.transpose(out_bthd, (0, 2, 1, 3))

    reps = num_heads // num_kv_heads
    if reps != 1:
        key_h = jnp.repeat(key, repeats=reps, axis=1)
        value_h = jnp.repeat(value, repeats=reps, axis=1)
    else:
        key_h = key
        value_h = value

    logits = jnp.einsum("bhtd,bhkd->bhtk", query * softmax_scale, key_h, optimize=True)
    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)

    aux = _normalize_softmax_aux(softmax_aux, num_heads=num_heads, num_kv_heads=num_kv_heads, dtype=logits.dtype)
    assert aux is not None
    sinks = jnp.broadcast_to(aux[None, :, None, :], (batch, num_heads, q_len, aux.shape[-1]))
    combined = jnp.concatenate([logits, sinks], axis=-1)
    probs = jax.nn.softmax(combined.astype(jnp.float32), axis=-1).astype(logits.dtype)
    weights = probs[..., :kv_len]

    out = jnp.einsum("bhtk,bhkd->bhtd", weights, value_h, optimize=True)
    out = out * (row_has_any & q_valid).astype(out.dtype)[:, None, :, None]
    return out
