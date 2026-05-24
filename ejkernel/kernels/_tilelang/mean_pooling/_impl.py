# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""JAX glue around the tile-lang mean-pooling prim_funcs.

This module provides:
- Kernel compilation caches (one per forward/backward variant, keyed by
  ``(batch, seq_len, hidden_dim, block_s, block_d, dtype_str)``).
- ``_mean_pool_core`` / ``_mean_pool_varlen_core``: ``jax.custom_vjp``
  primitives that dispatch to the appropriate compiled FFI callable.
- ``mean_pooling_tilelang``: the public entry-point used by ``_interface.py``.

Thread safety: all cache lookups and insertions are serialised with
``_LOCK`` (``threading.Lock``), so compilation happens at most once per
unique key even under concurrent JAX tracing.
"""

from __future__ import annotations

import threading

import jax
import jax.numpy as jnp

from ejkernel.callib._tilelang_call import build_tilelang_call
from ejkernel.callib._tilelang_ffi import has_tilelang_ffi_support

from ._kernel import make_bwd_prim_func, make_fwd_prim_func, make_varlen_bwd_prim_func, make_varlen_fwd_prim_func

_DEFAULT_COMPILE_FLAGS: tuple[str, ...] = ("-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",)


_FWD_CACHE: dict[tuple, callable] = {}
_BWD_CACHE: dict[tuple, callable] = {}
_VARLEN_FWD_CACHE: dict[tuple, callable] = {}
_VARLEN_BWD_CACHE: dict[tuple, callable] = {}
_LOCK = threading.Lock()


def _pick_tile(hidden_dim: int) -> tuple[int, int]:
    """Select ``(BLOCK_S, BLOCK_D)`` tile sizes for mean-pooling kernels.

    The heuristic balances CTA occupancy against launch overhead.  Larger
    ``BLOCK_S`` amortises the per-CTA fixed cost over more sequence tokens,
    while ``BLOCK_D`` controls the register pressure along the hidden axis.
    These values were chosen from an H100 benchmark sweep.

    Args:
        hidden_dim: Feature dimension of the input tensor.

    Returns:
        A ``(block_s, block_d)`` tuple: currently ``(256, 64)`` for
        ``hidden_dim <= 64`` and ``(256, 128)`` for larger hidden dims.
    """
    if hidden_dim <= 64:
        return 256, 64
    if hidden_dim <= 256:
        return 256, 128
    return 256, 128


def _get_fwd(batch, seq_len, hidden_dim, dtype):
    """Retrieve (compiling on first call) the mean-pool forward FFI callable.

    Results are cached under ``(batch, seq_len, hidden_dim, block_s, block_d,
    dtype_str)``.  The output ``ShapeDtypeStruct`` is ``(batch, hidden_dim)``
    in *dtype*.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        hidden_dim: Feature dimension.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(X) -> Y``.
    """
    bs, bd = _pick_tile(hidden_dim)
    key = (batch, seq_len, hidden_dim, bs, bd, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_fwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            block_s=bs,
            block_d=bd,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, hidden_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _FWD_CACHE[key] = ffi
        return ffi


def _get_bwd(batch, seq_len, hidden_dim, dtype):
    """Retrieve (compiling on first call) the mean-pool backward FFI callable.

    The output ``ShapeDtypeStruct`` is ``(batch, seq_len, hidden_dim)`` in
    *dtype*.  The backward expects the upstream gradient ``dY`` in float32.

    Args:
        batch: Batch size.
        seq_len: Sequence length.
        hidden_dim: Feature dimension.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(dY_f32) -> dX``.
    """
    bs, bd = _pick_tile(hidden_dim)
    key = (batch, seq_len, hidden_dim, bs, bd, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_bwd_prim_func(
            batch=batch,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            block_s=bs,
            block_d=bd,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((batch, seq_len, hidden_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _BWD_CACHE[key] = ffi
        return ffi


def _get_varlen_fwd(total_tokens, num_seqs, hidden_dim, dtype):
    """Retrieve (compiling on first call) the packed mean-pool forward FFI callable.

    The output ``ShapeDtypeStruct`` is ``(num_seqs, hidden_dim)`` in *dtype*.

    Args:
        total_tokens: Total token count across all sequences.
        num_seqs: Number of sequences.
        hidden_dim: Feature dimension.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(X, cu_seqlens) -> Y``.
    """
    bs, bd = _pick_tile(hidden_dim)
    key = (total_tokens, num_seqs, hidden_dim, bs, bd, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _VARLEN_FWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_varlen_fwd_prim_func(
            total_tokens=total_tokens,
            num_seqs=num_seqs,
            hidden_dim=hidden_dim,
            block_s=bs,
            block_d=bd,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((num_seqs, hidden_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _VARLEN_FWD_CACHE[key] = ffi
        return ffi


def _get_varlen_bwd(total_tokens, num_seqs, hidden_dim, dtype):
    """Retrieve (compiling on first call) the packed mean-pool backward FFI callable.

    The output ``ShapeDtypeStruct`` is ``(total_tokens, hidden_dim)`` in *dtype*.
    The backward expects the upstream gradient ``dY`` in float32.

    Args:
        total_tokens: Total token count across all sequences.
        num_seqs: Number of sequences.
        hidden_dim: Feature dimension.
        dtype: Activation dtype.

    Returns:
        A compiled FFI callable ``ffi(dY_f32, cu_seqlens) -> dX``.
    """
    bs, bd = _pick_tile(hidden_dim)
    key = (total_tokens, num_seqs, hidden_dim, bs, bd, str(jnp.dtype(dtype)))
    with _LOCK:
        cached = _VARLEN_BWD_CACHE.get(key)
        if cached is not None:
            return cached
        prim = make_varlen_bwd_prim_func(
            total_tokens=total_tokens,
            num_seqs=num_seqs,
            hidden_dim=hidden_dim,
            block_s=bs,
            block_d=bd,
            dtype=dtype,
        )
        ffi = build_tilelang_call(
            prim,
            output_shape_dtype=jax.ShapeDtypeStruct((total_tokens, hidden_dim), dtype),
            compile_flags=_DEFAULT_COMPILE_FLAGS,
        )
        _VARLEN_BWD_CACHE[key] = ffi
        return ffi


@jax.custom_vjp
def _mean_pool_core(x: jax.Array) -> jax.Array:
    """Mean-pool a padded ``(B, S, D)`` array; registered with a VJP via ``defvjp``."""
    if x.ndim != 3:
        raise ValueError("mean_pooling (tile-lang) expects (B, S, D); got " + str(x.shape))
    B, S, D = x.shape
    ffi = _get_fwd(B, S, D, x.dtype)
    return ffi(x)


def _fwd(x):
    """VJP primal for ``_mean_pool_core``. Returns ``(output, residual)``."""
    B, S, D = x.shape
    ffi = _get_fwd(B, S, D, x.dtype)
    return ffi(x), (x,)


def _bwd(residual, dy):
    """VJP backward for ``_mean_pool_core``. Casts *dy* to float32 before dispatch."""
    (x,) = residual
    B, S, D = x.shape
    ffi = _get_bwd(B, S, D, x.dtype)
    dy_f32 = dy.astype(jnp.float32)
    dx = ffi(dy_f32)
    return (dx,)


_mean_pool_core.defvjp(_fwd, _bwd)


@jax.custom_vjp
def _mean_pool_varlen_core(x: jax.Array, cu_seqlens: jax.Array) -> jax.Array:
    """Mean-pool a packed ``(T, D)`` array; registered with a VJP via ``defvjp``.

    ``cu_seqlens`` has shape ``(num_seqs + 1,)`` and dtype int32.
    """
    if x.ndim != 2:
        raise ValueError("mean_pooling (tile-lang varlen) expects (T, D); got " + str(x.shape))
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be rank-1; got " + str(cu_seqlens.shape))
    T, D = x.shape
    B = cu_seqlens.shape[0] - 1
    ffi = _get_varlen_fwd(T, B, D, x.dtype)
    return ffi(x, cu_seqlens)


def _varlen_fwd(x, cu_seqlens):
    """VJP primal for ``_mean_pool_varlen_core``. Residual stores ``cu_seqlens`` and shape."""
    T, D = x.shape
    B = cu_seqlens.shape[0] - 1
    ffi = _get_varlen_fwd(T, B, D, x.dtype)
    return ffi(x, cu_seqlens), (cu_seqlens, x.shape)


def _varlen_bwd(residual, dy):
    """VJP backward for ``_mean_pool_varlen_core``. Returns ``(dX, None)``."""
    cu_seqlens, shape = residual
    T, D = shape
    B = cu_seqlens.shape[0] - 1
    ffi = _get_varlen_bwd(T, B, D, dy.dtype)
    dy_f32 = dy.astype(jnp.float32)
    dx = ffi(dy_f32, cu_seqlens)
    return dx, None


_mean_pool_varlen_core.defvjp(_varlen_fwd, _varlen_bwd)


def mean_pooling_tilelang(x: jax.Array, cu_seqlens: jax.Array | None = None) -> jax.Array:
    """Tile-lang mean-pooling over the sequence axis (forward + differentiable backward).

    Dispatches to either the padded or packed kernel depending on whether
    *cu_seqlens* is provided.

    Args:
        x: Input activations.  Either ``(batch, seq_len, hidden_dim)`` for the
            padded variant, or ``(total_tokens, hidden_dim)`` for the packed
            (variable-length) variant.
        cu_seqlens: Cumulative sequence lengths of shape ``(num_seqs + 1,)``
            and dtype int32.  Required when *x* is packed.  Must be ``None``
            when *x* is padded.

    Returns:
        ``(batch, hidden_dim)`` float array of per-sequence means, in the same
        dtype as *x*.

    Raises:
        RuntimeError: If ``tilelang`` or ``jax_tvm_ffi`` are not installed.
    """
    if not has_tilelang_ffi_support():
        raise RuntimeError("tile-lang mean_pooling requires both `tilelang` and `jax_tvm_ffi`.")
    if cu_seqlens is not None:
        return _mean_pool_varlen_core(x, cu_seqlens)
    return _mean_pool_core(x)
