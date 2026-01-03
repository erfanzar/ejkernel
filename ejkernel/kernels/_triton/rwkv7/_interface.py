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

"""RWKV-7 Diagonal Plus Low Rank (DPLR) recurrence implementation using Triton.

This module provides a GPU-accelerated implementation of the RWKV-7 linear
attention mechanism. RWKV-7 introduces a Diagonal Plus Low-Rank (DPLR)
parameterization for the state transition matrix, enabling more expressive
state dynamics while maintaining O(N) complexity.

Key improvements over RWKV-6:
- DPLR state transition: Combines diagonal decay with low-rank updates
- Enhanced expressiveness: Can model richer state transitions
- Flexible parameterization: Supports both additive (a, b) and multiplicative
  (kk, a) forms

The RWKV-7 DPLR recurrence computes:
    h_t = diag(exp(w_t)) * (h_{t-1} + a_t^T * (b_t @ h_{t-1})) + k_t^T * v_t
    o_t = softmax_scale * (r_t @ h_{t-1})

The low-rank update (a, b) allows the model to selectively modify the hidden
state based on learned projections, providing more flexible information routing
than simple diagonal decay.

Key components:
- Receptance (r): Query-like projection for reading from state
- Log-decay (w): Per-timestep, per-head diagonal decay rates
- Key (k): Used with value for rank-1 state updates
- Value (v): The values to be accumulated in state
- Low-rank a: First component of DPLR update (rank-1 outer product)
- Low-rank b: Second component of DPLR update (projection vector)

Alternative multiplicative parameterization (rwkv7_mul):
- kk: Multiplicative scaling factor
- a: Base update vector
- Computes a' = kk * a, b' = -kk internally

Key features:
- O(N) time complexity for sequence processing
- DPLR state dynamics for enhanced expressiveness
- Variable sequence length support via cu_seqlens
- Bidirectional processing via reverse flag
- Custom Triton kernel for GPU acceleration

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.rwkv7 import rwkv7
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> a = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> b = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = rwkv7(r, w, k, v, a, b)

Reference:
    Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence
    https://arxiv.org/abs/2404.05892
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ..._xla.rwkv7 import rwkv7 as xla_rwkv7
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    softmax_scale: float | None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
):
    """Forward pass for RWKV-7 DPLR recurrence in a custom VJP.

    Computes the RWKV-7 recurrence with DPLR state transition and saves
    residuals for backward pass.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        a: Low-rank update vector of shape `[B, T, H, K]`.
        b: Low-rank projection vector of shape `[B, T, H, K]`.
        softmax_scale: Scaling factor for receptance.
        initial_state: Optional initial hidden state `[B, H, K, V]`.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.

    Returns:
        A tuple containing (output, final_state) and residuals for backward.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    out, final_state = fwd_triton_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    residual = (r, w, k, v, a, b, softmax_scale, initial_state, reverse, cu_seqlens)
    return (out, final_state), residual


def _bwd_call(
    softmax_scale: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residual,
    grads,
):
    """Backward pass for RWKV-7 DPLR recurrence in a custom VJP.

    Computes gradients with respect to all inputs using JAX autodiff
    through the XLA reference implementation.

    Args:
        softmax_scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        residual: Tensors saved from the forward pass.
        grads: A tuple containing gradients (do, dht) of output and final state.

    Returns:
        A tuple of gradients (dr, dw, dk, dv, da, db, dh0) for all inputs.
    """
    (r, w, k, v, a, b, softmax_scale_saved, initial_state, reverse_saved, cu_seqlens_saved) = residual
    do, dht = grads
    del reverse_saved, cu_seqlens_saved

    if softmax_scale is None:
        softmax_scale = softmax_scale_saved

    def f(r_, w_, k_, v_, a_, b_, h0_):
        return xla_rwkv7(
            r=r_,
            w=w_,
            k=k_,
            v=v_,
            a=a_,
            b=b_,
            softmax_scale=softmax_scale,
            initial_state=h0_,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )

    if initial_state is None:
        if cu_seqlens is None:
            B, _, H, K = r.shape
            V = v.shape[-1]
            h0 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        else:
            N = cu_seqlens.shape[0] - 1
            H, K = r.shape[2], r.shape[3]
            V = v.shape[-1]
            h0 = jnp.zeros((N, H, K, V), dtype=jnp.float32)
        h0_in = None
    else:
        h0 = initial_state
        h0_in = initial_state

    (o_ref, ht_ref), vjp = jax.vjp(f, r, w, k, v, a, b, h0)
    del o_ref, ht_ref
    dr, dw, dk, dv, da, db, dh0 = vjp((do, dht))
    if h0_in is None:
        dh0 = None
    return dr, dw, dk, dv, da, db, dh0


@partial(jax.custom_vjp, nondiff_argnums=(6, 8, 9))
@partial(jax.jit, static_argnums=(6, 8))
def _rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Core JIT-compiled RWKV-7 function with a custom VJP.

    This is an internal function that directly calls the Triton implementation
    and is registered with JAX's custom differentiation system.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        a: Low-rank update vector of shape `[B, T, H, K]`.
        b: Low-rank projection vector of shape `[B, T, H, K]`.
        softmax_scale: Scaling factor for receptance (static argument).
        initial_state: Optional initial hidden state `[B, H, K, V]`.
        reverse: If True, process sequence in reverse (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.

    Returns:
        A tuple containing:
            - output: The attention output tensor of shape `[B, T, H, V]`.
            - final_state: The final hidden state of shape `[B, H, K, V]`.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    return fwd_triton_impl(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


_rwkv7.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv7", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 DPLR recurrence (a,b) (Triton GPU implementation).

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        a: Low-rank update vector `[B, T, H, K]`.
        b: Low-rank projection vector `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv7(r, w, k, v, a, b, softmax_scale, initial_state, reverse, cu_seqlens)


@kernel_registry.register("rwkv7_mul", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-7 multiplicative (kk, a) parameterization (Triton GPU implementation).

    Converts (kk, a) to standard DPLR form: a' = kk * a, b' = -kk.

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        kk: Multiplicative factor `[B, T, H, K]`.
        a: Low-rank update base `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv7(
        r=r,
        w=w,
        k=k,
        v=v,
        a=kk * a,
        b=-kk,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
