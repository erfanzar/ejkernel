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

"""RWKV-6 linear attention recurrence implementation using Triton.

This module provides a GPU-accelerated implementation of the RWKV-6 linear
attention mechanism. RWKV-6 extends RWKV-4 with a multi-head architecture
and per-timestep time-decay parameters, enabling more expressive sequence
modeling while maintaining O(N) complexity.

Key improvements over RWKV-4:
- Multi-head architecture for parallel processing of different subspaces
- Data-dependent decay (w) that varies per timestep and head
- Separate query-key (K) and value (V) head dimensions
- Support for grouped-query attention patterns

The RWKV-6 recurrence computes:
    h_t = diag(exp(w_t)) * h_{t-1} + k_t^T * v_t
    o_t = softmax_scale * (r_t * (u * (k_t^T * v_t) + h_{t-1}))

Key components:
- Receptance (r): Analogous to query in standard attention
- Key (k): Used with value to form outer product updates
- Value (v): The values to be aggregated
- Log-decay (w): Per-timestep, per-head decay rates in log-space
- Bonus (u): Direct contribution from current timestep

Key features:
- O(N) time complexity for sequence processing
- Multi-head attention with independent decay per head
- Variable sequence length support via cu_seqlens
- Bidirectional processing via reverse flag
- Custom Triton kernel for GPU acceleration

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.rwkv6 import rwkv6
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> r = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> w = jnp.zeros((batch, seq_len, num_heads, head_dim))
    >>> u = jnp.zeros((num_heads, head_dim))
    >>>
    >>> output, final_state = rwkv6(r, k, v, w, u)

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
from ..._xla.rwkv6 import rwkv6 as xla_rwkv6
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    softmax_scale: float | None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
):
    """Forward pass for RWKV-6 recurrence in a custom VJP.

    Computes the RWKV-6 linear attention and saves residuals for backward pass.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        u: Bonus tensor of shape `[H, K]`.
        softmax_scale: Scaling factor for attention computation.
        initial_state: Optional initial hidden state of shape `[B, H, K, V]`.
        reverse: Whether to process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.

    Returns:
        A tuple containing (output, final_state) and residuals for backward.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out, final_state = fwd_triton_impl(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    residual = (r, k, v, w, u, softmax_scale, initial_state, reverse, cu_seqlens)
    return (out, final_state), residual


def _bwd_call(
    softmax_scale: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residual,
    grads,
):
    """Backward pass for RWKV-6 recurrence in a custom VJP.

    Computes gradients with respect to all inputs using JAX autodiff
    through the XLA reference implementation.

    Args:
        softmax_scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        residual: Tensors saved from the forward pass.
        grads: A tuple containing gradients (do, dht) of output and final state.

    Returns:
        A tuple of gradients (dr, dk, dv, dw, du, dh0) for all inputs.
    """
    (r, k, v, w, u, softmax_scale_saved, initial_state, reverse_saved, cu_seqlens_saved) = residual
    do, dht = grads
    del reverse_saved, cu_seqlens_saved

    if softmax_scale is None:
        softmax_scale = softmax_scale_saved

    def f(r_, k_, v_, w_, u_, h0_):
        return xla_rwkv6(
            r=r_,
            k=k_,
            v=v_,
            w=w_,
            u=u_,
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

    (o_ref, ht_ref), vjp = jax.vjp(f, r, k, v, w, u, h0)
    del o_ref, ht_ref
    dr, dk, dv, dw, du, dh0 = vjp((do, dht))
    if h0_in is None:
        dh0 = None
    return dr, dk, dv, dw, du, dh0


@partial(jax.custom_vjp, nondiff_argnums=(5, 7, 8))
@partial(jax.jit, static_argnums=(5, 7))
def _rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """Core JIT-compiled RWKV-6 function with custom VJP.

    This internal function directly calls the Triton implementation and is
    registered with JAX's custom differentiation system for memory-efficient
    gradient computation.

    Args:
        r: Receptance tensor of shape `[B, T, H, K]`.
        k: Key tensor of shape `[B, T, H, K]`.
        v: Value tensor of shape `[B, T, H, V]`.
        w: Log-decay tensor of shape `[B, T, H, K]`.
        u: Bonus tensor of shape `[H, K]`.
        softmax_scale: Scaling factor (static argument).
        initial_state: Optional initial hidden state of shape `[B, H, K, V]`.
        reverse: Process in reverse order (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.

    Returns:
        A tuple containing:
            - output: The attention output tensor of shape `[B, T, H, V]`.
            - final_state: The final hidden state of shape `[B, H, K, V]`.
    """
    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5
    return fwd_triton_impl(
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=float(softmax_scale),
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


_rwkv6.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv6", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len num_heads v_head_dim"],
    Float[Array, "... num_heads qk_head_dim v_head_dim"],
]:
    """RWKV-6 linear attention recurrence (Triton GPU implementation).

    Args:
        r: Receptance tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        w: Log decay tensor `[B, T, H, K]`.
        u: Bonus tensor `[H, K]`.
        softmax_scale: Optional scale for receptance.
        initial_state: Optional initial state `[B, H, K, V]`.
        reverse: Process sequence in reverse order.
        cu_seqlens: Cumulative sequence lengths for packed mode.

    Returns:
        Tuple of (output `[B, T, H, V]`, final_state `[B, H, K, V]`).
    """
    return _rwkv6(
        r,
        k,
        v,
        w,
        u,
        softmax_scale,
        initial_state,
        reverse,
        cu_seqlens,
    )
