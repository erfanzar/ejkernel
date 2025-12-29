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

"""RWKV-6 recurrent kernel (Triton)."""

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
