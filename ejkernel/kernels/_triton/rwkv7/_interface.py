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

"""RWKV-7 recurrent kernel (Triton)."""

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
