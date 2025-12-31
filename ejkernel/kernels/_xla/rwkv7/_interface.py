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

"""RWKV-7 recurrent kernel (XLA).

RWKV-7 can be expressed as a DPLR (Diagonal + Low-Rank) state update:

    h_t = diag(exp(w_t)) @ h_{t-1} + a_t (b_t^T h_{t-1}) + k_t v_t^T
    o_t = r_t^T h_t

where `h` has shape [K, V] per head (we store it as [K, V]).

This file provides:
    - `rwkv7`: (a,b) parameterization
    - `rwkv7_mul`: (kk,a) parameterization used by some optimized kernels
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry


def _rwkv7_update(
    h: Array,
    r_t: Array,
    k_t: Array,
    v_t: Array,
    w_t: Array,
    a_t: Array,
    b_t: Array,
) -> tuple[Array, Array]:
    # h: [B,H,K,V]
    # r_t,k_t,w_t,a_t,b_t: [B,H,K]
    # v_t: [B,H,V]
    hb = jnp.einsum("bhk,bhkv->bhv", b_t, h)  # b^T h -> [B,H,V]
    h = h * jnp.exp(w_t)[..., :, None] + a_t[..., :, None] * hb[..., None, :] + k_t[..., :, None] * v_t[..., None, :]
    o_t = jnp.einsum("bhk,bhkv->bhv", r_t, h)
    return h, o_t


def _validate_rwkv7_inputs(r: Array, k: Array, v: Array, w: Array, a: Array, b: Array) -> None:
    if r.ndim != 4 or k.ndim != 4 or w.ndim != 4 or a.ndim != 4 or b.ndim != 4 or v.ndim != 4:
        raise ValueError(
            "Expected r,k,w,a,b rank-4 [B,T,H,K] and v rank-4 [B,T,H,V], "
            f"got r={r.shape}, k={k.shape}, v={v.shape}, w={w.shape}, a={a.shape}, b={b.shape}."
        )
    if r.shape != k.shape or r.shape != w.shape or r.shape != a.shape or r.shape != b.shape:
        raise ValueError(
            f"`r`, `k`, `w`, `a`, and `b` must have identical shapes, "
            f"got r={r.shape}, k={k.shape}, w={w.shape}, a={a.shape}, b={b.shape}."
        )
    if v.shape[:3] != r.shape[:3]:
        raise ValueError(f"`v` must match [B,T,H,*], got v={v.shape}, r={r.shape}.")


def _rwkv7_scan(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    a: Array,
    b: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    if reverse:
        r = r[:, ::-1, :, :]
        k = k[:, ::-1, :, :]
        v = v[:, ::-1, :, :]
        w = w[:, ::-1, :, :]
        a = a[:, ::-1, :, :]
        b = b[:, ::-1, :, :]

    r = r * softmax_scale

    def step(h, xs):
        r_t, k_t, v_t, w_t, a_t, b_t = xs
        return _rwkv7_update(h, r_t, k_t, v_t, w_t, a_t, b_t)

    xs = (
        jnp.swapaxes(r, 0, 1),
        jnp.swapaxes(k, 0, 1),
        jnp.swapaxes(v, 0, 1),
        jnp.swapaxes(w, 0, 1),
        jnp.swapaxes(a, 0, 1),
        jnp.swapaxes(b, 0, 1),
    )
    h_final, oT = jax.lax.scan(step, initial_state, xs)
    o = jnp.swapaxes(oT, 0, 1)
    if reverse:
        o = o[:, ::-1, :, :]
    return o, h_final


def _rwkv7_varlen(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    a: Array,
    b: Array,
    cu_seqlens: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    if r.shape[0] != 1:
        raise ValueError(f"Packed mode expects batch size 1, got {r.shape[0]}.")
    total_tokens = r.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    if initial_state.shape[0] != num_seqs:
        raise ValueError(f"`initial_state` must have shape [N,H,K,V] with N={num_seqs}, got {initial_state.shape}.")

    idx = jnp.arange(total_tokens, dtype=cu_seqlens.dtype)
    seq_id = jnp.searchsorted(cu_seqlens[1:], idx, side="right")
    starts = cu_seqlens[seq_id]
    ends = cu_seqlens[seq_id + 1] - 1
    is_start = idx == starts
    is_end = idx == ends

    if reverse:
        r = r[:, ::-1, :, :]
        k = k[:, ::-1, :, :]
        v = v[:, ::-1, :, :]
        w = w[:, ::-1, :, :]
        a = a[:, ::-1, :, :]
        b = b[:, ::-1, :, :]
        seq_id = seq_id[::-1]
        is_start, is_end = is_end[::-1], is_start[::-1]

    r = r * softmax_scale

    def step(carry, xs):
        h, finals = carry
        r_t, k_t, v_t, w_t, a_t, b_t, sid, start, end = xs
        h = jax.lax.cond(start, lambda _: initial_state[sid][None, ...], lambda _: h, operand=None)
        h, o_t = _rwkv7_update(
            h,
            r_t[None, ...],
            k_t[None, ...],
            v_t[None, ...],
            w_t[None, ...],
            a_t[None, ...],
            b_t[None, ...],
        )
        finals = jax.lax.cond(end, lambda f: f.at[sid].set(h[0]), lambda f: f, finals)
        return (h, finals), o_t[0]

    xs = (
        r[0],
        k[0],
        v[0],
        w[0],
        a[0],
        b[0],
        seq_id.astype(jnp.int32),
        is_start,
        is_end,
    )
    h_init = jnp.zeros((1, *initial_state[0].shape), dtype=initial_state.dtype)
    (h_last, finals), o = jax.lax.scan(step, (h_init, initial_state), xs)
    del h_last
    o = o[None, ...]
    if reverse:
        o = o[:, ::-1, :, :]
    return o, finals


@kernel_registry.register("rwkv7", Platform.XLA, Backend.ANY)
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
    """RWKV-7 DPLR recurrence in JAX/XLA."""
    _validate_rwkv7_inputs(r, k, v, w, a, b)

    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out_dtype = v.dtype
    r_f32 = r.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    w_f32 = w.astype(jnp.float32)
    a_f32 = a.astype(jnp.float32)
    b_f32 = b.astype(jnp.float32)

    if cu_seqlens is None:
        B, _, H, K = r.shape
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv7_scan(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            a_f32,
            b_f32,
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )
    else:
        num_seqs = cu_seqlens.shape[0] - 1
        H, K = r.shape[2], r.shape[3]
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((num_seqs, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv7_varlen(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            a_f32,
            b_f32,
            cu_seqlens.astype(jnp.int32),
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )

    return o_f32.astype(out_dtype), final_state


@kernel_registry.register("rwkv7_mul", Platform.XLA, Backend.ANY)
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
    """RWKV-7 multiplicative parameterization wrapper.

    Uses the same DPLR update as `rwkv7` with:
        a' = kk * a
        b' = -kk
    """
    return rwkv7(
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
