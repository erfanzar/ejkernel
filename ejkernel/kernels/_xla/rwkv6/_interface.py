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

"""RWKV-6 recurrent kernel (XLA).

This matches the semantics of Flash-Linear-Attention's RWKV-6 fused recurrent op:
    - inputs in `[B, T, H, ...]` (head-last) format
    - optional packed variable-length mode via `cu_seqlens`
    - optional reverse recurrence
    - returns `(o, final_state)` where final_state is float32
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry


def _rwkv6_update(
    h: Array,
    q_t: Array,
    k_t: Array,
    v_t: Array,
    w_t: Array,
    u: Array,
) -> tuple[Array, Array]:
    # h: [B, H, K, V]
    # q_t,k_t,w_t: [B, H, K]
    # v_t: [B, H, V]
    kv = k_t[..., :, None] * v_t[..., None, :]  # [B, H, K, V]
    o_t = jnp.einsum("bhk,bhkv->bhv", q_t, h + kv * u[None, :, :, None])  # [B, H, V]
    h_next = h * jnp.exp(w_t)[..., :, None] + kv
    return h_next, o_t


def _rwkv6_scan(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    # Shapes:
    # r,k,w: [B, T, H, K]
    # v:     [B, T, H, V]
    # u:     [H, K]
    if reverse:
        r = r[:, ::-1, :, :]
        k = k[:, ::-1, :, :]
        v = v[:, ::-1, :, :]
        w = w[:, ::-1, :, :]

    r = r * softmax_scale

    def step(h, xs):
        q_t, k_t, v_t, w_t = xs
        h_next, o_t = _rwkv6_update(h, q_t, k_t, v_t, w_t, u)
        return h_next, o_t

    xs = (jnp.swapaxes(r, 0, 1), jnp.swapaxes(k, 0, 1), jnp.swapaxes(v, 0, 1), jnp.swapaxes(w, 0, 1))
    h_final, oT = jax.lax.scan(step, initial_state, xs)  # oT: [T, B, H, V]
    o = jnp.swapaxes(oT, 0, 1)
    if reverse:
        o = o[:, ::-1, :, :]
    return o, h_final


def _validate_rwkv6_inputs(r: Array, k: Array, v: Array, w: Array, u: Array) -> None:
    if r.ndim != 4 or k.ndim != 4 or w.ndim != 4 or v.ndim != 4:
        raise ValueError(
            f"Expected r,k,w rank-4 [B,T,H,K] and v rank-4 [B,T,H,V], got "
            f"r={r.shape}, k={k.shape}, v={v.shape}, w={w.shape}."
        )
    if r.shape != k.shape or r.shape != w.shape:
        raise ValueError(f"`r`, `k`, and `w` must have the same shape, got r={r.shape}, k={k.shape}, w={w.shape}.")
    if v.shape[:3] != r.shape[:3]:
        raise ValueError(f"`v` must match [B,T,H,*], got v={v.shape}, r={r.shape}.")
    if u.ndim != 2 or u.shape[0] != r.shape[2] or u.shape[1] != r.shape[3]:
        raise ValueError(f"`u` must have shape [H,K]={(r.shape[2], r.shape[3])}, got {u.shape}.")


def _rwkv6_varlen(
    r: Array,
    k: Array,
    v: Array,
    w: Array,
    u: Array,
    cu_seqlens: Array,
    *,
    softmax_scale: float,
    initial_state: Array,
    reverse: bool,
) -> tuple[Array, Array]:
    # Packed mode expects B==1 and T==total_tokens.
    if r.shape[0] != 1:
        raise ValueError(f"Packed mode expects batch size 1, got {r.shape[0]}.")
    total_tokens = r.shape[1]
    num_seqs = cu_seqlens.shape[0] - 1
    if initial_state.shape[0] != num_seqs:
        raise ValueError(f"`initial_state` must have shape [N,H,K,V] with N={num_seqs}, got {initial_state.shape}.")

    # Precompute per-token metadata.
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
        seq_id = seq_id[::-1]
        is_start, is_end = is_end[::-1], is_start[::-1]

    r = r * softmax_scale

    def step(carry, xs):
        h, finals = carry
        q_t, k_t, v_t, w_t, sid, start, end = xs

        h = jax.lax.cond(start, lambda _: initial_state[sid][None, ...], lambda _: h, operand=None)
        h, o_t = _rwkv6_update(h, q_t[None, ...], k_t[None, ...], v_t[None, ...], w_t[None, ...], u)
        finals = jax.lax.cond(end, lambda f: f.at[sid].set(h[0]), lambda f: f, finals)
        return (h, finals), o_t[0]

    # Scan over packed tokens (batch dim is 1, so squeeze it).
    xs = (
        r[0],
        k[0],
        v[0],
        w[0],
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


@kernel_registry.register("rwkv6", Platform.XLA, Backend.ANY)
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
    """RWKV-6 recurrence in JAX/XLA.

    Args:
        r: Query/receptance `[B, T, H, K]`.
        k: Key `[B, T, H, K]`.
        v: Value `[B, T, H, V]`.
        w: Log decay `[B, T, H, K]`.
        u: Bonus `[H, K]`.
        softmax_scale: Optional scale for `r`; defaults to `K**-0.5`.
        initial_state: Optional initial state `[B,H,K,V]` (or `[N,H,K,V]` in packed mode).
        reverse: If True, process the recurrence in reverse order.
        cu_seqlens: Optional packed variable-length description (FlashAttention-style).

    Returns:
        o: Output `[B, T, H, V]` (dtype matches `v`).
        final_state: `[B,H,K,V]` (or `[N,H,K,V]` in packed mode), float32.
    """
    _validate_rwkv6_inputs(r, k, v, w, u)

    if softmax_scale is None:
        softmax_scale = r.shape[-1] ** -0.5

    out_dtype = v.dtype
    r_f32 = r.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)
    w_f32 = w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    if cu_seqlens is None:
        B, _, H, K = r.shape
        V = v.shape[-1]
        if initial_state is None:
            initial_state_f32 = jnp.zeros((B, H, K, V), dtype=jnp.float32)
        else:
            initial_state_f32 = initial_state.astype(jnp.float32)
        o_f32, final_state = _rwkv6_scan(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            u_f32,
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
        o_f32, final_state = _rwkv6_varlen(
            r_f32,
            k_f32,
            v_f32,
            w_f32,
            u_f32,
            cu_seqlens.astype(jnp.int32),
            softmax_scale=float(softmax_scale),
            initial_state=initial_state_f32,
            reverse=reverse,
        )

    return o_f32.astype(out_dtype), final_state
