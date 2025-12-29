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

"""RWKV-4 recurrent time-mix kernel (XLA).

Implements the numerically-stable RWKV-4 recurrence used in Flash-Linear-Attention's
`fla/ops/rwkv4/fused_recurrent.py`, but as a pure JAX/XLA implementation.

State layout follows the common RWKV-4 formulation:
    state[:, 0, :] = alpha
    state[:, 1, :] = beta
    state[:, 2, :] = eps

Shapes:
    w:     [C]   (time_decay parameter in log space; internally uses `-exp(w)`)
    u:     [C]
    k, v:  [B, T, C]
    state: [B, 3, C]
Returns:
    wkv:        [B, T, C]
    final_state:[B, 3, C]   (float32)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


def _rwkv4_step(
    carry: tuple[Array, Array, Array],
    x: tuple[Array, Array],
    *,
    w: Array,
    u: Array,
) -> tuple[tuple[Array, Array, Array], Array]:
    alpha, beta, eps = carry
    kt, vt = x

    ukt = u + kt
    tau = jnp.maximum(ukt, eps)
    e1a = jnp.exp(eps - tau)
    e2a = jnp.exp(ukt - tau)
    wkv = (e1a * alpha + e2a * vt) / (e1a * beta + e2a)

    w_eps = w + eps
    eps_next = jnp.maximum(w_eps, kt)
    e1b = jnp.exp(w_eps - eps_next)
    e2b = jnp.exp(kt - eps_next)
    alpha_next = e1b * alpha + e2b * vt
    beta_next = e1b * beta + e2b

    return (alpha_next, beta_next, eps_next), wkv


@kernel_registry.register("rwkv4", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 recurrent time-mix.

    Args:
        w: Time-decay parameter in log space (will use `-exp(w)` internally).
        u: Time-mix bias.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` (alpha, beta, eps).
            If None, initializes alpha=0, beta=0, eps=-1e30.

    Returns:
        wkv: `[B, T, C]` (dtype matches `v`).
        final_state: `[B, 3, C]` (float32).
    """
    if k.shape != v.shape:
        raise ValueError(f"`k` and `v` must have the same shape, got k={k.shape}, v={v.shape}.")
    if k.ndim != 3:
        raise ValueError(f"`k` must be rank-3 [B, T, C], got shape {k.shape}.")
    if w.ndim != 1 or u.ndim != 1:
        raise ValueError(f"`w` and `u` must be rank-1 [C], got w={w.shape}, u={u.shape}.")
    if w.shape[0] != k.shape[-1] or u.shape[0] != k.shape[-1]:
        raise ValueError(f"Channel dim mismatch: C={k.shape[-1]}, w={w.shape}, u={u.shape}.")

    orig_dtype = v.dtype
    w_f32 = -jnp.exp(w.astype(jnp.float32))
    u_f32 = u.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    bsz, _, chans = k.shape
    if state is None:
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
    else:
        if state.shape != (bsz, 3, chans):
            raise ValueError(f"`state` must have shape [B, 3, C]={(bsz, 3, chans)}, got {state.shape}.")
        alpha0 = state[:, 0, :].astype(jnp.float32)
        beta0 = state[:, 1, :].astype(jnp.float32)
        eps0 = state[:, 2, :].astype(jnp.float32)

    xs = (jnp.swapaxes(k_f32, 0, 1), jnp.swapaxes(v_f32, 0, 1))  # [T, B, C]
    (alphaT, betaT, epsT), wkvT = jax.lax.scan(
        lambda carry, x: _rwkv4_step(carry, x, w=w_f32, u=u_f32),
        (alpha0, beta0, eps0),
        xs,
    )
    wkv = jnp.swapaxes(wkvT, 0, 1).astype(orig_dtype)  # [B, T, C]
    final_state = jnp.stack([alphaT, betaT, epsT], axis=1)  # [B, 3, C]
    return wkv, final_state
