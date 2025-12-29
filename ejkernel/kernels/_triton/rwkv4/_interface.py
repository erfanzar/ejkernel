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

"""RWKV-4 recurrent time-mix kernel (Triton)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ..._xla.rwkv4 import rwkv4 as xla_rwkv4
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None,
):
    state_was_none = state is None
    if state is None:
        bsz, _, chans = k.shape
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
        state = jnp.stack([alpha0, beta0, eps0], axis=1)

    w_neg = -jnp.exp(w.astype(jnp.float32))
    o, final_state = fwd_triton_impl(w_neg, u.astype(jnp.float32), k, v, state.astype(jnp.float32))
    residual = (w, u, k, v, state, state_was_none)
    return (o, final_state), residual


def _bwd_call(
    residual,
    grads,
):
    (w, u, k, v, state, state_was_none) = residual
    do, dstate = grads

    def f(w_, u_, k_, v_, state_):
        return xla_rwkv4(w_, u_, k_, v_, state_)

    (o_ref, state_ref), vjp = jax.vjp(f, w, u, k, v, state)
    del o_ref, state_ref
    dw, du, dk, dv, dstate_in = vjp((do, dstate))
    if state_was_none:
        dstate_in = None
    return dw, du, dk, dv, dstate_in


@partial(jax.custom_vjp)
def _rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    if state is None:
        bsz, _, chans = k.shape
        alpha0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        beta0 = jnp.zeros((bsz, chans), dtype=jnp.float32)
        eps0 = jnp.full((bsz, chans), -1e30, dtype=jnp.float32)
        state = jnp.stack([alpha0, beta0, eps0], axis=1)

    w_neg = -jnp.exp(w.astype(jnp.float32))
    return fwd_triton_impl(w_neg, u.astype(jnp.float32), k, v, state.astype(jnp.float32))


_rwkv4.defvjp(_fwd_call, _bwd_call)


@kernel_registry.register("rwkv4", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 time-mix recurrence (Triton GPU implementation).

    Args:
        w: Time-decay parameter in log space `[C]`.
        u: Time-mix bias `[C]`.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` (alpha, beta, eps).

    Returns:
        Tuple of (output `[B, T, C]`, final_state `[B, 3, C]`).
    """
    return _rwkv4(w, u, k, v, state)
