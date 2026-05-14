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

"""tile-lang RWKV-4 time-mix attention registration.

Exposes the forward and backward kernels through the ``jax.custom_vjp``
mechanism defined in ``_impl.py``.  Both passes are native tile-lang
kernels; no XLA fallback is used.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._impl import rwkv4_tilelang


@kernel_registry.register("rwkv4", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len chans"],
    Float[Array, "batch three chans"],
]:
    """Tile-lang RWKV-4 time-mix forward (channel-parallel scan).

    Registered as ``"rwkv4"`` on ``Platform.TILELANG / Backend.GPU``.

    Supports full forward + backward via native tile-lang kernels. The
    backward materialises every hidden state and runs a reverse-time adjoint
    scan.

    Args:
        w: ``(chans,)`` time-decay in log space (used as ``-exp(w)`` by the
            recurrence — must already be in additive log form).
        u: ``(chans,)`` time-mix bonus applied at the current token only.
        k: ``(batch, seq_len, chans)`` keys.
        v: ``(batch, seq_len, chans)`` values.
        state: optional fp32 ``(batch, 3, chans)`` initial state
            ``(alpha, beta, eps)``; defaults to all-zeros / ``eps=-1e30``.

    Returns:
        ``(wkv, final_state)`` — ``wkv`` is ``(batch, seq_len, chans)`` in the
        input dtype; ``final_state`` is fp32 ``(batch, 3, chans)``.

    Raises:
        RuntimeError: if tilelang or jax_tvm_ffi is not available.
        ValueError: if tensor shapes do not satisfy ``(B, S, C)`` / ``(C,)``
            constraints.
    """
    return rwkv4_tilelang(w, u, k, v, state=state)


__all__ = ["rwkv4"]
