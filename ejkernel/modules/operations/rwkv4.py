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

"""RWKV-4 time-mix recurrence operation module."""

from __future__ import annotations

import os
import typing
from typing import Literal

from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RWKV4Config


class RWKV4(Kernel[RWKV4Config, Array]):
    """RWKV-4 time-mix recurrence kernel wrapper.

    Implements the numerically-stable RWKV-4 recurrence as described in
    Flash-Linear-Attention's fused recurrent formulation. The state tracks
    three components (alpha, beta, eps) for numerical stability.

    The recurrence computes:
        tau = max(u + k_t, eps)
        wkv_t = (exp(eps - tau) * alpha + exp(u + k_t - tau) * v_t) /
                (exp(eps - tau) * beta + exp(u + k_t - tau))

    Attributes:
        op_id: Operation identifier ("rwkv4").
    """

    def __init__(self) -> None:
        super().__init__(op_id="rwkv4")

    def get_impl(self, cfg: RWKV4Config):
        platform = detect_platform("rwkv4", cfg.platform)
        return kernel_registry.get("rwkv4", platform=platform, backend=cfg.backend)

    def run(
        self,
        w: Float[Array, "chans"],
        u: Float[Array, "chans"],
        k: Float[Array, "batch seq_len chans"],
        v: Float[Array, "batch seq_len chans"],
        state: Float[Array, "batch three chans"] | None = None,
        *,
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RWKV4Config,
    ) -> (
        Float[Array, "batch seq_len chans"]
        | tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]
    ):
        if platform is not None:
            cfg = RWKV4Config(
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)
        out, final_state = impl(w=w, u=u, k=k, v=v, state=state)
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV4Config, Array]) -> RWKV4Config:
        del inv
        return RWKV4Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV4Config, Array]):
        del inv
        return []


_executor: Executor[RWKV4Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv4"),
    )
)


def rwkv4(
    w: Float[Array, "chans"],
    u: Float[Array, "chans"],
    k: Float[Array, "batch seq_len chans"],
    v: Float[Array, "batch seq_len chans"],
    state: Float[Array, "batch three chans"] | None = None,
    /,
    *,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RWKV4Config | None = None,
) -> Float[Array, "batch seq_len chans"] | tuple[Float[Array, "batch seq_len chans"], Float[Array, "batch three chans"]]:
    """RWKV-4 time-mix recurrence with automatic backend selection.

    Computes the RWKV-4 time-mixing recurrence over a sequence.

    Args:
        w: Time-decay parameter in log space `[C]`. Internally uses `-exp(w)`.
        u: Time-mix bias `[C]`.
        k: Key tensor `[B, T, C]`.
        v: Value tensor `[B, T, C]`.
        state: Optional initial state `[B, 3, C]` containing (alpha, beta, eps).
            If None, initializes with alpha=0, beta=0, eps=-1e30.
        return_state: If True, also return the final state.
        platform: Backend platform override ("triton", "pallas", "cuda", "xla", "auto").
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, C]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is `[B, 3, C]`.
    """
    return _executor(
        RWKV4(),
        w=w,
        u=u,
        k=k,
        v=v,
        state=state,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )
