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

"""RWKV-6 recurrence operation module."""

from __future__ import annotations

import os
import typing
from typing import Literal

from jaxtyping import Array, Float, Int

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
from .configs import RWKV6Config


class RWKV6(Kernel[RWKV6Config, Array]):
    """RWKV-6 recurrence kernel wrapper.

    Implements the RWKV-6 linear attention recurrence with multi-head support
    and optional variable-length sequence packing. The recurrence computes:

        kv_t = k_t^T @ v_t
        o_t = r_t^T @ (h + kv_t * u)
        h_{t+1} = h * exp(w_t) + kv_t

    where h is the state matrix of shape [H, K, V] per batch element.

    Attributes:
        op_id: Operation identifier ("rwkv6").
    """

    def __init__(self) -> None:
        super().__init__(op_id="rwkv6")

    def get_impl(self, cfg: RWKV6Config):
        platform = detect_platform("rwkv6", cfg.platform)
        return kernel_registry.get("rwkv6", platform=platform, backend=cfg.backend)

    def run(
        self,
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
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RWKV6Config,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        if platform is not None:
            cfg = RWKV6Config(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)

        impl = self.get_impl(cfg)
        out, final_state = impl(
            r=r,
            k=k,
            v=v,
            w=w,
            u=u,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV6Config, Array]) -> RWKV6Config:
        del inv
        return RWKV6Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV6Config, Array]):
        del inv
        return []


_executor: Executor[RWKV6Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv6"),
    )
)


def rwkv6(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    u: Float[Array, "num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RWKV6Config | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-6 recurrence with automatic backend selection.

    Computes the RWKV-6 linear attention recurrence over a sequence with
    multi-head support and optional variable-length packing.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        w: Log decay tensor `[B, T, H, K]`.
        u: Bonus tensor (per-head) `[H, K]`.
        softmax_scale: Optional scale for receptance; defaults to `K**-0.5`.
        initial_state: Optional initial state `[B, H, K, V]` (or `[N, H, K, V]`
            in packed mode).
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences
            (FlashAttention-style), shape `[N+1]`.
        return_state: If True, also return the final state.
        platform: Backend platform override.
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, H, V]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is float32.
    """
    return _executor(
        RWKV6(),
        r=r,
        k=k,
        v=v,
        w=w,
        u=u,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )
