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

"""RWKV-7 recurrence operation modules."""

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
from .configs import RWKV7Config, RWKV7MulConfig


class RWKV7(Kernel[RWKV7Config, Array]):
    """RWKV-7 (a,b) DPLR recurrence wrapper.

    Implements the RWKV-7 Diagonal + Low-Rank (DPLR) state update:

        h_t = diag(exp(w_t)) @ h_{t-1} + a_t (b_t^T h_{t-1}) + k_t v_t^T
        o_t = r_t^T @ h_t

    where h is the state matrix of shape [H, K, V] per batch element.
    This is the (a, b) parameterization where a and b are explicit inputs.

    Attributes:
        op_id: Operation identifier ("rwkv7").
    """

    def __init__(self) -> None:
        super().__init__(op_id="rwkv7")

    def get_impl(self, cfg: RWKV7Config):
        platform = detect_platform("rwkv7", cfg.platform)
        return kernel_registry.get("rwkv7", platform=platform, backend=cfg.backend)

    def run(
        self,
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
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RWKV7Config,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        if platform is not None:
            cfg = RWKV7Config(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)

        impl = self.get_impl(cfg)
        out, final_state = impl(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV7Config, Array]) -> RWKV7Config:
        del inv
        return RWKV7Config(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV7Config, Array]):
        del inv
        return []


class RWKV7Mul(Kernel[RWKV7MulConfig, Array]):
    """RWKV-7 multiplicative (kk, a) parameterization wrapper.

    A reparameterization of RWKV-7 using (kk, a) inputs instead of (a, b).
    Internally converts to the standard DPLR form via:

        a' = kk * a
        b' = -kk

    This is often used by optimized kernel implementations.

    Attributes:
        op_id: Operation identifier ("rwkv7_mul").
    """

    def __init__(self) -> None:
        super().__init__(op_id="rwkv7_mul")

    def get_impl(self, cfg: RWKV7MulConfig):
        platform = detect_platform("rwkv7_mul", cfg.platform)
        return kernel_registry.get("rwkv7_mul", platform=platform, backend=cfg.backend)

    def run(
        self,
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
        return_state: bool = False,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RWKV7MulConfig,
    ) -> (
        Float[Array, "batch seq_len num_heads v_head_dim"]
        | tuple[
            Float[Array, "batch seq_len num_heads v_head_dim"],
            Float[Array, "... num_heads qk_head_dim v_head_dim"],
        ]
    ):
        if platform is not None:
            cfg = RWKV7MulConfig(platform=platform, backend=Backend.ANY if platform == "xla" else cfg.backend)

        impl = self.get_impl(cfg)
        out, final_state = impl(
            r=r,
            w=w,
            k=k,
            v=v,
            kk=kk,
            a=a,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )
        if return_state:
            return out, final_state
        return out

    def heuristic_cfg(self, inv: Invocation[RWKV7MulConfig, Array]) -> RWKV7MulConfig:
        del inv
        return RWKV7MulConfig(platform="auto", backend="any")

    def candidate_cfgs(self, inv: Invocation[RWKV7MulConfig, Array]):
        del inv
        return []


_executor_rwkv7: Executor[RWKV7Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv7"),
    )
)

_executor_rwkv7_mul: Executor[RWKV7MulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "heuristics"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("rwkv7_mul"),
    )
)


def rwkv7(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    b: Float[Array, "batch seq_len num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RWKV7Config | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-7 DPLR recurrence (a,b) with automatic backend selection.

    Computes the RWKV-7 Diagonal + Low-Rank recurrence over a sequence.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        a: Low-rank update vector `[B, T, H, K]`.
        b: Low-rank projection vector `[B, T, H, K]`.
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
    return _executor_rwkv7(
        RWKV7(),
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )


def rwkv7_mul(
    r: Float[Array, "batch seq_len num_heads qk_head_dim"],
    w: Float[Array, "batch seq_len num_heads qk_head_dim"],
    k: Float[Array, "batch seq_len num_heads qk_head_dim"],
    v: Float[Array, "batch seq_len num_heads v_head_dim"],
    kk: Float[Array, "batch seq_len num_heads qk_head_dim"],
    a: Float[Array, "batch seq_len num_heads qk_head_dim"],
    /,
    *,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    return_state: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RWKV7MulConfig | None = None,
) -> (
    Float[Array, "batch seq_len num_heads v_head_dim"]
    | tuple[
        Float[Array, "batch seq_len num_heads v_head_dim"],
        Float[Array, "... num_heads qk_head_dim v_head_dim"],
    ]
):
    """RWKV-7 recurrence (kk,a) multiplicative parameterization.

    Alternative parameterization of RWKV-7 that converts internally:
        a' = kk * a
        b' = -kk

    This form is commonly used by optimized kernel implementations.

    Args:
        r: Receptance (query) tensor `[B, T, H, K]`.
        w: Log decay tensor `[B, T, H, K]`.
        k: Key tensor `[B, T, H, K]`.
        v: Value tensor `[B, T, H, V]`.
        kk: Multiplicative factor `[B, T, H, K]`.
        a: Low-rank update base `[B, T, H, K]`.
        softmax_scale: Optional scale for receptance; defaults to `K**-0.5`.
        initial_state: Optional initial state `[B, H, K, V]` (or `[N, H, K, V]`
            in packed mode).
        reverse: If True, process sequence in reverse order.
        cu_seqlens: Optional cumulative sequence lengths for packed sequences.
        return_state: If True, also return the final state.
        platform: Backend platform override.
        cfg: Optional configuration object.

    Returns:
        Output tensor `[B, T, H, V]` (dtype matches `v`), or tuple of
        (output, final_state) if `return_state=True`. Final state is float32.
    """
    return _executor_rwkv7_mul(
        RWKV7Mul(),
        r=r,
        w=w,
        k=k,
        v=v,
        kk=kk,
        a=a,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        return_state=return_state,
        platform=platform,
        _cfg=cfg,
    )
