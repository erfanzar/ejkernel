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


"""Recurrent Attention module with automatic optimization."""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class RecurrentAttention(Kernel[KernelConfig, Array]):
    """Recurrent Attention with custom optimization logic."""

    def __init__(self):
        """Initialize Recurrent Attention module."""
        super().__init__(op_id="recurrent")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry."""
        platform = detect_platform("recurrent", cfg.platform)
        return kernel_registry.get("recurrent", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        g_gamma: Float[Array, "batch num_heads"] | None = None,
        gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        scale: float | None = None,
        initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Execute recurrent attention."""

        if platform is not None:
            cfg = KernelConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d if hasattr(cfg, "block_d") else None,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
            scale=scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration with block sizes."""
        return KernelConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning."""
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (128, 128, 64, 8, 2),
        ]

        candidates = []
        for block_q, block_k, block_d, num_warps, num_stages in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_recurrent_executor = create_default_executor()


def recurrent_attention(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "batch num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """Execute recurrent attention with automatic optimization.

    Recurrent attention processes sequences with stateful computation,
    maintaining hidden states across timesteps for efficient sequential processing.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        g: Gating tensor for query [batch, seq_len, num_heads, head_dim]
        g_gamma: Gating gamma [batch, num_heads]
        gk: Gating tensor for keys [batch, seq_len, num_heads, head_dim]
        gv: Gating tensor for values [batch, seq_len, num_heads, head_dim]
        scale: Scaling factor for attention
        initial_state: Initial hidden state
        reverse: Whether to process sequence in reverse
        cu_seqlens: Cumulative sequence lengths for variable-length sequences

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = recurrent(query, key, value)
        >>>
        >>>
        >>> out = recurrent(query, key, value, g=gates, gk=key_gates, gv=value_gates)
        >>>
        >>>
        >>> out = recurrent(query, key, value, initial_state=h0)
    """
    return _recurrent_executor(
        RecurrentAttention(),
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
