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

"""Pooling operation modules with automatic optimization."""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class MeanPooling(Kernel[KernelConfig, Array]):
    """Mean Pooling with custom optimization logic."""

    def __init__(self):
        """Initialize Mean Pooling module."""
        super().__init__(op_id="mean_pooling")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry."""
        platform = detect_platform("mean_pooling", cfg.platform)
        return kernel_registry.get("mean_pooling", platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "batch seq_len hidden_dim"],
        chunk_size: int = 32,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch hidden_dim"]:
        """Execute mean pooling."""
        # Override platform in config if specified
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
        return impl(x=x, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration with block sizes."""
        return KernelConfig(
            block_q=32,
            block_k=32,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning."""
        block_configs = [
            (16, 16, 4, 1),
            (32, 32, 4, 1),
            (64, 64, 8, 2),
        ]

        candidates = []
        for block_q, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_mean_pooling_executor = create_default_executor()


def mean_pooling(
    x: Float[Array, "batch seq_len hidden_dim"],
    chunk_size: int = 32,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Execute mean pooling with automatic optimization.

    Efficiently computes the mean of sequence elements along the sequence dimension,
    optimized for variable-length sequences and chunked processing.

    Args:
        x: Input tensor [batch, seq_len, hidden_dim]
        chunk_size: Size of chunks for processing (default: 32)
        cu_seqlens: Cumulative sequence lengths for variable-length sequences

            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Mean pooled output [batch, hidden_dim]

    Example:
        >>> # Standard mean pooling
        >>> pooled = mean_pooling(x)
        >>>
        >>> # With larger chunk size for better memory efficiency
        >>> pooled = mean_pooling(x, chunk_size=64)
        >>>
        >>> # Variable-length sequences
        >>> pooled = mean_pooling(x, cu_seqlens=cu_seqs)
            >>>
        >>> # Force specific platform
        >>> out = mean_pooling(..., platform="triton")
    """
    return _mean_pooling_executor(
        MeanPooling(),
        x=x,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        platform=platform,
    )
