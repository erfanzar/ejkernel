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


"""Pooling operation modules with automatic optimization.

This module implements efficient pooling operations for sequence data, optimized
for JAX execution. Mean pooling is particularly useful for:
    - Sentence embeddings in NLP (pooling token representations)
    - Sequence classification (reducing sequence to fixed-size representation)
    - Feature aggregation across time steps
    - Dimensionality reduction in transformer outputs

The implementation supports variable-length sequences via cumulative sequence
lengths, enabling efficient batched processing of sequences with different lengths.
"""

from __future__ import annotations

import os
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
from .configs import MeanPoolingConfig


class MeanPooling(Kernel[MeanPoolingConfig, Array]):
    """Mean Pooling with custom optimization logic.

    Computes the mean of sequence elements along the sequence dimension, with
    support for variable-length sequences and chunked processing for memory efficiency.

    Features:
        - Efficient mean computation over sequence dimension
        - Support for variable-length sequences via cu_seqlens
        - Configurable chunk size for memory-efficient processing
        - Automatic platform selection (Triton/Pallas/XLA/CUDA)
        - Proper handling of padding in variable-length scenarios

    This is commonly used to convert variable-length token sequences into
    fixed-size representations for classification or embedding tasks.
    """

    def __init__(self):
        """Initialize Mean Pooling module."""
        super().__init__(op_id="mean_pooling")

    def get_impl(self, cfg: MeanPoolingConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for mean pooling

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("mean_pooling", cfg.platform)
        return kernel_registry.get("mean_pooling", platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "batch seq_len hidden_dim"],
        chunk_size: int = 32,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: MeanPoolingConfig,
    ) -> Float[Array, "batch hidden_dim"]:
        """Execute mean pooling over sequence dimension.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            chunk_size: Size of chunks for processing (default: 32)
            cu_seqlens: Optional cumulative sequence lengths [num_seqs + 1] for
                variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Pooled output [batch, hidden_dim]

        Note:
            When cu_seqlens is provided, padding tokens are excluded from the mean
            computation, ensuring accurate pooling for variable-length sequences.
        """

        if platform is not None:
            cfg = MeanPoolingConfig(
                block_size=cfg.block_size,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(x=x, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

    def heuristic_cfg(self, inv: Invocation[MeanPoolingConfig, Array]) -> MeanPoolingConfig:
        """Provide default configuration with block sizes.

        Selects default block size and warp configuration based on typical
        sequence pooling workloads. These defaults work well for most cases
        but can be overridden via autotuning.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block_size=64, num_warps=4, num_stages=1
        """
        return MeanPoolingConfig(
            block_size=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[MeanPoolingConfig, Array]):
        """Generate candidate configurations for autotuning.

        Mean pooling has tunable block_size for chunked processing. Generates
        configurations with varying block sizes to find optimal performance
        for the specific hardware and input dimensions.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            List of candidate configurations with different block sizes (32, 64, 128)
            and corresponding warp/stage configurations

        Note:
            Smaller block sizes (32) reduce memory usage but may have lower throughput.
            Larger block sizes (128) improve throughput for large sequences.
        """
        block_configs = [
            (32, 4, 1),
            (64, 4, 1),
            (128, 8, 2),
        ]

        candidates = []
        for block_size, num_warps, num_stages in block_configs:
            candidates.append(
                MeanPoolingConfig(
                    block_size=block_size,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_mean_pooling_executor: Executor[MeanPoolingConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("pooling"),
    )
)


def mean_pooling(
    x: Float[Array, "batch seq_len hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    chunk_size: int = 32,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: MeanPoolingConfig | None = None,
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
        >>>
        >>> pooled = mean_pooling(x)
        >>>
        >>>
        >>> pooled = mean_pooling(x, chunk_size=64)
        >>>
        >>>
        >>> pooled = mean_pooling(x, cu_seqlens=cu_seqs)
            >>>
        >>>
        >>> out = mean_pooling(..., platform="triton")
    """
    return _mean_pooling_executor(
        MeanPooling(),
        x=x,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        platform=platform,
        _cfg=cfg,
    )
