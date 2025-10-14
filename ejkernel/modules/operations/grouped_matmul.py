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


"""Grouped matrix multiplication kernel module with automatic optimization.

This module implements grouped matrix multiplication, an efficient operation for
batched matrix multiplication with variable group sizes. This is particularly
useful for mixture-of-experts models, grouped convolutions, and other scenarios
where different groups of inputs need to be multiplied with different weight matrices.

Unlike standard batched matrix multiplication which assumes uniform batch sizes,
grouped matmul handles variable-sized groups efficiently by:
    1. Processing groups of different sizes in a single operation
    2. Optimized memory access patterns for grouped computation
    3. Support for both transposed and non-transposed RHS matrices
    4. Optional accumulation into existing output tensors
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
from .configs import GroupedMatmulConfig


class GroupedMatmul(Kernel[GroupedMatmulConfig, Array]):
    """Grouped Matrix Multiplication with custom optimization logic.

    Performs efficient matrix multiplication for grouped inputs, where each group
    can have a different size. This is essential for mixture-of-experts (MoE) models
    where tokens are dynamically routed to different experts.

    Features:
        - Variable group size support via group_sizes array
        - Configurable tiling for memory and compute efficiency
        - Support for RHS transposition
        - Optional output accumulation (for multi-pass operations)
        - Group offset for partial computation
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    Typical use cases:
        - MoE layer computation (different tokens to different experts)
        - Grouped linear layers
        - Dynamic routing architectures
    """

    def __init__(self):
        """Initialize Grouped Matmul module."""
        super().__init__(op_id="grouped_matmul")

    def get_impl(self, cfg: GroupedMatmulConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for grouped matmul

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("grouped_matmul", cfg.platform)
        return kernel_registry.get("grouped_matmul", platform=platform, backend=cfg.backend)

    def run(
        self,
        lhs: Float[Array, "m k"],
        rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
        group_sizes: Int[Array, "num_groups"],
        preferred_element_type=None,
        tiling: tuple[int, int, int] | None = (128, 128, 128),
        group_offset: Int[Array, "..."] | None = None,
        existing_out: Float[Array, "m n"] | None = None,
        transpose_rhs: bool = False,
        interpret: bool = False,
        precision=None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: GroupedMatmulConfig,
    ) -> Float[Array, "m n"]:
        """Execute grouped matrix multiplication.

        Args:
            lhs: Left-hand side matrix [m, k] (typically activations)
            rhs: Right-hand side grouped matrices [num_groups, k, n] or [num_groups, n, k]
            group_sizes: Size of each group [num_groups], sum(group_sizes) == m
            preferred_element_type: Optional dtype for computation
            tiling: Tile sizes (m_tile, n_tile, k_tile) for blocking strategy
            group_offset: Optional offset into groups for partial computation
            existing_out: Optional existing output to accumulate into [m, n]
            transpose_rhs: Whether RHS matrices are transposed
            interpret: Use interpreted mode (for debugging)
            precision: Computation precision setting
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Matrix multiplication result [m, n]

        Note:
            The group_sizes array partitions the m dimension of lhs. Each partition
            is multiplied with the corresponding group matrix from rhs.
        """

        if platform is not None:
            cfg = GroupedMatmulConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            preferred_element_type=preferred_element_type,
            tiling=tiling,
            group_offset=group_offset,
            existing_out=existing_out,
            transpose_rhs=transpose_rhs,
            interpret=interpret,
            precision=precision,
        )

    def heuristic_cfg(self, inv: Invocation[GroupedMatmulConfig, Array]) -> GroupedMatmulConfig:
        """Provide default configuration with block sizes.

        Selects balanced block sizes suitable for typical grouped matmul workloads.
        The default 128x128x128 tiling provides good cache utilization for most
        problem sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with 128x128x128 blocks, 4 warps, 2 stages
        """
        return GroupedMatmulConfig(
            block_m=128,
            block_n=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[GroupedMatmulConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations with different block sizes to explore the
        performance space. Grouped matmul benefits from various tile sizes
        depending on group size distribution and matrix dimensions.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations with block sizes from 64 to 256

        Note:
            - Smaller blocks (64x64) work better for many small groups
            - Larger blocks (256x256) are efficient for fewer, larger groups
            - The k dimension blocking affects memory bandwidth utilization
        """
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 128, 128, 4, 2),
            (256, 256, 128, 8, 3),
        ]

        candidates = []
        for block_m, block_n, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                GroupedMatmulConfig(
                    block_m=block_m,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_grouped_matmul_executor: Executor[GroupedMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("grouped-matmul"),
    )
)


def grouped_matmul(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups"],
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    /,
    *,
    preferred_element_type=None,
    tiling: tuple[int, int, int] | None = (128, 128, 128),
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision=None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: GroupedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Execute grouped matrix multiplication with automatic optimization.

    Performs efficient batched matrix multiplication with variable group sizes,
    optimized for scenarios where different groups have different sizes.

    Args:
        lhs: Left-hand side matrix [m, k]
        rhs: Right-hand side matrices [num_groups, k, n] or [num_groups, n, k]
        group_sizes: Size of each group [num_groups]
        preferred_element_type: Preferred dtype for computation
        tiling: Tile sizes (m_tile, n_tile, k_tile) for blocking
        group_offset: Offset into groups (for partial computation)
        existing_out: Existing output to accumulate into
        transpose_rhs: Whether to transpose RHS matrices
        interpret: Use interpreted mode (slower but more debuggable)
        precision: Computation precision setting

            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Matrix multiplication result [m, n]

    Example:
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes)
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes, tiling=(256, 256, 256))
        >>>
        >>> out = grouped_matmul(lhs, rhs_transposed, group_sizes, transpose_rhs=True)
        >>>
        >>> out = grouped_matmul(lhs, rhs, group_sizes, existing_out=prev_out)
        >>>
        >>> out = grouped_matmul(..., platform="pallas")
    """
    return _grouped_matmul_executor(
        GroupedMatmul(),
        lhs=lhs,
        rhs=rhs,
        group_sizes=group_sizes,
        preferred_element_type=preferred_element_type,
        tiling=tiling,
        group_offset=group_offset,
        existing_out=existing_out,
        transpose_rhs=transpose_rhs,
        interpret=interpret,
        precision=precision,
        platform=platform,
        _cfg=cfg,
    )
