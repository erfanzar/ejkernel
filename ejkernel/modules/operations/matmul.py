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

"""Matrix multiplication kernel modules with automatic optimization."""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class GroupedMatmul(Kernel[KernelConfig, Array]):
    """Grouped Matrix Multiplication with custom optimization logic."""

    def __init__(self):
        """Initialize Grouped Matmul module."""
        super().__init__(op_id="grouped_matmul")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry."""
        platform = detect_platform("grouped_matmul", cfg.platform)
        return kernel_registry.get("grouped_matmul", platform=platform, backend=cfg.backend)

    def run(
        self,
        lhs: Float[Array, "m k"],
        rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
        group_sizes: Int[Array, "num_groups"],
        preferred_element_type=None,
        tiling: tuple[int, int, int] | None = (128, 128, 128),
        group_offset: Int[Array, "1"] | None = None,
        existing_out: Float[Array, "m n"] | None = None,
        transpose_rhs: bool = False,
        interpret: bool = False,
        precision=None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "m n"]:
        """Execute grouped matrix multiplication."""
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

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration with block sizes."""
        return KernelConfig(
            block_q=128,
            block_k=128,
            block_d=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning."""
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 128, 128, 4, 2),
            (256, 256, 128, 8, 3),
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


_grouped_matmul_executor = create_default_executor()


def grouped_matmul(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups"],
    preferred_element_type=None,
    tiling: tuple[int, int, int] | None = (128, 128, 128),
    group_offset: Int[Array, "1"] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision=None,
    platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
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
        >>> # Standard grouped matmul
        >>> out = grouped_matmul(lhs, rhs, group_sizes)
        >>>
        >>> # With custom tiling
        >>> out = grouped_matmul(lhs, rhs, group_sizes, tiling=(256, 256, 256))
        >>>
        >>> # With transposed RHS
        >>> out = grouped_matmul(lhs, rhs_transposed, group_sizes, transpose_rhs=True)
        >>>
        >>> # Accumulate into existing output
        >>> out = grouped_matmul(lhs, rhs, group_sizes, existing_out=prev_out)
            >>>
        >>> # Force specific platform
        >>> out = grouped_matmul(..., platform="triton")
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
    )
