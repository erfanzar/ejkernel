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


from __future__ import annotations

import contextlib
import typing

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.experimental import xla_metadata
from jaxtyping import Array, DTypeLike, Float, Int

from ..._registry import Backend, Platform, kernel_registry

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

set_xla_metadata = xla_metadata.set_xla_metadata


@kernel_registry.register("grouped_matmul", Platform.XLA, Backend.ANY)
@kernel_registry.register("grouped_matmulv2", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmul(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n"]:
    """Grouped Matrix Multiplication: Compute separate matrix products for each group.

    This function performs grouped matrix multiplication where different row slices of
    the left-hand side matrix are multiplied with different matrices from the right-hand
    side tensor. Mathematically, for each group i:
        out[start_i:end_i, :] = lhs[start_i:end_i, :] @ rhs[i, :, :]
    where start_i and end_i are determined by group_sizes.

    The implementation uses Pallas to generate efficient TPU kernels that:
    - Process multiple groups in a single kernel launch
    - Handle groups that don't align with tile boundaries
    - Support incremental accumulation for memory efficiency
    - Optimize memory access patterns for TPU's memory hierarchy

    Args:
        lhs: Left-hand side matrix of shape [m, k] where m is the total number
            of rows across all groups and k is the inner dimension.
        rhs: Right-hand side tensor of shape [num_groups, k, n] containing a
            separate matrix for each group. Can be transposed if transpose_rhs=True.
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the number of rows in lhs belonging to that group.
            Must sum to m (first dimension of lhs).
        preferred_element_type: Output dtype. Defaults to float32. The kernel uses
            float32 for accumulation regardless, then casts to this type.
        tiling: Tile sizes as (tm, tk, tn) tuple, or a callable that returns tile
            sizes based on problem dimensions. Typical TPU tiles are 128x128.
            The callable signature is (m, k, n) -> (tm, tk, tn) | None.
        group_offset: Starting group index for sharded execution. Defaults to 0.
            Useful when distributing groups across multiple devices.
        existing_out: Optional pre-existing output tensor to accumulate into.
            Must have shape [m, n] and dtype matching preferred_element_type.
            Enables incremental computation and memory reuse.
        transpose_rhs: If True, expects rhs shape [num_groups, n, k] and transposes
            during multiplication. Useful for certain memory layouts.
        interpret: Run kernel in interpret mode for debugging. Slower but provides
            better error messages and doesn't require compilation.

    Returns:
        Output matrix of shape [m, n] containing the concatenated results of all
        group matrix multiplications.

    Algorithm Overview:
        1. Validate inputs and determine computation parameters
        2. Create group metadata for efficient tile-to-group mapping
        3. Define Pallas kernel that:
           - Loads tiles from lhs and group-specific slices from rhs
           - Accumulates partial products in on-chip memory
           - Masks and stores results respecting group boundaries
        4. Launch kernel with appropriate grid dimensions
        5. Zero unprocessed regions if doing partial computation

    TPU Optimizations:
        - Tiles sized to match TPU's Matrix Multiply Units (typically 128x128)
        - Prefetch scheduling for hiding memory latency
        - VMEM scratch space for accumulation to minimize HBM traffic
        - Efficient masking for partial tiles using TPU's predication
        - Dimension semantics hints for XLA compiler optimization

    Example:
        >>>
        >>> lhs = jnp.randn(300, 64)
        >>> rhs = jnp.randn(3, 64, 32)
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> result = grouped_matmul(lhs, rhs, group_sizes)

    Notes:
        - The k dimension can have partial tiles (handled via masking)
        - The m dimension must be divisible by tm for correctness
        - Empty groups (size 0) are skipped for efficiency
        - Cost estimation helps XLA make scheduling decisions
    """
    if tiling is None:
        manager = contextlib.nullcontext()
    else:
        manager = set_xla_metadata(ragged_dot_tiling=",".join([str(t) for t in tiling]))
    with manager:
        out = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
            ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
                dot_dimension_numbers=(((1,), (2,)) if transpose_rhs else ((1,), (1,)), ((), ())),
                lhs_ragged_dimensions=(0,),
                rhs_group_dimensions=(0,),
            ),
        )
    return out
