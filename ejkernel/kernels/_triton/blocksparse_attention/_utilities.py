# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


import math
from typing import Literal

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.utils import get_strides


def generate_blocksparse_layout(
    seq_len: int,
    num_heads: int,
    block_size: int,
    local_blocks: int = 4,
    vert_stride: int = 0,
    homo_head: bool = True,
    return_dense: bool = False,
    dtype: jnp.dtype = jnp.bool_,
) -> Bool[Array, "num_heads num_blocks_q num_blocks_k"] | Bool[Array, "num_blocks_q num_blocks_k"]:
    """Generate block-sparse attention pattern.

    Creates a block-level sparsity mask for attention computation, supporting
    local attention windows and vertical stride patterns.

    Args:
        seq_len: Sequence length (must be divisible by block_size)
        num_heads: Number of attention heads
        block_size: Size of each attention block
        local_blocks: Number of nearest blocks to attend to (local attention window)
        vert_stride: Vertical stride for additional block connections
        homo_head: Whether all heads share the same pattern (homogeneous)
        return_dense: Return dense mask instead of sparse indices
        dtype: Data type for the mask array

    Returns:
        Block-sparse attention pattern as boolean mask array
    """
    assert seq_len % block_size == 0, f"seq_len {seq_len} must be divisible by block_size {block_size}"

    num_blocks = seq_len // block_size

    # Create base pattern
    pattern = jnp.zeros((num_blocks, num_blocks), dtype=dtype)

    # Add local attention blocks
    for i in range(num_blocks):
        # Causal: only attend to previous and current blocks
        start = max(0, i - local_blocks + 1)
        end = min(num_blocks, i + 1)
        pattern = pattern.at[i, start:end].set(True)

    # Add vertical stride connections if specified
    if vert_stride > 0:
        for i in range(num_blocks):
            for j in range(0, i, vert_stride):
                pattern = pattern.at[i, j].set(True)

    # Handle heterogeneous vs homogeneous heads
    if homo_head or num_heads == 1:
        return pattern
    else:
        # For heterogeneous heads, can create different patterns per head
        # For now, replicate the same pattern
        return jnp.broadcast_to(pattern[None, :, :], (num_heads, num_blocks, num_blocks))


def compute_crow_indices(
    layout: Bool[Array, "num_blocks_q num_blocks_k"],
    dtype: jnp.dtype = jnp.int32,
) -> tuple[Int[Array, "num_blocks_q + 1"], Int[Array, "nnz"]]:
    """Convert block sparse layout to Compressed Row Storage (CSR) format.

    Used for efficient forward pass traversal of sparse blocks.

    Args:
        layout: Boolean mask indicating which blocks are active
        dtype: Integer dtype for indices

    Returns:
        tuple: (crow_ptr, col_indices) for CSR representation
            - crow_ptr: Row pointers of shape [num_blocks_q + 1]
            - col_indices: Column indices of shape [nnz]
    """
    num_blocks_q, num_blocks_k = layout.shape

    # Count non-zero blocks per row
    row_counts = jnp.sum(layout, axis=1, dtype=dtype)

    # Compute row pointers (cumulative sum)
    crow_ptr = jnp.zeros(num_blocks_q + 1, dtype=dtype)
    crow_ptr = crow_ptr.at[1:].set(jnp.cumsum(row_counts))

    # Extract column indices for non-zero blocks
    row_indices, col_indices = jnp.nonzero(layout)

    return crow_ptr, col_indices.astype(dtype)


def compute_ccol_indices(
    layout: Bool[Array, "num_blocks_q num_blocks_k"],
    dtype: jnp.dtype = jnp.int32,
) -> tuple[Int[Array, "num_blocks_k + 1"], Int[Array, "nnz"]]:
    """Convert block sparse layout to Compressed Column Storage (CSC) format.

    Used for efficient backward pass traversal of sparse blocks.

    Args:
        layout: Boolean mask indicating which blocks are active
        dtype: Integer dtype for indices

    Returns:
        tuple: (ccol_ptr, row_indices) for CSC representation
            - ccol_ptr: Column pointers of shape [num_blocks_k + 1]
            - row_indices: Row indices of shape [nnz]
    """
    # Transpose for column-wise processing
    layout_t = layout.T
    num_blocks_k, num_blocks_q = layout_t.shape

    # Count non-zero blocks per column
    col_counts = jnp.sum(layout_t, axis=1, dtype=dtype)

    # Compute column pointers
    ccol_ptr = jnp.zeros(num_blocks_k + 1, dtype=dtype)
    ccol_ptr = ccol_ptr.at[1:].set(jnp.cumsum(col_counts))

    # Extract row indices for non-zero blocks
    col_indices, row_indices = jnp.nonzero(layout_t)

    return ccol_ptr, row_indices.astype(dtype)


def create_blocksparse_metadata(
    seq_len_q: int,
    seq_len_k: int,
    num_heads: int,
    block_m: int,
    block_n: int,
    local_blocks: int = 4,
    vert_stride: int = 0,
    homo_head: bool = True,
) -> dict:
    """Create metadata for blocksparse attention computation.

    Generates all necessary indices and pointers for efficient sparse
    attention computation in both forward and backward passes.

    Args:
        seq_len_q: Query sequence length
        seq_len_k: Key sequence length
        num_heads: Number of attention heads
        block_m: Block size for queries
        block_n: Block size for keys
        local_blocks: Number of local attention blocks
        vert_stride: Vertical stride for sparse pattern
        homo_head: Whether heads share the same pattern

    Returns:
        dict: Metadata containing layout, crow/ccol indices, and config
    """
    # For now, assume seq_len_q == seq_len_k for simplicity
    assert seq_len_q == seq_len_k, "Different Q/K lengths not yet supported"
    assert seq_len_q % block_m == 0, f"seq_len_q {seq_len_q} must be divisible by block_m {block_m}"
    assert seq_len_k % block_n == 0, f"seq_len_k {seq_len_k} must be divisible by block_n {block_n}"

    # Generate sparse layout
    layout = generate_blocksparse_layout(
        seq_len=seq_len_q,
        num_heads=num_heads if not homo_head else 1,
        block_size=block_m,  # Assuming square blocks for now
        local_blocks=local_blocks,
        vert_stride=vert_stride,
        homo_head=homo_head,
    )

    # Compute compression indices
    if layout.ndim == 2:
        crow_ptr, col_indices = compute_crow_indices(layout)
        ccol_ptr, row_indices = compute_ccol_indices(layout)
    else:
        # Handle per-head patterns
        crow_ptrs, col_indices_list = [], []
        ccol_ptrs, row_indices_list = [], []
        for h in range(num_heads):
            crow, cols = compute_crow_indices(layout[h])
            ccol, rows = compute_ccol_indices(layout[h])
            crow_ptrs.append(crow)
            col_indices_list.append(cols)
            ccol_ptrs.append(ccol)
            row_indices_list.append(rows)

        crow_ptr = jnp.stack(crow_ptrs)
        col_indices = jnp.stack(col_indices_list)
        ccol_ptr = jnp.stack(ccol_ptrs)
        row_indices = jnp.stack(row_indices_list)

    return {
        "layout": layout,
        "crow_ptr": crow_ptr,
        "col_indices": col_indices,
        "ccol_ptr": ccol_ptr,
        "row_indices": row_indices,
        "num_blocks_q": seq_len_q // block_m,
        "num_blocks_k": seq_len_k // block_n,
        "block_m": block_m,
        "block_n": block_n,
        "nnz": jnp.sum(layout),
    }


@triton.jit
def padded_load(
    ptrs,
    offs_a,
    offs_b,
    PA0: tl.constexpr,
    PA1: tl.constexpr,
    LA0: tl.constexpr,
    LA1: tl.constexpr,
):
    """Load data from memory with optional padding for boundary conditions.

    Conditionally loads data with masking based on compile-time constants,
    optimizing for different padding scenarios.

    Args:
        ptrs: Pointer to memory location
        offs_a: Offsets for first dimension
        offs_b: Offsets for second dimension
        PA0: Whether first dimension needs padding check
        PA1: Whether second dimension needs padding check
        LA0: Actual length of first dimension
        LA1: Actual length of second dimension

    Returns:
        Loaded tensor with zeros for out-of-bounds elements
    """
    if PA0:
        if PA1:
            x = tl.load(
                ptrs,
                mask=(offs_a[:, None] < LA0) & (offs_b[None, :] < LA1),
                other=0.0,
            )
        else:
            x = tl.load(
                ptrs,
                mask=offs_a[:, None] < LA0,
                other=0.0,
            )
    else:
        if PA1:
            x = tl.load(
                ptrs,
                mask=offs_b[None, :] < LA1,
                other=0.0,
            )
        else:
            x = tl.load(ptrs)
    return x