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


"""Operation-specific configuration classes.

This module defines configuration dataclasses for each attention operation,
providing type-safe, operation-specific parameters for kernel execution
and autotuning.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class BaseOperationConfig:
    """Base configuration for all operations."""

    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] = "auto"
    backend: str = "any"


@dataclass
class FlashAttentionConfig(BaseOperationConfig):
    """Configuration for Flash Attention operation.

    Args:
        chunk_size_q: Query chunk size for tiling (default: 128)
        chunk_size_k: Key chunk size for tiling (default: 128)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages for Triton (default: 2)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    chunk_size_q: int = 128
    chunk_size_k: int = 128
    num_warps: int = 4
    num_stages: int = 2


@dataclass
class BlockSparseAttentionConfig(BaseOperationConfig):
    """Configuration for Block Sparse Attention operation.

    Args:
        q_blocksize: Query block size for forward pass (default: 512)
        kv_blocksize: Key/value block size for forward pass (default: 512)
        bwd_q_blocksize: Query block size for backward pass (default: 1024)
        bwd_kv_blocksize: Key/value block size for backward pass (default: 1024)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    q_blocksize: int = 512
    kv_blocksize: int = 512
    bwd_q_blocksize: int = 1024
    bwd_kv_blocksize: int = 1024
    num_warps: int = 4
    num_stages: int = 2


@dataclass
class NativeSparseAttentionConfig(BaseOperationConfig):
    """Configuration for Native Sparse Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        block_size: Size of attention blocks for sparsity (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    block_size: int = 64
    num_warps: int = 4
    num_stages: int = 1


@dataclass
class RecurrentAttentionConfig(BaseOperationConfig):
    """Configuration for Recurrent Attention operation.

    Args:
        block_q: Query block size (default: 64)
        block_k: Key block size (default: 64)
        block_d: Head dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 64
    block_k: int = 64
    block_d: int = 64
    num_warps: int = 4
    num_stages: int = 1


@dataclass
class RingAttentionConfig(BaseOperationConfig):
    """Configuration for Ring Attention operation.

    Args:
        block_q: Query block size (default: 128)
        block_k: Key block size (default: 128)
        query_chunk_size: Chunk size for query processing (default: 512)
        key_chunk_size: Chunk size for key processing (default: 512)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 128
    block_k: int = 128
    query_chunk_size: int = 512
    key_chunk_size: int = 512
    num_warps: int = 4
    num_stages: int = 2


@dataclass
class PageAttentionConfig(BaseOperationConfig):
    """Configuration for Page Attention operation.

    Args:
        num_splits: Number of partitions for splitting contexts (default: 0 for auto)
        pages_per_compute_block: Pages per compute block (default: None)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_splits: int = 0
    pages_per_compute_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1


@dataclass
class AttentionConfig(BaseOperationConfig):
    """Configuration for basic Attention operation.

    Args:
        block_q: Query block size (default: 128)
        block_k: Key block size (default: 128)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_q: int = 128
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2


@dataclass
class GroupedMatmulConfig(BaseOperationConfig):
    """Configuration for Grouped Matrix Multiplication operation.

    Args:
        block_m: M dimension block size (default: 128)
        block_n: N dimension block size (default: 128)
        block_k: K dimension block size (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 2)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2


@dataclass
class MeanPoolingConfig(BaseOperationConfig):
    """Configuration for Mean Pooling operation.

    Args:
        block_size: Block size for pooling (default: 64)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    block_size: int = 64
    num_warps: int = 4
    num_stages: int = 1
