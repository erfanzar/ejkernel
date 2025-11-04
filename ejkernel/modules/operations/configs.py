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

from ejkernel.ops import BwdParams, FwdParams


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

    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)


@dataclass
class BlockSparseAttentionConfig(BaseOperationConfig):
    """Configuration for Block Sparse Attention operation.

    Args:
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    fwd_params: FwdParams | None = None
    bwd_params: BwdParams | None = None

    def __post_init__(self):
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)
        if isinstance(self.bwd_params, dict):
            self.bwd_params = BwdParams(**self.bwd_params)


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
    block_k: int = 128
    num_warps: int = 4
    num_stages: int = 2
    bypass_xla_tiling: bool = False


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


@dataclass
class RaggedDecodeAttentionConfig(BaseOperationConfig):
    """Configuration for Ragged Decode Attention operation.

    Args:
        block_size: Block size for computation tiling (default: 256)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    fwd_params: FwdParams | None = None

    def __post_init__(self):
        if isinstance(self.fwd_params, dict):
            self.fwd_params = FwdParams(**self.fwd_params)


@dataclass
class RaggedPageAttentionConfig(BaseOperationConfig):
    """Configuration for Ragged Page Attention operation.

    Args:
        num_kv_pages_per_block: Number of KV pages to process per compute block (default: None for auto)
        num_queries_per_block: Number of queries to process per compute block (default: None for auto)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1


@dataclass
class RaggedPageAttentionv3Config(BaseOperationConfig):
    """Configuration for Ragged Page Attention operation.

    Args:
        num_kv_pages_per_block: Number of KV pages to process per compute block (default: None for auto)
        num_queries_per_block: Number of queries to process per compute block (default: None for auto)
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    chunk_prefill_size: int | None = None
    num_kv_pages_per_block: int | None = None
    num_queries_per_block: int | None = None
    num_warps: int = 4
    num_stages: int = 1


@dataclass
class GLAttentionConfig(BaseOperationConfig):
    """Configuration for Gated Linear Attention operation.

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
class LightningAttentionConfig(BaseOperationConfig):
    """Configuration for Lightning Attention operation.

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
class FlashMLAConfig(BaseOperationConfig):
    """Configuration for Flash Multi-head Latent Attention operation.

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
class ScaledDotProductAttentionConfig(BaseOperationConfig):
    """Configuration for Scaled Dot Product Attention operation.

    Note: This operation uses XLA primitives directly without tunable block sizes.
    The config exists primarily for platform/backend selection.

    Args:
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    pass
