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

import hashlib
from dataclasses import dataclass
from typing import Literal

from ejkernel.ops import BwdParams, FwdParams


def get_safe_hash_int(text, algorithm="md5"):
    """Generate a hash of text using specified algorithm with safety checks."""
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def hash_fn(self) -> int:
    """Generate a hash for an object based on its dictionary values."""
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


@dataclass
class BaseOperationConfig:
    """Base configuration for all operations."""

    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] = "auto"
    backend: str = "any"

    __hash__ = hash_fn


@dataclass
class FlashAttentionConfig(BaseOperationConfig):
    """Configuration for Flash Attention operation.

    Args:
        fwd_params: Forward kernel parameters (uses `q_blocksize`/`kv_blocksize` for tiling).
        bwd_params: Backward kernel parameters (optional).
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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


@dataclass
class RingAttentionConfig(BaseOperationConfig):
    """Configuration for Ring Attention operation.

    Args:
        fwd_params: Forward pass block size parameters
        bwd_params: Backward pass block size parameters
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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


@dataclass
class UnifiedAttentionConfig(BaseOperationConfig):
    """Configuration for vLLM-style unified (paged) attention operation.

    Args:
        seq_threshold_3d: Threshold (in #seqs) for selecting the segmented 3D
            decode kernel on GPU (Triton only).
        num_par_softmax_segments: Number of parallel softmax segments used by
            the segmented 3D decode kernel (Triton only).
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    seq_threshold_3d: int | None = None
    num_par_softmax_segments: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


@dataclass
class RaggedPageAttentionv2Config(BaseOperationConfig):
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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


@dataclass
class KernelDeltaAttentionConfig(BaseOperationConfig):
    """Configuration for Kernel Delta Attention (KDA) operation.

    Note: This operation currently uses an XLA implementation without tunable
    block sizes. The config exists primarily for platform/backend selection.

    Args:
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    pass

    __hash__ = hash_fn


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

    __hash__ = hash_fn


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

    __hash__ = hash_fn


@dataclass
class PrefillPageAttentionConfig(BaseOperationConfig):
    """Configuration for Prefill Page Attention operation.

    Args:
        num_warps: Number of warps for Triton kernels (default: 4)
        num_stages: Number of pipeline stages (default: 1)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    num_warps: int = 4
    num_stages: int = 1

    __hash__ = hash_fn


@dataclass
class StateSpaceV1Config(BaseOperationConfig):
    """Configuration for SSM1 (Mamba1-style) Selective State Space operation.

    Note: This operation uses XLA implementation primarily without tunable
    block sizes. The config exists primarily for platform/backend selection.

    Args:
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    __hash__ = hash_fn


@dataclass
class StateSpaceV2Config(BaseOperationConfig):
    """Configuration for SSM2 (Mamba2-style) Selective State Space operation.

    Args:
        n_groups: Number of groups for B, C parameters (default: 1)
        use_gated_rmsnorm: Whether to use gated RMSNorm for output (default: False)
        rmsnorm_eps: Epsilon for RMSNorm stability (default: 1e-5)
        platform: Target platform (triton/pallas/cuda/xla/auto)
        backend: Backend specification (default: "any")
    """

    n_groups: int = 1
    use_gated_rmsnorm: bool = False
    rmsnorm_eps: float = 1e-5

    __hash__ = hash_fn
