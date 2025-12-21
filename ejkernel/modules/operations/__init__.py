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


"""Attention kernel modules with automatic optimization.

This module provides a collection of high-performance attention mechanisms
and related operations optimized for JAX. All implementations support automatic
platform selection (XLA, Triton, Pallas, CUDA) and optional autotuning.

Available Attention Variants:
    - Attention: Standard multi-head attention with XLA optimization
    - FlashAttention: Memory-efficient O(N) complexity attention
    - FlashMLA: Multi-head latent attention with low-rank compression
    - GLAttention: Gated linear attention mechanism
    - LightningAttention: Layer-aware attention optimization
    - NativeSparseAttention: Sparse attention with block patterns
    - PageAttention: Paged KV cache for serving workloads
    - RaggedPageAttentionv2: Page attention for variable-length sequences
    - RaggedPageAttentionv3: Advanced page attention for variable-length sequences v3
    - RecurrentAttention: Stateful recurrent attention
    - RingAttention: Distributed attention with ring topology
    - ScaledDotProductAttention: Standard scaled dot-product attention

Additional Operations:
    - GroupedMatmul: Efficient grouped matrix multiplication
    - MeanPooling: Sequence mean pooling operation

Features:
    - Automatic kernel selection based on hardware and input shapes
    - Configuration caching for consistent performance
    - Optional autotuning to find optimal block sizes
    - Support for causal masking, dropout, and sliding windows
    - Variable-length sequence handling via cumulative lengths
    - Gradient-checkpointing support for memory efficiency

Example:
    >>> from ejkernel.modules.operations import flash_attention
    >>>
    >>>
    >>> output = flash_attention(query, key, value, causal=True)
    >>>
    >>>
    >>> output = flash_attention(
    ...     query, key, value,
    ...     softmax_scale=0.125,
    ...     dropout_prob=0.1,
    ...     sliding_window=(256, 256)
    ... )

Note:
    All attention functions automatically handle mixed precision and
    select the best available backend for your hardware.
"""

from .attention import Attention, attention
from .blocksparse_attention import BlockSparseAttention, blocksparse_attention
from .configs import (
    AttentionConfig,
    BlockSparseAttentionConfig,
    FlashAttentionConfig,
    FlashMLAConfig,
    GLAttentionConfig,
    GroupedMatmulConfig,
    KernelDeltaAttentionConfig,
    LightningAttentionConfig,
    NativeSparseAttentionConfig,
    PageAttentionConfig,
    PrefillPageAttentionConfig,
    RaggedDecodeAttentionConfig,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv3Config,
    RecurrentAttentionConfig,
    RingAttentionConfig,
    ScaledDotProductAttentionConfig,
    StateSpaceV1Config,
    StateSpaceV2Config,
    UnifiedAttentionConfig,
)
from .flash_attention import FlashAttention, flash_attention
from .gated_linear_attention import GLAttention, gla_attention
from .grouped_matmul import GroupedMatmul, grouped_matmul
from .kernel_delta_attention import KernelDeltaAttention, kda_attention, kernel_delta_attention
from .lightning_attention import LightningAttention, lightning_attention
from .multi_head_latent_attention import FlashMLA, mla_attention
from .native_sparse_attention import NativeSparseAttention, native_sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .prefill_page_attention import PrefillPageAttention, prefill_page_attention
from .ragged_decode_attention import RaggedDecodeAttention, ragged_decode_attention
from .ragged_page_attention_v2 import RaggedPageAttentionv2, ragged_page_attention_v2
from .ragged_page_attention_v3 import RaggedPageAttentionv3, ragged_page_attention_v3
from .recurrent import RecurrentAttention, recurrent_attention
from .ring_attention import RingAttention, ring_attention
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention
from .state_space_v1 import StateSpaceV1, state_space_v1
from .state_space_v2 import StateSpaceV2, state_space_v2
from .unified_attention import UnifiedAttention, unified_attention

__all__ = (
    "Attention",
    "AttentionConfig",
    "BlockSparseAttention",
    "BlockSparseAttentionConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "FlashMLA",
    "FlashMLAConfig",
    "GLAttention",
    "GLAttentionConfig",
    "GroupedMatmul",
    "GroupedMatmulConfig",
    "KernelDeltaAttention",
    "KernelDeltaAttentionConfig",
    "LightningAttention",
    "LightningAttentionConfig",
    "MeanPooling",
    "NativeSparseAttention",
    "NativeSparseAttentionConfig",
    "PageAttention",
    "PageAttentionConfig",
    "PrefillPageAttention",
    "PrefillPageAttention",
    "PrefillPageAttentionConfig",
    "RaggedDecodeAttention",
    "RaggedDecodeAttentionConfig",
    "RaggedPageAttentionv2",
    "RaggedPageAttentionv2Config",
    "RaggedPageAttentionv3",
    "RaggedPageAttentionv3Config",
    "RecurrentAttention",
    "RecurrentAttentionConfig",
    "RingAttention",
    "RingAttentionConfig",
    "ScaledDotProductAttention",
    "ScaledDotProductAttentionConfig",
    "StateSpaceV1",
    "StateSpaceV1Config",
    "StateSpaceV2",
    "StateSpaceV2Config",
    "UnifiedAttention",
    "UnifiedAttentionConfig",
    "attention",
    "blocksparse_attention",
    "flash_attention",
    "gla_attention",
    "grouped_matmul",
    "kda_attention",
    "kernel_delta_attention",
    "lightning_attention",
    "mean_pooling",
    "mla_attention",
    "native_sparse_attention",
    "page_attention",
    "prefill_page_attention",
    "prefill_page_attention",
    "ragged_decode_attention",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "recurrent_attention",
    "ring_attention",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
)
