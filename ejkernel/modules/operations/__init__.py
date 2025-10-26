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
    - RaggedPageAttention: Page attention for variable-length sequences
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
    LightningAttentionConfig,
    NativeSparseAttentionConfig,
    PageAttentionConfig,
    RaggedDecodeAttentionConfig,
    RaggedPageAttentionConfig,
    RecurrentAttentionConfig,
    RingAttentionConfig,
    ScaledDotProductAttentionConfig,
)
from .flash_attention import FlashAttention, flash_attention
from .gated_linear_attention import GLAttention, gla_attention
from .grouped_matmul import GroupedMatmul, grouped_matmul
from .lightning_attention import LightningAttention, lightning_attention
from .multi_head_latent_attention import FlashMLA, mla_attention
from .native_sparse_attention import NativeSparseAttention, native_sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .ragged_decode_attention import RaggedDecodeAttention, ragged_decode_attention
from .ragged_page_attention import RaggedPageAttention, ragged_page_attention
from .ragged_page_attention_v3 import RaggedPageAttentionv3, ragged_page_attentionv3
from .recurrent import RecurrentAttention, recurrent_attention
from .ring_attention import RingAttention, ring_attention
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention

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
    "LightningAttention",
    "LightningAttentionConfig",
    "MeanPooling",
    "NativeSparseAttention",
    "NativeSparseAttentionConfig",
    "PageAttention",
    "PageAttentionConfig",
    "RaggedDecodeAttention",
    "RaggedDecodeAttentionConfig",
    "RaggedPageAttention",
    "RaggedPageAttentionConfig",
    "RaggedPageAttentionv3",
    "RecurrentAttention",
    "RecurrentAttentionConfig",
    "RingAttention",
    "RingAttentionConfig",
    "ScaledDotProductAttention",
    "ScaledDotProductAttentionConfig",
    "attention",
    "blocksparse_attention",
    "flash_attention",
    "gla_attention",
    "grouped_matmul",
    "lightning_attention",
    "mean_pooling",
    "mla_attention",
    "native_sparse_attention",
    "page_attention",
    "ragged_decode_attention",
    "ragged_page_attention",
    "ragged_page_attentionv3",
    "recurrent_attention",
    "ring_attention",
    "scaled_dot_product_attention",
)
