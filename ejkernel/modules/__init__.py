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


"""High-level kernel modules with automatic optimization.

This module provides user-friendly interfaces for kernel operations using the
ejkernel.ops framework for automatic configuration management and performance tuning.

Available Modules:
    Operations:
        - FlashAttention: Memory-efficient exact attention
        - BlockSparseAttention: Memory-efficient Sparse attention
        - PageAttention: Paged KV cache attention for serving
        - RaggedPageAttentionv2: Variable-length page attention
        - NativeSparseAttention: Block-wise sparse attention
        - Recurrent: Linear-time recurrent attention
        - GLAttention: Gated linear attention
        - LightningAttention: Lightning attention with decay
        - RingAttention: Distributed ring attention
        - MeanPooling: Efficient sequence mean pooling
        - GroupedMatmul: Grouped matrix multiplication

Example:
    >>> from ejkernel.modules import FlashAttention, create_default_executor
    >>>
    >>>
    >>> executor = create_default_executor("/tmp/kernel_cache")
    >>>
    >>>
    >>> attn = FlashAttention()
    >>> output = executor(attn, q, k, v, causal=True)
"""

from .operations import (
    Attention,
    AttentionConfig,
    BlockSparseAttention,
    BlockSparseAttentionConfig,
    FlashAttention,
    FlashAttentionConfig,
    FlashMLA,
    FlashMLAConfig,
    GLAttention,
    GLAttentionConfig,
    GroupedMatmul,
    GroupedMatmulConfig,
    KernelDeltaAttention,
    KernelDeltaAttentionConfig,
    LightningAttention,
    LightningAttentionConfig,
    MeanPooling,
    NativeSparseAttention,
    NativeSparseAttentionConfig,
    PageAttention,
    PageAttentionConfig,
    PrefillPageAttention,
    PrefillPageAttentionConfig,
    RaggedDecodeAttention,
    RaggedDecodeAttentionConfig,
    RaggedPageAttentionv2,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv3,
    RaggedPageAttentionv3Config,
    RecurrentAttention,
    RecurrentAttentionConfig,
    RingAttention,
    RingAttentionConfig,
    ScaledDotProductAttention,
    ScaledDotProductAttentionConfig,
    StateSpaceV1,
    StateSpaceV1Config,
    StateSpaceV2,
    StateSpaceV2Config,
    UnifiedAttention,
    UnifiedAttentionConfig,
    attention,
    blocksparse_attention,
    flash_attention,
    gla_attention,
    grouped_matmul,
    kda_attention,
    kernel_delta_attention,
    lightning_attention,
    mean_pooling,
    mla_attention,
    native_sparse_attention,
    page_attention,
    prefill_page_attention,
    ragged_decode_attention,
    ragged_page_attention_v2,
    ragged_page_attention_v3,
    recurrent_attention,
    ring_attention,
    scaled_dot_product_attention,
    state_space_v1,
    state_space_v2,
    unified_attention,
)

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
