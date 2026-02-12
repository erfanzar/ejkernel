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

Available Attention Modules:
    Standard Attention:
        - Attention: Standard multi-head attention with XLA optimization
        - FlashAttention: Memory-efficient O(N) complexity attention
        - ScaledDotProductAttention: Standard scaled dot-product attention

    Paged/Serving Attention:
        - PageAttention: Paged KV cache for decode phase
        - PrefillPageAttention: Chunked prefill with paged KV cache
        - RaggedPageAttentionv2: Page attention for variable-length sequences
        - RaggedPageAttentionv3: Advanced variable-length page attention

    Sparse Attention:
        - BlockSparseAttention: Memory-efficient block-sparse attention
        - NativeSparseAttention: Sparse attention with explicit block patterns

    Linear/Recurrent Attention:
        - GLAttention: Gated linear attention mechanism
        - LightningAttention: Lightning attention with decay
        - KernelDeltaAttention: Linear attention with delta rule updates
        - RecurrentAttention: Stateful recurrent attention

    Distributed Attention:
        - RingAttention: Distributed attention with ring topology

    Memory-Efficient Attention:
        - FlashMLA: Multi-head latent attention with low-rank KV compression

Linear Recurrent Models:
    - RWKV4: Linear attention with time-decay
    - RWKV6: Enhanced RWKV with token-dependent decay
    - RWKV7: Latest RWKV with improved state updates
    - RWKV7Mul: RWKV-7 with multiplicative state updates

State Space Models:
    - StateSpaceV1: Mamba-1 style selective state space model
    - StateSpaceV2: Mamba-2 style selective state space model

Additional Operations:
    - GroupedMatmul: Efficient grouped matrix multiplication (for MoE)
    - MeanPooling: Sequence mean pooling operation

Example:
    >>> from ejkernel.modules import FlashAttention
    >>>
    >>> # Basic usage
    >>> attn = FlashAttention()
    >>> output = attn.run(q, k, v, causal=True, cfg=attn.heuristic_cfg(None))

    >>> # Using the functional interface with auto-optimization
    >>> from ejkernel.modules import flash_attention
    >>> output = flash_attention(q, k, v, causal=True)
"""

from .operations import (
    RWKV4,
    RWKV6,
    RWKV7,
    AllGatherMatmul,
    AllGatherMatmulConfig,
    Attention,
    AttentionConfig,
    BlockSparseAttention,
    BlockSparseAttentionConfig,
    ChunkedPrefillPagedDecode,
    ChunkedPrefillPagedDecodeConfig,
    DecodeAttention,
    DecodeAttentionConfig,
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
    MeanPoolingConfig,
    NativeSparseAttention,
    NativeSparseAttentionConfig,
    PageAttention,
    PageAttentionConfig,
    PrefillPageAttention,
    PrefillPageAttentionConfig,
    QuantizedMatmul,
    QuantizedMatmulConfig,
    RaggedDecodeAttention,
    RaggedDecodeAttentionConfig,
    RaggedPageAttentionv2,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv3,
    RaggedPageAttentionv3Config,
    RecurrentAttention,
    RecurrentAttentionConfig,
    ReduceScatterMatmul,
    ReduceScatterMatmulConfig,
    RingAttention,
    RingAttentionConfig,
    RWKV4Config,
    RWKV6Config,
    RWKV7Config,
    RWKV7Mul,
    RWKV7MulConfig,
    ScaledDotProductAttention,
    ScaledDotProductAttentionConfig,
    StateSpaceV1,
    StateSpaceV1Config,
    StateSpaceV2,
    StateSpaceV2Config,
    UnifiedAttention,
    UnifiedAttentionConfig,
    all_gather_matmul,
    attention,
    blocksparse_attention,
    chunked_prefill_paged_decode,
    decode_attention,
    flash_attention,
    flash_mla,
    gla_attention,
    grouped_matmul,
    kda_attention,
    kernel_delta_attention,
    lightning_attention,
    mean_pooling,
    native_sparse_attention,
    page_attention,
    prefill_page_attention,
    quantized_matmul,
    ragged_decode_attention,
    ragged_page_attention_v2,
    ragged_page_attention_v3,
    recurrent_attention,
    reduce_scatter_matmul,
    ring_attention,
    rwkv4,
    rwkv6,
    rwkv7,
    rwkv7_mul,
    scaled_dot_product_attention,
    state_space_v1,
    state_space_v2,
    unified_attention,
)

__all__ = (
    "RWKV4",
    "RWKV6",
    "RWKV7",
    "AllGatherMatmul",
    "AllGatherMatmulConfig",
    "Attention",
    "AttentionConfig",
    "BlockSparseAttention",
    "BlockSparseAttentionConfig",
    "ChunkedPrefillPagedDecode",
    "ChunkedPrefillPagedDecodeConfig",
    "DecodeAttention",
    "DecodeAttentionConfig",
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
    "MeanPoolingConfig",
    "NativeSparseAttention",
    "NativeSparseAttentionConfig",
    "PageAttention",
    "PageAttentionConfig",
    "PrefillPageAttention",
    "PrefillPageAttentionConfig",
    "QuantizedMatmul",
    "QuantizedMatmulConfig",
    "RWKV4Config",
    "RWKV6Config",
    "RWKV7Config",
    "RWKV7Mul",
    "RWKV7MulConfig",
    "RaggedDecodeAttention",
    "RaggedDecodeAttentionConfig",
    "RaggedPageAttentionv2",
    "RaggedPageAttentionv2Config",
    "RaggedPageAttentionv3",
    "RaggedPageAttentionv3Config",
    "RecurrentAttention",
    "RecurrentAttentionConfig",
    "ReduceScatterMatmul",
    "ReduceScatterMatmulConfig",
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
    "all_gather_matmul",
    "attention",
    "blocksparse_attention",
    "chunked_prefill_paged_decode",
    "decode_attention",
    "flash_attention",
    "flash_mla",
    "gla_attention",
    "grouped_matmul",
    "kda_attention",
    "kernel_delta_attention",
    "lightning_attention",
    "mean_pooling",
    "native_sparse_attention",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "recurrent_attention",
    "reduce_scatter_matmul",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
)
