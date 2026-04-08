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
        - RaggedPageAttentionv2TurboQuant: TurboQuant-compressed read-only page attention
        - RaggedPageAttentionv3: Advanced variable-length page attention
        - RaggedPageAttentionv3TurboQuant: TurboQuant-compressed fused page attention

    Sparse Attention:
        - BlockSparseAttention: Memory-efficient block-sparse attention
        - NativeSparseAttention: Sparse attention with explicit block patterns

    Linear/Recurrent Attention:
        - GatedDeltaRule: Gated delta rule linear attention
        - GLAttention: Gated linear attention mechanism
        - LightningAttention: Lightning attention with decay
        - KernelDeltaAttention: Linear attention with delta rule updates
        - RecurrentAttention: Stateful recurrent attention

    Distributed Attention:
        - RingAttention: Distributed attention with ring topology

    Memory-Efficient Attention:
        - DeepSeekAttention: DeepSeek Sparse Attention (MLA + Lightning Indexer)
        - FlashMLA: Multi-head latent attention with low-rank KV compression
        - MultiLatentRaggedPageAttention: MLA ragged paged attention with cache updates
        - MultiLatentRaggedPageAttentionV2: MLA ragged paged attention v2

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
    DeepSeekAttention,
    DeepSeekAttentionConfig,
    FlashAttention,
    FlashAttentionConfig,
    FlashMLA,
    FlashMLAConfig,
    GatedDeltaRule,
    GatedDeltaRuleConfig,
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
    MultiLatentRaggedPageAttention,
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2,
    MultiLatentRaggedPageAttentionV2Config,
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
    RaggedGatedDeltaRule,
    RaggedGatedDeltaRuleConfig,
    RaggedPageAttentionv2,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv2TurboQuant,
    RaggedPageAttentionv2TurboQuantConfig,
    RaggedPageAttentionv3,
    RaggedPageAttentionv3Config,
    RaggedPageAttentionv3TurboQuant,
    RaggedPageAttentionv3TurboQuantConfig,
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
    deepseek_attn,
    flash_attention,
    flash_mla,
    gated_delta_rule,
    gdr_attention,
    gla_attention,
    grouped_matmul,
    kda_attention,
    kernel_delta_attention,
    lightning_attention,
    mean_pooling,
    multi_latent_ragged_page_attention,
    multi_latent_ragged_page_attention_v2,
    native_sparse_attention,
    page_attention,
    prefill_page_attention,
    quantized_matmul,
    ragged_decode_attention,
    ragged_gated_delta_rule,
    ragged_page_attention_v2,
    ragged_page_attention_v2_turboquant,
    ragged_page_attention_v3,
    ragged_page_attention_v3_turboquant,
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
    "DeepSeekAttention",
    "DeepSeekAttentionConfig",
    "FlashAttention",
    "FlashAttentionConfig",
    "FlashMLA",
    "FlashMLAConfig",
    "GLAttention",
    "GLAttentionConfig",
    "GatedDeltaRule",
    "GatedDeltaRuleConfig",
    "GroupedMatmul",
    "GroupedMatmulConfig",
    "KernelDeltaAttention",
    "KernelDeltaAttentionConfig",
    "LightningAttention",
    "LightningAttentionConfig",
    "MeanPooling",
    "MeanPoolingConfig",
    "MultiLatentRaggedPageAttention",
    "MultiLatentRaggedPageAttentionConfig",
    "MultiLatentRaggedPageAttentionV2",
    "MultiLatentRaggedPageAttentionV2Config",
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
    "RaggedGatedDeltaRule",
    "RaggedGatedDeltaRuleConfig",
    "RaggedPageAttentionv2",
    "RaggedPageAttentionv2Config",
    "RaggedPageAttentionv2TurboQuant",
    "RaggedPageAttentionv2TurboQuantConfig",
    "RaggedPageAttentionv3",
    "RaggedPageAttentionv3Config",
    "RaggedPageAttentionv3TurboQuant",
    "RaggedPageAttentionv3TurboQuantConfig",
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
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "gated_delta_rule",
    "gdr_attention",
    "gla_attention",
    "grouped_matmul",
    "kda_attention",
    "kernel_delta_attention",
    "lightning_attention",
    "mean_pooling",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "native_sparse_attention",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v2_turboquant",
    "ragged_page_attention_v3",
    "ragged_page_attention_v3_turboquant",
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
