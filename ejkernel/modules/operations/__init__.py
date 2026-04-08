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
    - DeepSeekAttention: DeepSeek Sparse Attention (MLA + Lightning Indexer)
    - FlashAttention: Memory-efficient O(N) complexity attention
    - FlashMLA: Multi-head latent attention with low-rank KV compression
    - GLAttention: Gated linear attention mechanism
    - KernelDeltaAttention: Linear attention with delta rule updates (O(N))
    - LightningAttention: Layer-aware attention optimization
    - NativeSparseAttention: Sparse attention with explicit block patterns
    - PageAttention: Paged KV cache for serving workloads
    - PrefillPageAttention: Chunked prefill attention with paged KV cache
    - RaggedPageAttentionv2: Page attention for variable-length sequences
    - RaggedPageAttentionv2TurboQuant: TurboQuant-compressed read-only page attention
    - RaggedPageAttentionv3: Advanced page attention for variable-length sequences v3
    - RaggedPageAttentionv3TurboQuant: TurboQuant-compressed fused page attention
    - RecurrentAttention: Stateful recurrent attention
    - RingAttention: Distributed attention with ring topology
    - ScaledDotProductAttention: Standard scaled dot-product attention

Linear Recurrent Models:
    - RWKV4: Linear attention with time-decay (efficient language modeling)
    - RWKV6: Enhanced RWKV with token-dependent decay
    - RWKV7: Latest RWKV with improved state updates
    - RWKV7Mul: RWKV-7 with multiplicative state updates

State Space Models:
    - StateSpaceV1: Mamba-1 style selective state space model
    - StateSpaceV2: Mamba-2 style selective state space model

Additional Operations:
    - GroupedMatmul: Efficient grouped matrix multiplication (for MoE)
    - MeanPooling: Sequence mean pooling operation
    - MultiLatentRaggedPageAttention: MLA ragged paged attention
    - MultiLatentRaggedPageAttentionV2: MLA ragged paged attention v2
    - QuantizedMatmul: Packed uint32 quantized matmul

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
    >>> # Basic causal attention
    >>> output = flash_attention(query, key, value, causal=True)
    >>>
    >>> # With all features
    >>> output = flash_attention(
    ...     query, key, value,
    ...     softmax_scale=0.125,
    ...     dropout_prob=0.1,
    ...     sliding_window=(256, 256)
    ... )

    >>> # Using MLA for memory-efficient inference
    >>> from ejkernel.modules.operations import flash_mla
    >>> output = flash_mla(query, key_value, w_kc, w_vc, causal=True)

    >>> # Using linear attention for O(N) complexity
    >>> from ejkernel.modules.operations import kernel_delta_attention
    >>> output = kernel_delta_attention(query, key, value, beta, decay)

Note:
    All attention functions automatically handle mixed precision and
    select the best available backend for your hardware.
"""

from .all_gather_matmul import AllGatherMatmul, all_gather_matmul
from .attention import Attention, attention
from .blocksparse_attention import BlockSparseAttention, blocksparse_attention
from .chunked_prefill_paged_decode import ChunkedPrefillPagedDecode, chunked_prefill_paged_decode
from .configs import (
    AllGatherMatmulConfig,
    AttentionConfig,
    BlockSparseAttentionConfig,
    ChunkedPrefillPagedDecodeConfig,
    DecodeAttentionConfig,
    DeepSeekAttentionConfig,
    FlashAttentionConfig,
    FlashMLAConfig,
    GatedDeltaRuleConfig,
    GLAttentionConfig,
    GroupedMatmulConfig,
    KernelDeltaAttentionConfig,
    LightningAttentionConfig,
    MeanPoolingConfig,
    MultiLatentRaggedPageAttentionConfig,
    MultiLatentRaggedPageAttentionV2Config,
    NativeSparseAttentionConfig,
    PageAttentionConfig,
    PrefillPageAttentionConfig,
    QuantizedMatmulConfig,
    RaggedDecodeAttentionConfig,
    RaggedGatedDeltaRuleConfig,
    RaggedPageAttentionv2Config,
    RaggedPageAttentionv2TurboQuantConfig,
    RaggedPageAttentionv3Config,
    RaggedPageAttentionv3TurboQuantConfig,
    RecurrentAttentionConfig,
    ReduceScatterMatmulConfig,
    RingAttentionConfig,
    RWKV4Config,
    RWKV6Config,
    RWKV7Config,
    RWKV7MulConfig,
    ScaledDotProductAttentionConfig,
    StateSpaceV1Config,
    StateSpaceV2Config,
    UnifiedAttentionConfig,
)
from .decode_attention import DecodeAttention, decode_attention
from .deepseek_attn import DeepSeekAttention, deepseek_attn
from .flash_attention import FlashAttention, flash_attention
from .gated_delta_rule import GatedDeltaRule, gated_delta_rule
from .gated_linear_attention import GLAttention, gla_attention
from .grouped_matmul import GroupedMatmul, grouped_matmul
from .kernel_delta_attention import KernelDeltaAttention, kda_attention, kernel_delta_attention
from .lightning_attention import LightningAttention, lightning_attention
from .multi_head_latent_attention import FlashMLA, flash_mla
from .multi_latent_ragged_page_attention import (
    MultiLatentRaggedPageAttention,
    multi_latent_ragged_page_attention,
)
from .multi_latent_ragged_page_attention_v2 import (
    MultiLatentRaggedPageAttentionV2,
    multi_latent_ragged_page_attention_v2,
)
from .native_sparse_attention import NativeSparseAttention, native_sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .prefill_page_attention import PrefillPageAttention, prefill_page_attention
from .quantized_matmul import QuantizedMatmul, quantized_matmul
from .ragged_decode_attention import RaggedDecodeAttention, ragged_decode_attention
from .ragged_gated_delta_rule import RaggedGatedDeltaRule, ragged_gated_delta_rule
from .ragged_page_attention_v2 import RaggedPageAttentionv2, ragged_page_attention_v2
from .ragged_page_attention_v2_turboquant import (
    RaggedPageAttentionv2TurboQuant,
    ragged_page_attention_v2_turboquant,
)
from .ragged_page_attention_v3 import RaggedPageAttentionv3, ragged_page_attention_v3
from .ragged_page_attention_v3_turboquant import (
    RaggedPageAttentionv3TurboQuant,
    ragged_page_attention_v3_turboquant,
)
from .recurrent import RecurrentAttention, recurrent_attention
from .reduce_scatter_matmul import ReduceScatterMatmul, reduce_scatter_matmul
from .ring_attention import RingAttention, ring_attention
from .rwkv4 import RWKV4, rwkv4
from .rwkv6 import RWKV6, rwkv6
from .rwkv7 import RWKV7, RWKV7Mul, rwkv7, rwkv7_mul
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention
from .state_space_v1 import StateSpaceV1, state_space_v1
from .state_space_v2 import StateSpaceV2, state_space_v2
from .unified_attention import UnifiedAttention, unified_attention

gdr_attention = gated_delta_rule

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
