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
    >>> # Simple usage with automatic optimization
    >>> output = flash_attention(query, key, value, causal=True)
    >>>
    >>> # Advanced usage with custom parameters
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
from .flash import FlashAttention, flash_attention
from .gla import GLAttention, gla_attention
from .lightning import LightningAttention, lightning_attention
from .matmul import GroupedMatmul, grouped_matmul
from .mla import FlashMLA, mla_attention
from .native_sparse_attention import NativeSparseAttention, sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .ragged_page_attention import RaggedPageAttention, ragged_page_attention
from .recurrent import RecurrentAttention, recurrent_attention
from .ring import RingAttention, ring_attention
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention

__all__ = (
    "Attention",
    "FlashAttention",
    "FlashMLA",
    "GLAttention",
    "GroupedMatmul",
    "LightningAttention",
    "MeanPooling",
    "NativeSparseAttention",
    "PageAttention",
    "RaggedPageAttention",
    "RecurrentAttention",
    "RingAttention",
    "ScaledDotProductAttention",
    "attention",
    "flash_attention",
    "gla_attention",
    "grouped_matmul",
    "lightning_attention",
    "mean_pooling",
    "mla_attention",
    "page_attention",
    "ragged_page_attention",
    "recurrent_attention",
    "ring_attention",
    "scaled_dot_product_attention",
    "sparse_attention",
)
