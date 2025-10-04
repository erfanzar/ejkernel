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
        - PageAttention: Paged KV cache attention for serving
        - RaggedPageAttention: Variable-length page attention
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
    >>> # Create executor with caching and autotuning
    >>> executor = create_default_executor("/tmp/kernel_cache")
    >>>
    >>> # Use module
    >>> attn = FlashAttention()
    >>> output = executor(attn, q, k, v, causal=True)
"""

from .base import KernelConfig, create_default_executor
from .operations import (
    Attention,
    FlashAttention,
    FlashMLA,
    GLAttention,
    GroupedMatmul,
    LightningAttention,
    MeanPooling,
    NativeSparseAttention,
    PageAttention,
    RaggedPageAttention,
    RecurrentAttention,
    RingAttention,
    attention,
    flash_attention,
    gla_attention,
    grouped_matmul,
    lightning_attention,
    mean_pooling,
    mla_attention,
    page_attention,
    ragged_page_attention,
    recurrent_attention,
    ring_attention,
    sparse_attention,
)

__all__ = (
    "Attention",
    "FlashAttention",
    "FlashMLA",
    "GLAttention",
    "GroupedMatmul",
    "KernelConfig",
    "LightningAttention",
    "MeanPooling",
    "NativeSparseAttention",
    "PageAttention",
    "RaggedPageAttention",
    "RecurrentAttention",
    "RingAttention",
    "attention",
    "create_default_executor",
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
    "sparse_attention",
)
