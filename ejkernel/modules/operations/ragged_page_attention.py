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


"""Ragged Page Attention module with automatic optimization.

This module implements ragged page attention, combining the benefits of both
ragged (variable-length) sequence processing and paged KV cache management.
This approach is particularly efficient for serving scenarios where sequences
have variable lengths and KV cache is organized in fixed-size pages.

Ragged page attention addresses key challenges in LLM inference:
    - Variable-length sequences without padding overhead
    - Efficient memory management through paged KV cache
    - Dynamic batching with different sequence lengths
    - Memory sharing for beam search and prefix caching

Key Concepts:
    Ragged Layout: Sequences are concatenated without padding, with start
        locations tracking where each sequence begins
    Pages: Fixed-size blocks holding portions of KV cache
    Block Tables: Mapping from logical sequence positions to physical pages

The combination provides:
    - Zero padding overhead (ragged layout)
    - Flexible memory allocation (paged cache)
    - Efficient batching of variable-length sequences
    - Support for dynamic sequence management

Memory Layout:
    Queries: [total_tokens, num_heads, head_dim] (ragged, no padding)
    KV Cache: [num_pages, page_size, num_heads, head_dim] (paged)

Mathematical Foundation:
    For token i in sequence s:
        start_idx = query_start_loc[s]
        end_idx = query_start_loc[s + 1]
        output[i] = attention(Q[start_idx:end_idx], K[pages[s]], V[pages[s]])

This is the most memory-efficient attention variant for serving workloads.
"""

from __future__ import annotations

from typing import Literal

from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
)

from ..base import detect_platform
from .configs import PageAttentionConfig


class RaggedPageAttention(Kernel[PageAttentionConfig, Array]):
    """Ragged Page Attention with custom optimization logic.

    Combines ragged (variable-length) sequence processing with paged KV cache
    management for maximum memory efficiency in serving workloads.

    Features:
        - Zero padding overhead through ragged layout
        - Efficient paged KV cache management
        - Support for variable context lengths per sequence
        - Sliding window attention for long contexts
        - Logit soft capping for numerical stability
        - Attention sink mechanism for improved long-context performance
        - Configurable block sizes and memory limits
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    This implementation is particularly efficient for:
        - LLM serving with dynamic batching
        - Variable-length inference workloads
        - Memory-constrained deployment
        - Scenarios requiring efficient KV cache sharing

    The ragged layout eliminates padding overhead while paged cache
    enables flexible memory management and sharing.
    """

    def __init__(self):
        """Initialize Ragged Page Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="ragged_page_attention")

    def get_impl(self, cfg: PageAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_page_attention", cfg.platform)
        return kernel_registry.get("ragged_page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs pages_per_seq"],
        query_start_loc: Int[Array, "num_seqs_plus_one"],
        num_seqs: Array | int,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        softmax_scale: float | None = None,
        logit_soft_cap: float | None = None,
        compute_dtype: DTypeLike = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        mask_value: float | None = None,
        num_kv_pages_per_block: int | None = None,
        num_queries_per_block: int | None = None,
        vmem_limit_bytes: int | None = None,
        *,
        cfg: PageAttentionConfig,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute ragged page attention over variable-length sequences.

        Computes attention where queries are in ragged (concatenated) format
        and KV cache is organized in pages, providing maximum memory efficiency.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim]
                All sequences concatenated without padding
            kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim]
                Combined key-value cache in page format
            context_lens: Actual context length per sequence [num_seqs]
            block_tables: Page mapping [num_seqs, pages_per_seq] mapping logical
                pages to physical page indices
            query_start_loc: Start indices for each sequence in queries [num_seqs + 1]
                query_start_loc[i] to query_start_loc[i+1] defines sequence i
            num_seqs: Number of sequences in the batch
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            logit_soft_cap: Optional soft cap to bound attention logits
            compute_dtype: Data type for computation (default: bfloat16)
            optimized: Use optimized kernel implementation
            sliding_window: Window size for local attention (None for full attention)
            softmax_aux: Optional attention sink logits for long-context handling
            mask_value: Value to use for masked positions (default: -inf)
            num_kv_pages_per_block: Number of KV pages to process per compute block
            num_queries_per_block: Number of queries to process per compute block
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific)
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [total_tokens, num_q_heads, head_dim] in ragged format

        Note:
            The ragged format eliminates all padding overhead. Combined with paged
            KV cache, this provides the most memory-efficient attention implementation
            for serving workloads with variable-length sequences.

        Example:
            >>>
            >>> query_start_loc = jnp.array([0, 10, 25])
            >>> out = ragged_page_attention(
            ...     queries, kv_pages, context_lens,
            ...     block_tables, query_start_loc, num_seqs=2
            ... )
        """

        if platform is not None:
            cfg = PageAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d if hasattr(cfg, "block_d") else None,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=num_seqs,
            softmax_scale=softmax_scale,
            logit_soft_cap=logit_soft_cap,
            compute_dtype=compute_dtype,
            optimized=optimized,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            mask_value=mask_value,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=vmem_limit_bytes,
        )

    def heuristic_cfg(self, inv: Invocation[PageAttentionConfig, Array]) -> PageAttentionConfig:
        """Provide default configuration optimized for ragged page attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical ragged attention workloads with variable sequence lengths
        """
        return PageAttentionConfig(
            block_q=64,
            block_k=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[PageAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for ragged attention scenarios with
        various batch sizes and sequence lengths.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ragged attention performance depends on the distribution of sequence
            lengths and the page size. Candidates are chosen to work well across
            common serving scenarios.
        """
        block_configs = [(32, 64, 4, 1)]

        candidates = []
        for block_q, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                PageAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_ragged_page_attention_executor: Executor[PageAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True),
        tuner=Tuner(warmup=2, iters=5),
    )
)


def ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    softmax_scale: float | None = None,
    logit_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    *,
    cfg: PageAttentionConfig | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Execute ragged page attention with automatic optimization.

    Ragged page attention efficiently handles variable-length sequences
    in a single batch using flattened token representation and page-based KV cache.

    Args:
        queries: Flattened query tensor [total_tokens, num_q_heads, head_dim]
        kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim]
        context_lens: Context length per sequence [num_seqs]
        block_tables: Block mapping table [num_seqs, pages_per_seq]
        query_start_loc: Start locations for each sequence [num_seqs + 1]
        num_seqs: Number of sequences in the batch
        softmax_scale: Softmax scaling factor
        logit_soft_cap: Soft capping value for logits
        compute_dtype: Computation dtype (default: bfloat16)
        optimized: Use optimized implementation
        sliding_window: Sliding window size for local attention
        softmax_aux: Attention sink logits
        mask_value: Value for masked positions
        num_kv_pages_per_block: Number of KV pages per compute block
        num_queries_per_block: Number of queries per compute block
        vmem_limit_bytes: Memory limit in bytes

            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim]

    Example:
        >>>
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, sliding_window=256
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, optimized=True, logit_soft_cap=50.0
        ... )
        >>>
        >>>
        >>> out = ragged_page_attention(..., platform="triton")
    """
    return _ragged_page_attention_executor(
        RaggedPageAttention(),
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        logit_soft_cap=logit_soft_cap,
        compute_dtype=compute_dtype,
        optimized=optimized,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        _cfg=cfg,
    )
