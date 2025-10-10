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


"""Page Attention module with automatic optimization.

This module implements Page Attention, a specialized attention mechanism designed
for efficient KV cache management in serving and inference workloads. Page Attention
organizes the KV cache in fixed-size blocks (pages), enabling:
    - Dynamic memory allocation without pre-allocating for max sequence length
    - Efficient memory sharing across sequences (e.g., for beam search or prefix caching)
    - Reduced memory fragmentation compared to contiguous allocation
    - Better GPU memory utilization through page-level management

Page Attention is particularly valuable for:
    - LLM serving with variable-length sequences
    - Batch inference with dynamic batching
    - Memory-constrained deployment scenarios
    - Systems requiring efficient KV cache sharing

Key Concepts:
    Pages: Fixed-size blocks holding a portion of KV cache (e.g., 16 or 32 tokens)
    Block Tables: Mapping from logical sequence positions to physical page indices
    Context Lengths: Actual sequence lengths (excluding padding)

The paged approach enables:
    - Near-zero memory waste (only last page per sequence may be partially filled)
    - Easy insertion/deletion of sequences without memory reshuffling
    - Natural support for prefix sharing in beam search

Mathematical Foundation:
    For query position i:
        output[i] = sum_{j in valid_pages} softmax(Q[i] @ K[pages[j]].T) @ V[pages[j]]

    Where valid_pages are determined by block_tables and context_lens.

Memory Layout:
    Instead of: [seq_len, num_heads, head_dim] (contiguous per sequence)
    Use: [num_pages, page_size, num_heads, head_dim] (page-based allocation)

References:
    Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention"
    https://arxiv.org/abs/2309.06180 (vLLM paper)
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class PageAttention(Kernel[KernelConfig, Array]):
    """Page Attention with custom optimization logic.

    Efficient attention over paged KV cache for serving workloads.
    Optimized for dynamic batching with variable context lengths.

    Features:
        - Paged KV cache management for memory efficiency
        - Support for variable context lengths per sequence
        - Automatic partitioning for long contexts
        - Multi-split attention for improved throughput
        - Optimized for inference and serving workloads
        - Logit soft capping for numerical stability
        - Configurable pages per compute block
        - TPU megacore mode support

    The paged layout provides:
        - O(1) insertion/deletion of sequences
        - Efficient prefix sharing for beam search
        - Minimal memory fragmentation
        - Better batch utilization through dynamic allocation
    """

    def __init__(self):
        """Initialize Page Attention module.

        Sets up the kernel for paged KV cache attention computation
        with automatic platform selection and optimization.
        """
        super().__init__(op_id="page_attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("page_attention", cfg.platform)
        return kernel_registry.get("page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "num_seqs num_heads head_dim"],
        key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
        context_lens: Int[Array, "num_seqs"],
        block_tables: Int[Array, "num_seqs max_blocks"],
        attn_scale: float | None = None,
        max_context_len: int | None = None,
        num_splits: int = 0,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
        mask_value: float = -2.381976426469702e38,
        attn_logits_soft_cap: float | None = None,
        pages_per_compute_block: int | None = None,
        megacore_mode: str | None = None,
        inline_seq_dim: bool = True,
    ) -> Float[Array, "num_seqs num_heads head_dim"]:
        """Execute page attention over paged KV cache.

        Computes attention where the KV cache is organized in fixed-size pages,
        with each sequence's tokens potentially scattered across non-contiguous pages.

        Args:
            query: Query tensor [num_seqs, num_heads, head_dim] for current decode step
            key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
            value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
            context_lens: Actual context length per sequence [num_seqs]
            block_tables: Page index mapping [num_seqs, max_blocks] where block_tables[i, j]
                gives the physical page index for sequence i's jth logical block
            attn_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
            max_context_len: Maximum context length across all sequences
            num_splits: Number of splits for partitioned attention (0 = auto, 1 = no split)
            mask_value: Value used for masked positions (default: -inf)
            attn_logits_soft_cap: Optional soft cap for attention logits
            pages_per_compute_block: Number of pages to process per compute block
            megacore_mode: TPU-specific megacore execution mode
            inline_seq_dim: Whether to inline the sequence dimension
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [num_seqs, num_heads, head_dim]

        Note:
            Block tables define the mapping from logical to physical pages:
                logical_page_idx = position // block_size
                physical_page_idx = block_tables[seq_idx, logical_page_idx]

        Example:
            >>>
            >>>
            >>> block_tables = jnp.array([[3, 7, 0], [1, 5, 0]])
            >>> context_lens = jnp.array([32, 24])
            >>> out = page_attention(q, k_cache, v_cache, context_lens, block_tables)
        """

        if platform is not None:
            cfg = KernelConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d if hasattr(cfg, "block_d") else None,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            context_lens=context_lens,
            block_tables=block_tables,
            attn_scale=attn_scale,
            max_context_len=max_context_len,
            num_splits=num_splits,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            pages_per_compute_block=pages_per_compute_block,
            megacore_mode=megacore_mode,
            inline_seq_dim=inline_seq_dim,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration optimized for paged attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default KernelConfig with block sizes suitable for typical
            serving workloads with variable context lengths
        """
        return KernelConfig(
            block_q=64,
            block_k=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for different batch sizes and
        context lengths commonly seen in serving scenarios.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Page attention performance is sensitive to the ratio of context_len
            to page_size and the number of sequences in the batch.
        """
        block_configs = [
            (32, 64, 4, 1),
            (64, 64, 4, 1),
            (64, 128, 8, 2),
            (128, 128, 8, 2),
        ]

        candidates = []
        for block_q, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_page_attention_executor = create_default_executor()


def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    mask_value: float = -2.381976426469702e38,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
    platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Execute page attention with automatic optimization.

    Page attention performs efficient attention computation over paged KV cache
    for serving and inference workloads with dynamic batching.

    Args:
        query: Query tensor [num_seqs, num_heads, head_dim]
        key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
        context_lens: Context length per sequence [num_seqs]
        block_tables: Block mapping table [num_seqs, max_blocks]
        attn_scale: Attention scaling factor
        max_context_len: Maximum context length across all sequences
        num_splits: Number of splits for partitioned attention (0=auto)
        mask_value: Value for masked positions (default: -inf)
        attn_logits_soft_cap: Soft cap value for attention logits
        pages_per_compute_block: Pages per compute block
        megacore_mode: Megacore execution mode
        inline_seq_dim: Whether to inline sequence dimension

            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output [num_seqs, num_heads, head_dim]

    Example:
        >>>
        >>> out = page_attention(query, key_cache, value_cache, context_lens, block_tables)
        >>>
        >>>
        >>> out = page_attention(
        ...     query, key_cache, value_cache, context_lens, block_tables,
        ...     num_splits=4, max_context_len=8192
        ... )
        >>>
        >>>
        >>> out = page_attention(
        ...     query, key_cache, value_cache, context_lens, block_tables,
        ...     attn_logits_soft_cap=50.0
        ... )
            >>>
        >>>
        >>> out = page_attention(..., platform="triton")
    """
    return _page_attention_executor(
        PageAttention(),
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        max_context_len=max_context_len,
        num_splits=num_splits,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        pages_per_compute_block=pages_per_compute_block,
        megacore_mode=megacore_mode,
        inline_seq_dim=inline_seq_dim,
    )
