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


"""Page Attention modules with automatic optimization."""

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
    """

    def __init__(self):
        """Initialize Page Attention module."""
        super().__init__(op_id="page_attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry."""
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
        """Execute page attention."""

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
        """Provide default configuration with block sizes."""
        return KernelConfig(
            block_q=64,
            block_k=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning."""
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
