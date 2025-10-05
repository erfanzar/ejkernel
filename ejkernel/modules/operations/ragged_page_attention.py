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

from jax import numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class RaggedPageAttention(Kernel[KernelConfig, Array]):
    """Ragged Page Attention with custom optimization logic.

    Page attention optimized for ragged (variable-length) sequences
    with efficient memory layout and computation.
    """

    def __init__(self):
        """Initialize Ragged Page Attention module."""
        super().__init__(op_id="ragged_page_attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry."""
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
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
        softmax_scale: float | None = None,
        logit_soft_cap: float | None = None,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        optimized: bool = False,
        sliding_window: int | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        mask_value: float | None = None,
        num_kv_pages_per_block: int | None = None,
        num_queries_per_block: int | None = None,
        vmem_limit_bytes: int | None = None,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute ragged page attention."""
        # Override platform in config if specified
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


_ragged_page_attention_executor = create_default_executor()


def ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    softmax_scale: float | None = None,
    logit_soft_cap: float | None = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
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
        >>> # Standard ragged page attention
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs
        ... )
        >>>
        >>> # With sliding window
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, sliding_window=256
        ... )
        >>>
        >>> # Optimized mode with soft capping
        >>> out = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs, optimized=True, logit_soft_cap=50.0
        ... )
            >>>
        >>> # Force specific platform
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
    )
