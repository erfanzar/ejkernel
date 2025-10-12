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


"""ScaledDotProductAttention module with automatic optimization."""

from __future__ import annotations

import typing

from jaxtyping import Array, Bool, Float, Int

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
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import AttentionConfig


class ScaledDotProductAttention(Kernel[AttentionConfig, Array]):
    """ScaledDotProductAttention with custom optimization logic.

    Supports causal masking, dropout, sliding windows, and variable-length sequences.

    Features:
        - Automatic platform/backend selection (XLA Only ;0)
        - Configuration caching for consistent performance
        - Optional autotuning to find optimal implementation
        - Custom gradient support for efficient backpropagation
        - Support for variable-length sequences via cumulative sequence lengths
        - Sliding window attention for local attention patterns
        - Logits soft capping for numerical stability

    Example:
        >>> from ejkernel.modules import ScaledDotProductAttention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = ScaledDotProductAttention()
        >>>
        >>>
        >>> output = executor(attn, query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>>
        >>> output = executor(
        ...     attn, query, key, value,...
        ... )
        >>>
        >>>
        >>> output = executor(attn, query, key, value, sliding_window=(256, 256))
    """

    def __init__(self):
        """Initialize  ScaledDotProductAttention module."""
        super().__init__(op_id="scaled_dot_product_attention")

    def get_impl(self, cfg: AttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="scaled_dot_product_attention",
            platform=detect_platform("scaled_dot_product_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,
        cum_seqlens_k: Int[Array, "batch"] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: AttentionConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute scaled_dot_product_attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_heads, head_dim]
            value: Value tensor [batch, seq_len_k, num_heads, head_dim]
            attention_mask: Optional scaled_dot_product_attention mask (legacy, prefer bias)
            bias: Optional scaled_dot_product_attention bias tensor
            softmax_scale: Scaling factor for attention scores
            dropout_prob: Dropout probability for attention weights
            causal: Whether to apply causal masking
            dropout_seed: Random seed for dropout
            cum_seqlens_q: Cumulative sequence lengths for variable-length queries
            cum_seqlens_k: Cumulative sequence lengths for variable-length keys
            sliding_window: Window size for local attention
            logits_soft_cap: Optional soft cap value for logits
            softmax_aux: Optional attention sink logits
            cfg: Configuration object specifying platform/backend
            segment_ids: Segment IDs for grouped sequences (TPU-specific)
            block_sizes: Block sizes for kernel execution (TPU-specific)

        Returns:
            ScaledDotProductAttention output [batch, seq_len_q, num_heads, head_dim]
        """

        if platform is not None:
            cfg = AttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        impl = self.get_impl(cfg)

        return impl(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            init_bias=init_bias,
            sliding_window=sliding_window,
            causal=causal,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
        )

    def heuristic_cfg(self, inv: Invocation[AttentionConfig, Array]) -> AttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return AttentionConfig(
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[AttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        This operation uses XLA primitives directly without tunable block sizes,
        so autotuning provides no benefit. Returns empty list to skip autotuning.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Empty list - no candidates to autotune since XLA handles optimization

        Note:
            XLA's scaled_dot_product_attention primitive is not parameterized by
            block sizes, so there are no meaningful configurations to benchmark.
        """

        return []


_executor: Executor[AttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("sdpa"),
    )
)


def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Execute flash attention with automatic optimization.

    Convenience function that uses a default executor and flash attention module.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_heads, head_dim]
        attention_mask: Optional attention mask (legacy, prefer bias)
        bias: Optional attention bias tensor
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights
        causal: Whether to apply causal masking
        dropout_seed: Random seed for dropout
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Window size for local attention (int or (left, right) tuple)
        logits_soft_cap: Optional soft cap value for logits

    Returns:
        ScaledDotProductAttention output with same shape as query

    Example:
        >>>
        >>> out = scaled_dot_product_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = scaled_dot_product_attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = scaled_dot_product_attention(query, key, value, cum_seqlens_q=cu_q, cum_seqlens_k=cu_k)
    """

    return _executor(
        ScaledDotProductAttention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        init_bias=init_bias,
        sliding_window=sliding_window,
        causal=causal,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        platform=platform,
    )
