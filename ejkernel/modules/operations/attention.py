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


"""Standard multi-head attention module with automatic optimization.

This module implements standard multi-head attention (MHA) with XLA-optimized kernels.
It provides a flexible interface supporting various attention patterns including causal
masking, dropout, sliding windows, and variable-length sequences.

Unlike FlashAttention which uses tiling for memory efficiency, this implementation
leverages XLA's compiler optimizations for straightforward attention computation.
"""

from __future__ import annotations

import typing

from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class Attention(Kernel[KernelConfig, Array]):
    """Attention with custom optimization logic.

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
        >>> from ejkernel.modules import Attention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = Attention()
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
        """Initialize  Attention module."""
        super().__init__(op_id="attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="attention",
            platform=detect_platform("attention", cfg.platform),
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
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray = None,
        softmax_aux: Float[Array, "..."] | None = None,
        softmax_scale: float | None = None,
        dtype: jnp.dtype | None = jnp.bfloat16,
        softmax_dtype: jnp.dtype | None = None,
        dropout_prob: float = 0.0,
        sliding_window: int | tuple[int, int] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute flash attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_heads, head_dim]
            value: Value tensor [batch, seq_len_k, num_heads, head_dim]
            attention_mask: Optional attention mask (legacy, prefer bias)
            bias: Optional attention bias tensor
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
            debug: Enable debug mode

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim]
        """
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            init_bias=init_bias,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            dtype=dtype,
            softmax_dtype=softmax_dtype,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return KernelConfig(
            block_q=128,
            block_k=128,
            block_d=64,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates multiple block size configurations for benchmarking to find
        the optimal tiling parameters for the given input shapes.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of candidate configurations to test during autotuning

        Note:
            The autotuning system will benchmark each candidate and select
            the fastest one for the given input configuration.
        """

        block_configs = [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]

        candidates = []
        for block_q, block_k in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=None,
                    num_warps=None,
                    num_stages=None,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_executor = create_default_executor()


def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch 1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray = None,
    softmax_aux: Float[Array, "..."] | None = None,
    softmax_scale: float | None = None,
    dtype: jnp.dtype | None = jnp.bfloat16,
    softmax_dtype: jnp.dtype | None = None,
    dropout_prob: float = 0.0,
    sliding_window: int | tuple[int, int] | None = None,
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
        Attention output with same shape as query

    Example:
        >>>
        >>> out = attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = attention(query, key, value, cum_seqlens_q=cu_q, cum_seqlens_k=cu_k)
    """

    return _executor(
        Attention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        init_bias=init_bias,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        dtype=dtype,
        softmax_dtype=softmax_dtype,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
    )
