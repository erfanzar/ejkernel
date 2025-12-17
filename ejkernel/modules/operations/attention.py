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

import typing as tp

from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from ejkernel.kernels._registry import kernel_registry
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
from ejkernel.types.mask import MaskInfo

from ..base import detect_platform
from .configs import AttentionConfig


class Attention(Kernel[AttentionConfig, tuple[Array, Array]]):
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
            algorithm="attention",
            platform=detect_platform("attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_q_heads vhead_dim"],
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        softmax_scale: float | None = None,
        logits_soft_cap: float | None = None,
        dtype: DTypeLike | None = jnp.bfloat16,
        softmax_dtype: DTypeLike | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        *,
        cfg: AttentionConfig,
    ) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads seq_len kv_len"]]:
        """Execute flash attention with the given configuration.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_heads, head_dim]
            value: Value tensor [batch, seq_len_k, num_heads, head_dim]
            attention_mask: Optional attention mask (legacy, prefer bias)
            bias: Optional attention bias tensor
            softmax_scale: Scaling factor for attention scores
            dropout_prob: Dropout probability for attention weights
            sliding_window: Window size for local attention
            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
            cfg: Configuration object specifying platform/backend

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
            logits_soft_cap=logits_soft_cap,
            dropout_prob=dropout_prob,
            init_bias=init_bias,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            dtype=dtype,
            softmax_dtype=softmax_dtype,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            causal=causal,
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
            XLA's attention primitive is not parameterized by block sizes,
            so there are no meaningful configurations to benchmark.
        """

        return []


_executor: Executor[AttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=True),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("attention"),
    )
)


def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_q_heads vhead_dim"],
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    dropout_rng: PRNGKeyArray | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    init_bias: tp.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    dtype: DTypeLike | None = jnp.bfloat16,
    softmax_dtype: DTypeLike | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> Float[Array, "batch seq_len num_q_heads vhead_dim"]:
    """Execute flash attention with automatic optimization.

    Convenience function that uses a default executor and flash attention module.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_heads, head_dim]
        mask_info: Optional MaskInfo containing attention mask and/or segment IDs
        bias: Optional attention bias tensor
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights
        sliding_window: Window size for local attention (int or (left, right) tuple)
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = attention(query, key, value)
        >>>
        >>>
        >>> out = attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = attention(query, key, value, platform="xla")
    """

    attention_mask = None

    if mask_info is not None:
        attention_mask = mask_info.get_or_compute_attention_mask()

    return _executor(
        Attention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        dropout_prob=dropout_prob,
        init_bias=init_bias,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        dtype=dtype,
        softmax_dtype=softmax_dtype,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        causal=causal,
    )
