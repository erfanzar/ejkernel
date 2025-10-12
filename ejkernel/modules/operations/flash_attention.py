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


"""Flash Attention module with automatic optimization."""

from __future__ import annotations

from typing import Literal

from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

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
from .configs import FlashAttentionConfig


class FlashAttention(Kernel[FlashAttentionConfig, Array]):
    """Flash Attention with custom optimization logic.

    Memory-efficient exact attention with O(N) memory complexity.
    Supports causal masking, dropout, sliding windows, and variable-length sequences.

    Features:
        - Automatic platform/backend selection (Triton/Pallas/XLA)
        - Configuration caching for consistent performance
        - Optional autotuning to find optimal implementation
        - Custom gradient support for efficient backpropagation
        - Support for variable-length sequences via cumulative sequence lengths
        - Sliding window attention for local attention patterns
        - Logits soft capping for numerical stability

    Example:
        >>> from ejkernel.modules import FlashAttention, create_default_executor
        >>>
        >>>
        >>> executor = create_default_executor()
        >>> attn = FlashAttention()
        >>>
        >>>
        >>> output = executor(attn, query, key, value, causal=True, softmax_scale=0.125)
        >>>
        >>>
        >>> output = executor(
        ...     attn, query, key, value,
        ...     cum_seqlens_q=cu_seqlens_q,
        ...     cum_seqlens_k=cu_seqlens_k
        ... )
        >>>
        >>>
        >>> output = executor(attn, query, key, value, sliding_window=(256, 256))
    """

    def __init__(self):
        """Initialize Flash Attention module."""
        super().__init__(op_id="flash_attention")

    def get_impl(self, cfg: FlashAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation

        Raises:
            ValueError: If no matching implementation is found
        """
        return kernel_registry.get(
            algorithm="flash_attention",
            platform=detect_platform("flash_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        softmax_scale: float | None = None,
        dropout_prob: float = 0.0,
        causal: bool = False,
        dropout_seed: int | None = None,
        cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
        cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        normalize_output: bool = True,
        precision: lax.PrecisionLike = lax.Precision.DEFAULT,
        logits_dtype: DTypeLike = jnp.float32,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        cfg: FlashAttentionConfig,
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
            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
            cfg: Configuration object specifying platform/backend
            segment_ids: Segment IDs for grouped sequences (TPU-specific)
            block_sizes: Block sizes for kernel execution (TPU-specific)

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim]
        """

        if platform is not None:
            cfg = FlashAttentionConfig(
                chunk_size_q=cfg.chunk_size_q,
                chunk_size_k=cfg.chunk_size_k,
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
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            cum_seqlens_q=cum_seqlens_q,
            cum_seqlens_k=cum_seqlens_k,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
            normalize_output=normalize_output,
            precision=precision,
            logits_dtype=logits_dtype,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            chunk_size_q=cfg.chunk_size_q,
            chunk_size_k=cfg.chunk_size_k,
        )

    def heuristic_cfg(self, inv: Invocation[FlashAttentionConfig, Array]) -> FlashAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return FlashAttentionConfig(
            chunk_size_q=128,
            chunk_size_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FlashAttentionConfig, Array]):
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
        for chunk_q, chunk_k in block_configs:
            candidates.append(
                FlashAttentionConfig(
                    chunk_size_q=chunk_q,
                    chunk_size_k=chunk_k,
                    num_warps=4,
                    num_stages=2,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for autotuning.

        GPU/Triton kernels benefit from smaller blocks with more warps and stages.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of GPU-optimized candidate configurations
        """
        configs = []
        for chunk_q in [128, 256]:
            for chunk_k in [128, 256]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3]:
                        configs.append(
                            FlashAttentionConfig(
                                chunk_size_q=chunk_q,
                                chunk_size_k=chunk_k,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                platform="triton",
                                backend="gpu",
                            )
                        )
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning.

        TPU/Pallas kernels benefit from larger blocks with fewer stages.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of TPU-optimized candidate configurations
        """
        configs = []
        for chunk_q in [256, 512]:
            for chunk_k in [256, 512]:
                configs.append(
                    FlashAttentionConfig(
                        chunk_size_q=chunk_q,
                        chunk_size_k=chunk_k,
                        num_warps=4,
                        num_stages=1,
                        platform="pallas",
                        backend="tpu",
                    )
                )
        return configs

    def candidate_cfgs_xla(self, inv: Invocation[FlashAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        XLA implementations use medium blocks with moderate parallelism.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of XLA-optimized candidate configurations
        """
        configs = []
        for chunk_q in [128, 256]:
            for chunk_k in [128, 256]:
                configs.append(
                    FlashAttentionConfig(
                        chunk_size_q=chunk_q,
                        chunk_size_k=chunk_k,
                        num_warps=4,
                        num_stages=2,
                        platform="xla",
                        backend="any",
                    )
                )
        return configs


_flash_executor: Executor[FlashAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("flash-attn"),
    )
)


def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    *,
    cfg: FlashAttentionConfig | None = None,
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
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = flash_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, cum_seqlens_q=cu_q, cum_seqlens_k=cu_k)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, platform="triton")
    """

    return _flash_executor(
        FlashAttention(),
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        platform=platform,
        _cfg=cfg,
    )
