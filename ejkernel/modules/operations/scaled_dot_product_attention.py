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


"""Scaled Dot Product Attention module with automatic optimization.

This module implements the standard scaled dot-product attention mechanism,
which is the fundamental building block of transformer architectures. It computes:

    Attention(Q,K,V) = softmax((Q @ K^T) / sqrt(d_k)) @ V

where Q, K, V are the query, key, and value matrices, and d_k is the key dimension.

This implementation provides:
    - Automatic platform selection (XLA, Triton, Pallas, CUDA)
    - Support for various attention patterns (causal, sliding window)
    - Variable-length sequence handling
    - Distributed execution via shard_map
    - Attention biasing and masking
    - Numerical stability through soft capping

Unlike FlashAttention which uses tiling for memory efficiency, this implementation
relies on platform-specific optimizations (e.g., XLA's attention primitive).
"""

from __future__ import annotations

import typing

import jax
from jax import shard_map
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
from ejkernel.types.mask import MaskInfo

from ..base import detect_platform
from .configs import ScaledDotProductAttentionConfig


class ScaledDotProductAttention(Kernel[ScaledDotProductAttentionConfig, Array]):
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
        """Initialize ScaledDotProductAttention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="scaled_dot_product_attention")

    def get_impl(self, cfg: ScaledDotProductAttentionConfig):
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
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,
        cum_seqlens_k: Int[Array, "batch"] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: ScaledDotProductAttentionConfig,
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
            cfg = ScaledDotProductAttentionConfig(
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

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch kv_len num_kv_heads head_dim"],
        value: Float[Array, "batch kv_len num_kv_heads head_dim"],
        attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
        bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
        cum_seqlens_q: Int[Array, "batch"] | None = None,
        cum_seqlens_k: Int[Array, "batch"] | None = None,
        *,
        mesh: jax.sharding.Mesh,
        in_specs: tuple[jax.sharding.PartitionSpec, ...],
        out_specs: jax.sharding.PartitionSpec,
        check_vma: bool = False,
        cfg: ScaledDotProductAttentionConfig,
        init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        sliding_window: int | tuple[int, int] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    ):
        """Create a shard_map wrapper for distributed ScaledDotProductAttention execution.

        Enables efficient distributed execution of attention across multiple devices
        using JAX's shard_map functionality. This is particularly useful for model
        parallelism and handling very large attention computations.

        Args:
            query: Query tensor [batch, seq_len, num_q_heads, head_dim]
            key: Key tensor [batch, kv_len, num_kv_heads, head_dim]
            value: Value tensor [batch, kv_len, num_kv_heads, head_dim]
            attention_mask: Optional attention mask [batch, 1, seq_len, kv_len]
            bias: Optional attention bias [batch, num_heads, seq_len, kv_len]
            cum_seqlens_q: Cumulative sequence lengths for queries [batch]
            cum_seqlens_k: Cumulative sequence lengths for keys [batch]
            mesh: JAX mesh defining device topology for distributed execution
            in_specs: Partition specifications for each input tensor
            out_specs: Partition specification for output tensor
            check_vma: Whether to check for virtual memory access patterns
            cfg: Configuration object specifying platform/backend
            init_bias: Optional callable to initialize bias on-device
            softmax_scale: Scaling factor for attention scores
            causal: Whether to apply causal masking
            sliding_window: Window size for local attention
            platform: Optional platform override

        Returns:
            Tuple of (shard_map function, call args) where:
                - shard_map function: Callable for distributed execution
                - call args: Tuple of arguments to pass to the shard_map function

        Note:
            The shard_map wrapper handles device placement and communication
            automatically based on the provided mesh and partition specs.
        """
        impl = self.get_impl(cfg)

        def _wrapped_sdpa(
            query,
            key,
            value,
            bias,
            cum_seqlens_q,
            cum_seqlens_k,
            attention_mask,
        ):
            return impl(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                bias=bias,
                cum_seqlens_q=cum_seqlens_q,
                cum_seqlens_k=cum_seqlens_k,
                init_bias=init_bias,
                softmax_scale=softmax_scale,
                causal=causal,
                sliding_window=sliding_window,
            )

        call_args = (
            query,
            key,
            value,
            bias,
            cum_seqlens_q,
            cum_seqlens_k,
            attention_mask,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_sdpa,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def heuristic_cfg(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]) -> ScaledDotProductAttentionConfig:
        """Provide default configuration based on invocation context.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration for platform/backend selection
        """

        return ScaledDotProductAttentionConfig(
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[ScaledDotProductAttentionConfig, Array]):
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


_executor: Executor[ScaledDotProductAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="heuristics", validate_backward=True),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("sdpa"),
    )
)


def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
    /,
    *,
    mask_info: MaskInfo | None = None,
    init_bias: typing.Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    mesh: jax.sharding.Mesh | None = None,
    in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None,
    out_specs: jax.sharding.PartitionSpec | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Execute scaled dot product attention with automatic optimization.

    Convenience function that uses a default executor and flash attention module.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_heads, head_dim]
        mask_info: Optional MaskInfo containing attention mask and/or segment IDs
        bias: Optional attention bias tensor
        softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights
        causal: Whether to apply causal masking
        dropout_seed: Random seed for dropout
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Window size for local attention (int or (left, right) tuple)
        logits_soft_cap: Optional soft cap value for logits
        mesh: JAX device mesh for shard_map execution (optional)
        in_specs: Input partition specs for shard_map (optional)
        out_specs: Output partition spec for shard_map (optional)

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

    attention_mask = None

    if mask_info is not None:
        attention_mask = mask_info.get_or_compute_attention_mask()

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

        if mask_info is None:
            in_specs = (*in_specs, None)
        else:
            shardings = mask_info.get_shardings(False, mesh=mesh)
            in_specs = (*in_specs, shardings.attention_mask)
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
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
    )
