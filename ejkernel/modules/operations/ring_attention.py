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


"""Ring Attention module with automatic optimization.

This module implements Ring Attention, a distributed attention mechanism that enables
efficient processing of extremely long sequences by distributing computation across
multiple devices in a ring topology. Unlike standard attention which requires all KV
pairs to fit in a single device's memory, Ring Attention overlaps communication and
computation through pipelining.

Ring Attention is particularly valuable for:
    - Ultra-long sequence processing (100K+ tokens)
    - Training large language models with long contexts
    - Distributed inference scenarios
    - Memory-constrained environments requiring sequence parallelism

Key Innovation:
    Ring Attention partitions the KV pairs across devices and uses a ring-based
    communication pattern to stream KV blocks through each device. Each device:
    1. Computes attention with its local KV block
    2. Passes the KV block to the next device in the ring
    3. Receives the next KV block from the previous device
    4. Continues until all KV blocks have been processed

    This achieves O(N) memory per device while maintaining O(N^2) computation.

Mathematical Foundation:
    For a sequence of length N split across D devices:
    - Each device holds N/D query tokens
    - KV pairs are rotated through the ring
    - Attention is computed incrementally: softmax_i = exp(QK_i^T) / sum_j(exp(QK_j^T))
    - Running statistics (max, sum) are maintained for numerical stability

Communication Pattern:
    Device 0: KV_0 -> KV_1 -> ... -> KV_{D-1}
    Device 1: KV_1 -> KV_2 -> ... -> KV_0
    Device i: KV_i -> KV_{i+1} -> ... -> KV_{i-1} (mod D)

Performance Characteristics:
    - Memory: O(N/D) per device vs O(N) for standard attention
    - Computation: O(N^2/D) per device (same asymptotic cost)
    - Communication: O(N) per device (bandwidth-efficient with overlap)

References:
    Liu et al., "Ring Attention with Blockwise Transformers for Near-Infinite Context"
    https://arxiv.org/abs/2310.01889
"""

from __future__ import annotations

from typing import Literal

import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

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
from .configs import RingAttentionConfig


class RingAttention(Kernel[RingAttentionConfig, Array]):
    """Ring Attention with custom optimization logic.

    Implements distributed attention using ring communication topology for
    processing ultra-long sequences across multiple devices with memory efficiency.

    Features:
        - Distributed KV processing via ring communication
        - Overlapped computation and communication for efficiency
        - Causal and non-causal attention support
        - Sliding window attention for local patterns
        - Attention sink mechanism for long-context stability
        - Configurable chunk sizes for memory-computation tradeoffs
        - Gradient checkpointing support for training
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The implementation maintains numerical stability through:
        - Online softmax with running max/sum statistics
        - Logit soft capping to prevent overflow
        - Float32 logit accumulation (configurable)

    Typical Usage Patterns:
        - Multi-GPU training with sequence parallelism
        - Long-context inference on multiple devices
        - Blockwise transformer architectures
    """

    def __init__(self):
        """Initialize Ring Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and distributed execution management.
        """
        super().__init__(op_id="ring_attention")

    def get_impl(self, cfg: RingAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ring attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ring_attention", cfg.platform)
        return kernel_registry.get("ring_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len_q num_heads head_dim"],
        key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
        bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
        q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
        kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
        softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        cache_idx=None,
        axis_name: str | None = None,
        float32_logits: bool = True,
        softmax_scale: float | None = None,
        query_chunk_size: int = 512,
        key_chunk_size: int = 512,
        causal_block_size: int | None = None,
        deterministic: bool = True,
        dropout_rng: PRNGKeyArray | None = None,
        pdrop: float = 0.0,
        dtype: DTypeLike = jnp.float32,
        policy=jax.checkpoint_policies.nothing_saveable,
        precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
        prevent_cse: bool = True,
        sliding_window: int | tuple[int, int] | None = None,
        logit_soft_cap: float | None = None,
        attention_sink_size: int = 0,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: RingAttentionConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute ring attention with distributed KV processing.

        Computes attention across devices using ring communication pattern,
        enabling efficient processing of sequences that don't fit in single device memory.

        Args:
            query: Query tensor [batch, seq_len_q, num_heads, head_dim]
            key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim] (distributed)
            value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim] (distributed)
            bias: Optional attention bias [batch, num_heads, seq_len_q, seq_len_k]
            q_segment_ids: Optional query segment IDs [batch, seq_len_q]
            kv_segment_ids: Optional KV segment IDs [batch, seq_len_k]
            softmax_aux: Optional attention sink logits for long-context stability
            cache_idx: Optional cache index for incremental decoding
            axis_name: Name of the axis for collective operations (required for multi-device)
            float32_logits: Use float32 for logit computation (default: True)
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            query_chunk_size: Size of query chunks for tiling (default: 512)
            key_chunk_size: Size of key chunks for tiling (default: 512)
            causal_block_size: Block size for causal masking (None = no causal)
            deterministic: Use deterministic dropout (default: True)
            dropout_rng: PRNG key for dropout
            pdrop: Dropout probability (default: 0.0)
            dtype: Computation dtype (default: float32)
            policy: Gradient checkpointing policy
            precision: Matrix multiplication precision setting
            prevent_cse: Prevent common subexpression elimination (default: True)
            sliding_window: Window size for local attention (int or (left, right) tuple)
            logit_soft_cap: Soft cap value to bound attention logits
            attention_sink_size: Number of sink tokens for attention stability
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len_q, num_heads, head_dim]

        Note:
            Ring attention requires proper device mesh setup with the specified axis_name.
            Each device processes a slice of the sequence and communicates KV pairs
            through the ring topology.

        Example:
            >>>
            >>> mesh = jax.sharding.Mesh(devices, axis_names=['sp'])
            >>>
            >>>
            >>> with mesh:
            ...     out = ring_attention(q, k, v, axis_name='sp')
        """

        if platform is not None:
            cfg = RingAttentionConfig(
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
            query=query,
            key=key,
            value=value,
            bias=bias,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=softmax_aux,
            cache_idx=cache_idx,
            axis_name=axis_name,
            float32_logits=float32_logits,
            softmax_scale=softmax_scale,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            causal_block_size=causal_block_size,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            pdrop=pdrop,
            dtype=dtype,
            policy=policy,
            precision=precision,
            prevent_cse=prevent_cse,
            sliding_window=sliding_window,
            logit_soft_cap=logit_soft_cap,
            attention_sink_size=attention_sink_size,
        )

    def heuristic_cfg(self, inv: Invocation[RingAttentionConfig, Array]) -> RingAttentionConfig:
        """Provide default configuration optimized for ring attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default KernelConfig with block sizes balanced for communication
            and computation overlap in distributed settings
        """
        return RingAttentionConfig(
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RingAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for different sequence lengths and
        device counts, balancing chunk size with communication overhead.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ring attention performance is sensitive to chunk sizes relative
            to sequence length per device and communication bandwidth.
        """
        block_configs = [
            (64, 64, 4, 1),
            (128, 128, 4, 2),
            (256, 128, 8, 2),
        ]

        candidates = []
        for block_q, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                RingAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_ring_executor: Executor[RingAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True, cache_miss_fallback="autotune", validate_backward=True),
        tuner=Tuner(warmup=5, iters=50),
        persistent=PersistentCache("ring-attention"),
    )
)


def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    cache_idx=None,
    axis_name: str | None = None,
    float32_logits: bool = True,
    softmax_scale: float | None = None,
    query_chunk_size: int = 512,
    key_chunk_size: int = 512,
    causal_block_size: int | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    pdrop: float = 0.0,
    dtype: DTypeLike = jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    prevent_cse: bool = True,
    sliding_window: int | tuple[int, int] | None = None,
    logit_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    *,
    cfg: RingAttentionConfig | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Execute ring attention with automatic optimization.

    Ring attention distributes attention computation across devices in a ring topology,
    enabling efficient processing of very long sequences through communication-efficient
    parallelization.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        bias: Optional attention bias tensor
        q_segment_ids: Segment IDs for queries
        kv_segment_ids: Segment IDs for keys/values
        softmax_aux: Auxiliary softmax values
        cache_idx: Cache index for inference
        axis_name: Name of the axis for collective operations
        float32_logits: Use float32 for logit computation
        softmax_scale: Scaling factor for attention scores
        query_chunk_size: Chunk size for queries (default: 512)
        key_chunk_size: Chunk size for keys (default: 512)
        causal_block_size: Block size for causal masking
        deterministic: Use deterministic dropout
        dropout_rng: RNG for dropout
        pdrop: Dropout probability
        dtype: Data type for computation
        policy: Sharding policy
        precision: Computation precision
        prevent_cse: Prevent common subexpression elimination
        sliding_window: Window size for local attention
        logit_soft_cap: Soft capping value for logits
        attention_sink_size: Size of attention sink

            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = ring_attention(query, key, value)
        >>>
        >>>
        >>> out = ring_attention(
        ...     query, key, value,
        ...     causal_block_size=128,
        ...     query_chunk_size=256,
        ...     key_chunk_size=256
        ... )
        >>>
        >>>
        >>> out = ring_attention(
        ...     query, key, value,
        ...     sliding_window=1024,
        ...     pdrop=0.1,
        ...     dropout_rng=rng
        ... )
            >>>
        >>>
        >>> out = ring_attention(..., platform="triton")
    """
    return _ring_executor(
        RingAttention(),
        query=query,
        key=key,
        value=value,
        bias=bias,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        cache_idx=cache_idx,
        axis_name=axis_name,
        float32_logits=float32_logits,
        softmax_scale=softmax_scale,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        causal_block_size=causal_block_size,
        deterministic=deterministic,
        dropout_rng=dropout_rng,
        pdrop=pdrop,
        dtype=dtype,
        policy=policy,
        precision=precision,
        prevent_cse=prevent_cse,
        sliding_window=sliding_window,
        logit_soft_cap=logit_soft_cap,
        attention_sink_size=attention_sink_size,
        platform=platform,
        _cfg=cfg,
    )
