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


"""Ragged Decode Attention module with automatic optimization.

This module implements ragged decode attention, an efficient attention mechanism
optimized for inference scenarios with variable-length sequences in the decode phase.
Unlike standard attention which requires padded sequences, ragged attention processes
sequences with different lengths efficiently by using sequence start/end markers.

Ragged decode attention is particularly valuable for:
    - Inference workloads with batched sequences of varying lengths
    - Decoder-only models during generation
    - Serving scenarios requiring efficient batching
    - Situations where padding overhead is significant

The key innovation is using sequence_start and sequence_end arrays to define
valid attention ranges per sequence, eliminating the need for padding while
maintaining efficient vectorized computation.

Key Features:
    - Efficient variable-length sequence handling without padding
    - Support for sliding window attention for long contexts
    - Optional logit soft capping for numerical stability
    - Attention sink support for improved long-context performance
    - Configurable block sizes for memory-compute tradeoffs

Mathematical Foundation:
    For each query position i in sequence s:
        output[i] = softmax(Q[i] @ K[start[s]:end[s]].T / scale) @ V[start[s]:end[s]]

    Where start[s] and end[s] define the valid KV range for sequence s.
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class RaggedDecodeAttention(Kernel[KernelConfig, Array]):
    """Ragged Decode Attention with custom optimization logic.

    Implements efficient attention for variable-length sequences during inference decode phase.
    Uses sequence start/end markers to define valid attention ranges without padding overhead.

    Features:
        - Zero-padding overhead for variable-length sequences
        - Sliding window attention for local context
        - Logit soft capping for numerical stability
        - Attention sink mechanism for long contexts
        - Multiple platform support (Triton/Pallas/CUDA/XLA)
        - Configurable block sizes for performance tuning

    This implementation is particularly efficient for:
        - Batch inference with varying prompt/generation lengths
        - Serving workloads requiring dynamic batching
        - Decoder-only models in generation mode
    """

    def __init__(self):
        """Initialize Ragged Decode Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="ragged_decode_attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged decode attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_decode_attention", cfg.platform)
        return kernel_registry.get("ragged_decode_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        sequence_start: Int[Array, "batch"],
        sequence_end: Int[Array, "batch"],
        softmax_scale: float | None = 1,
        block_size: int = 256,
        sliding_window: tuple[int, int] | None = None,
        logit_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute ragged decode attention with variable-length sequences.

        Computes attention for batched queries where each sequence has a different
        valid key-value range defined by sequence_start and sequence_end markers.

        Args:
            query: Query tensor [batch, num_heads, head_dim] (typically single decode step)
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim] (full context)
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim] (full context)
            sequence_start: Start indices for valid KV range per sequence [batch]
            sequence_end: End indices (exclusive) for valid KV range per sequence [batch]
            softmax_scale: Scaling factor for attention scores (default: 1.0)
            block_size: Block size for computation tiling (default: 256)
            sliding_window: Optional (left, right) window sizes for local attention
            logit_soft_cap: Optional soft cap to bound attention logits
            softmax_aux: Optional attention sink logits for improved long-context performance
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object specifying block sizes and platform

        Returns:
            Attention output [total_tokens, num_q_heads, head_dim]

        Note:
            The sequence_start and sequence_end arrays define which KV positions
            are valid for each query. This enables efficient batching of sequences
            with different lengths without padding overhead.

        Example:
            >>>
            >>> sequence_start = jnp.array([0, 50])
            >>> sequence_end = jnp.array([50, 150])
            >>> out = ragged_decode_attention(q, k, v, sequence_start, sequence_end)
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
            key=key,
            value=value,
            softmax_scale=softmax_scale,
            logit_soft_cap=logit_soft_cap,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            block_size=block_size,
            platform=platform,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration optimized for decode attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default KernelConfig with conservative block sizes suitable for
            typical decode scenarios (small query sizes, variable KV lengths)
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

        Creates multiple configurations optimized for different decode scenarios,
        from small batches with short contexts to larger batches with longer contexts.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Decode attention typically has small query dimensions (batch size),
            so candidates focus on optimizing KV block sizes.
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


_ragged_decode_attention_executor = create_default_executor()


def ragged_decode_attention(
    query: Float[Array, "batch num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = 1,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logit_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Execute ragged decode attention with automatic optimization.

    Efficiently computes attention for variable-length sequences during the decode phase,
    using start/end indices to define valid attention ranges without padding overhead.

    Args:
        query: Query tensor [batch, num_heads, head_dim] for current decode step
        key: Full key context [batch, seq_len, num_kv_heads, head_dim]
        value: Full value context [batch, seq_len, num_kv_heads, head_dim]
        sequence_start: Start index of valid KV range per sequence [batch]
        sequence_end: End index (exclusive) of valid KV range per sequence [batch]
        softmax_scale: Attention score scaling factor (default: 1.0)
        block_size: Block size for tiled computation (default: 256)
        sliding_window: Optional (left, right) window sizes for local attention
        logit_soft_cap: Optional soft cap for attention logits (improves stability)
        softmax_aux: Optional attention sink values for long-context handling
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim]

    Example:
        >>>
        >>> out = ragged_decode_attention(q, k, v, starts, ends)
        >>>
        >>>
        >>> out = ragged_decode_attention(
        ...     q, k, v, starts, ends,
        ...     sliding_window=(256, 256),
        ...     block_size=128
        ... )
        >>>
        >>>
        >>> out = ragged_decode_attention(
        ...     q, k, v, starts, ends,
        ...     logit_soft_cap=50.0,
        ...     softmax_scale=0.125
        ... )
        >>>
        >>>
        >>> out = ragged_decode_attention(..., platform="triton")

    Note:
        This function is optimized for decode scenarios where query size is small
        (typically batch_size) and KV length varies per sequence. For prefill phase
        with large queries, consider using standard flash_attention instead.
    """
    return _ragged_decode_attention_executor(
        RaggedDecodeAttention(),
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        logit_soft_cap=logit_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        block_size=block_size,
        platform=platform,
    )
