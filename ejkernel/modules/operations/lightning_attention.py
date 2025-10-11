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


"""Lightning Attention module with automatic optimization.

This module implements Lightning Attention, a layer-aware attention mechanism that
adapts computation based on the layer position in the network. It's particularly
efficient for deep transformers where different layers may benefit from different
attention strategies.

Lightning Attention uses layer-specific optimizations and can maintain state across
sequence processing for improved efficiency in recurrent-style computation.
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class LightningAttention(Kernel[KernelConfig, Array]):
    """Lightning Attention with custom optimization logic.

    Implements a layer-aware attention mechanism optimized for deep transformer
    architectures. The attention computation adapts based on the layer index,
    allowing for more efficient processing in multi-layer networks.

    Features:
        - Layer-specific optimization strategies
        - Support for stateful computation with initial states
        - Bidirectional and reverse sequence processing
        - Variable-length sequence handling
        - Automatic platform selection (Triton/Pallas/XLA/CUDA)

    This is particularly useful for:
        - Very deep transformers where layer position matters
        - Models with recurrent-style attention patterns
        - Scenarios requiring different attention behavior per layer
    """

    def __init__(self):
        """Initialize Lightning Attention module."""
        super().__init__(op_id="lightning_attn")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for lightning attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("lightning_attn", cfg.platform)
        return kernel_registry.get("lightning_attn", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        layer_idx: int,
        num_layers: int,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Execute lightning attention with layer-specific optimization.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            layer_idx: Index of current layer in the model (0-indexed)
            num_layers: Total number of layers in the model
            softmax_scale: Optional scaling factor for attention scores
            initial_state: Optional initial hidden state [batch, num_heads, head_dim, head_dim]
            reverse: If True, process sequence in reverse order
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]

        Note:
            The layer_idx and num_layers parameters enable layer-specific
            optimizations that can improve performance in deep networks.
        """

        if platform is not None:
            cfg = KernelConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d,
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
            layer_idx=layer_idx,
            num_layers=num_layers,
            softmax_scale=softmax_scale,
            initial_state=initial_state,
            reverse=reverse,
            cu_seqlens=cu_seqlens,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration with block sizes."""
        return KernelConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
        """Generate candidate configurations for autotuning."""
        block_configs = [
            (64, 64, 64, 4, 1),
            (128, 64, 64, 4, 2),
            (128, 128, 64, 8, 2),
        ]

        candidates = []
        for block_q, block_k, block_d, num_warps, num_stages in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_lightning_executor = create_default_executor()


def lightning_attention(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    layer_idx: int,
    num_layers: int,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """Execute lightning attention with automatic optimization.

    Lightning attention is an efficient attention mechanism that uses
    layer-specific optimizations for improved performance.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        layer_idx: Current layer index in the model
        num_layers: Total number of layers in the model
        softmax_scale: Scaling factor for attention
        initial_state: Initial state for recurrent computation
        reverse: Whether to process sequence in reverse
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = lightning_attention(query, key, value, layer_idx=5, num_layers=32)
        >>>
        >>>
        >>> out = lightning_attention(query, key, value, layer_idx=0, num_layers=24, softmax_scale=0.125)
        >>>
        >>>
        >>> out = lightning_attention(query, key, value, layer_idx=10, num_layers=32, cu_seqlens=cu_seqs)
        >>>
        >>>
        >>> out = lightning_attention(query, key, value, layer_idx=0, num_layers=24, platform="pallas")
    """
    return _lightning_executor(
        LightningAttention(),
        query=query,
        key=key,
        value=value,
        layer_idx=layer_idx,
        num_layers=num_layers,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        platform=platform,
    )
