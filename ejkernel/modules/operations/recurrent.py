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


"""Recurrent Attention module with automatic optimization.

This module implements recurrent-style attention mechanisms that maintain and update
hidden states across sequence positions. Unlike standard attention which computes
all positions independently, recurrent attention processes sequences sequentially
with stateful computation.

Features:
    - Stateful attention with initial_state support
    - Separate gating for queries (g), keys (gk), and values (gv)
    - Layer-wise gating control via g_gamma
    - Bidirectional processing support (forward and reverse)
    - Variable-length sequence handling

This is particularly useful for:
    - Linear-time attention mechanisms
    - Models requiring sequential dependency modeling
    - Architectures with explicit state propagation
    - Efficient inference with incremental state updates
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class RecurrentAttention(Kernel[KernelConfig, Array]):
    """Recurrent Attention with custom optimization logic.

    Implements attention with recurrent state updates, enabling linear-time complexity
    for certain attention patterns. The mechanism maintains a hidden state that is
    updated at each sequence position.

    Features:
        - Stateful computation with hidden state propagation
        - Multiple gating mechanisms (g, gk, gv, g_gamma)
        - Forward and reverse processing modes
        - Support for initial states
        - Variable-length sequence handling via cu_seqlens
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The gating mechanisms provide fine-grained control:
        - g: Query-level gates
        - gk: Key-level gates
        - gv: Value-level gates
        - g_gamma: Layer-level gates
    """

    def __init__(self):
        """Initialize Recurrent Attention module."""
        super().__init__(op_id="recurrent")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for recurrent attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("recurrent", cfg.platform)
        return kernel_registry.get("recurrent", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        g_gamma: Float[Array, "batch num_heads"] | None = None,
        gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
        softmax_scale: float | None = None,
        initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
        reverse: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Execute recurrent attention with stateful computation.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            g: Query-level gating tensor [batch, seq_len, num_heads, head_dim]
            g_gamma: Layer-level gating parameter [batch, num_heads]
            gk: Key-level gating tensor [batch, seq_len, num_heads, head_dim]
            gv: Value-level gating tensor [batch, seq_len, num_heads, head_dim]
            softmax_scale: Optional scaling factor for attention scores
            initial_state: Initial hidden state [batch, num_heads, head_dim, head_dim]
            reverse: If True, process sequence in reverse order
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len, num_heads, head_dim]

        Note:
            All gating parameters (g, gk, gv, g_gamma) are optional. When provided,
            they enable more sophisticated gated recurrent mechanisms.
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
            g=g,
            g_gamma=g_gamma,
            gk=gk,
            gv=gv,
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


_recurrent_executor = create_default_executor()


def recurrent_attention(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "batch num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla"] | None = None,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """Execute recurrent attention with automatic optimization.

    Recurrent attention processes sequences with stateful computation,
    maintaining hidden states across timesteps for efficient sequential processing.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        g: Gating tensor for query [batch, seq_len, num_heads, head_dim]
        g_gamma: Gating gamma [batch, num_heads]
        gk: Gating tensor for keys [batch, seq_len, num_heads, head_dim]
        gv: Gating tensor for values [batch, seq_len, num_heads, head_dim]
        softmax_scale: Scaling factor for attention
        initial_state: Initial hidden state
        reverse: Whether to process sequence in reverse
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = recurrent_attention(query, key, value)
        >>>
        >>>
        >>> out = recurrent_attention(query, key, value, g=gates, gk=key_gates, gv=value_gates)
        >>>
        >>>
        >>> out = recurrent_attention(query, key, value, platform="xla")
    """
    return _recurrent_executor(
        RecurrentAttention(),
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
        platform=platform,
    )
