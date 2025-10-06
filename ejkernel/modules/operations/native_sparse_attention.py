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


"""Native sparse attention module with automatic optimization.

This module implements native sparse attention using explicit block indices to
define sparsity patterns. Unlike block-sparse attention which uses mask builders,
this implementation directly specifies which blocks to attend to via index arrays.

This approach is particularly efficient when:
    - The sparse pattern is known ahead of time
    - Block indices can be precomputed and reused
    - Fine-grained control over sparsity is needed

The sparse pattern is defined by block_indices and block_counts arrays, allowing
flexible sparse attention patterns like local windows, strided patterns, or
custom document-structure-aware sparsity.
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform


class NativeSparseAttention(Kernel[KernelConfig, Array]):
    """Native Sparse Attention with custom optimization logic.

    Implements sparse attention using explicit block index specification. This provides
    direct control over which blocks participate in attention computation, enabling
    efficient sparse patterns without runtime mask building.

    Features:
        - Direct block index specification for sparsity
        - Configurable block size and block counts
        - Support for variable-length sequences
        - Token-level sparse patterns via token_indices
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The sparsity is controlled by:
        - block_indices: Which blocks each query block attends to
        - block_counts: Number of key blocks per query block
        - token_indices: Fine-grained token-level sparsity (optional)
    """

    def __init__(self):
        """Initialize Native Sparse Attention module."""
        super().__init__(op_id="native_sparse_attention")

    def get_impl(self, cfg: KernelConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for native sparse attention

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("native_sparse_attention", cfg.platform)
        return kernel_registry.get("native_sparse_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_keys_blocks"] | None = None,
        block_counts: Int[Array, "batch num_kv_heads num_query_blocks"] | int = 16,
        block_size: int = 64,
        scale: float | None = None,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        token_indices: Int[Array, "total_tokens"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla"] | None = None,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Execute native sparse attention with explicit block indices.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            block_indices: Indices of key blocks to attend to for each query block
                [batch, num_kv_heads, num_query_blocks, num_keys_blocks]
            block_counts: Number of key blocks per query block (can be int or array)
            block_size: Size of each attention block (default: 64)
            scale: Optional scaling factor for attention scores
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            token_indices: Optional token-level indices for fine-grained sparsity
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Sparse attention output [batch, seq_len, num_heads, head_dim]

        Note:
            When block_indices is None, a default pattern may be used depending
            on the implementation. Providing explicit indices gives full control
            over the sparsity pattern.
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
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            scale=scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
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


_sparse_executor = create_default_executor()


def sparse_attention(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_keys_blocks"] | None = None,
    block_counts: Int[Array, "batch num_kv_heads num_query_blocks"] | int = 16,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """Execute native sparse attention with automatic optimization.

    Sparse attention computes attention only on specified blocks or patterns,
    reducing computational cost for long sequences.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        block_indices: Indices of blocks to attend to
        block_counts: Number of blocks per query block (default: 16)
        block_size: Size of each attention block (default: 64)
        scale: Scaling factor for attention
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        token_indices: Token indices for sparse patterns

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = native_sparse_attention(query, key, value)
        >>>
        >>>
        >>> out = native_sparse_attention(query, key, value, block_size=128, block_counts=32)
        >>>
        >>>
        >>> out = native_sparse_attention(query, key, value, block_indices=indices, block_size=64)
    """
    return _sparse_executor(
        NativeSparseAttention(),
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
        token_indices=token_indices,
    )
