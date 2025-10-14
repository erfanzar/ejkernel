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

import os
import typing

from jaxtyping import Array, Float, Int

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
from .configs import NativeSparseAttentionConfig


class NativeSparseAttention(Kernel[NativeSparseAttentionConfig, Array]):
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

    def get_impl(self, cfg: NativeSparseAttentionConfig):
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
        query: Float[Array, "batch seq_len num_q_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
        g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
        block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
        block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
        softmax_scale: float | None = None,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: NativeSparseAttentionConfig,
    ) -> Float[Array, "batch seq_len num_heads head_dim"]:
        """Execute native sparse attention with explicit block indices.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            block_indices: Indices of key blocks to attend to for each query block
                [batch, num_kv_heads, num_query_blocks, num_keys_blocks]
            block_counts: Number of key blocks per query block (can be int or array)
            softmax_scale: Optional scaling factor for attention scores
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
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
            cfg = NativeSparseAttentionConfig(
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                block_d=cfg.block_d,
                block_size=cfg.block_size,
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
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=cfg.block_size,
            softmax_scale=softmax_scale,
            cu_seqlens=cu_seqlens,
            g_cmp=g_cmp,
            g_slc=g_slc,
        )

    def heuristic_cfg(self, inv: Invocation[NativeSparseAttentionConfig, Array]) -> NativeSparseAttentionConfig:
        """Provide default configuration with block sizes.

        Selects balanced block sizes that work well for typical sparse patterns.
        The default configuration uses uniform block sizes for simplicity.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with block_size=64 for balanced performance
        """
        return NativeSparseAttentionConfig(
            block_q=64,
            block_k=64,
            block_d=64,
            block_size=64,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates a basic set of candidates for platform-agnostic tuning.
        Sparse attention benefits from consistent block sizes across dimensions.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List with single configuration using block_size=64 as baseline
        """
        block_configs = [(64, 64, 64)]

        candidates = []
        for block_q, block_k, block_d in block_configs:
            candidates.append(
                NativeSparseAttentionConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=block_d,
                    block_size=64,
                    num_warps=4,
                    num_stages=1,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for autotuning.

        Creates configurations tailored for GPU execution with Triton backend.
        Tests various block sizes (32, 64, 128) and warp counts (4, 8) to find
        optimal configuration for the specific GPU architecture.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of GPU-specific configurations with varying block sizes and warps
        """
        configs = []

        for block_size in [32, 64, 128]:
            for num_warps in [4, 8]:
                configs.append(
                    NativeSparseAttentionConfig(
                        block_q=block_size,
                        block_k=block_size,
                        block_d=block_size,
                        block_size=block_size,
                        num_warps=num_warps,
                        num_stages=1,
                        platform="triton",
                        backend="gpu",
                    )
                )
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning.

        Creates configurations tailored for TPU execution with Pallas backend.
        TPUs prefer larger block sizes (64, 128) for better vectorization.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of TPU-specific configurations optimized for matrix units
        """
        configs = []
        for block_size in [64, 128]:
            configs.append(
                NativeSparseAttentionConfig(
                    block_q=block_size,
                    block_k=block_size,
                    block_d=block_size,
                    block_size=block_size,
                    num_warps=4,
                    num_stages=1,
                    platform="pallas",
                    backend="tpu",
                )
            )
        return configs

    def candidate_cfgs_xla(self, inv: Invocation[NativeSparseAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        Creates configurations for XLA compiler backend. XLA handles optimization
        internally, so we provide conservative block sizes that work well across
        different hardware targets.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of XLA-compatible configurations with standard block sizes
        """
        configs = []
        for block_size in [64, 128]:
            configs.append(
                NativeSparseAttentionConfig(
                    block_q=block_size,
                    block_k=block_size,
                    block_d=block_size,
                    block_size=block_size,
                    num_warps=4,
                    num_stages=1,
                    platform="xla",
                    backend="any",
                )
            )
        return configs

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu
    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_xla = candidate_cfgs_xla


_sparse_executor: Executor[NativeSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("nsa"),
    )
)


def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: NativeSparseAttentionConfig | None = None,
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
        softmax_scale: Scaling factor for attention
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = native_sparse_attention(query, key, value)
        >>>
        >>>
        >>> out = native_sparse_attention(query, key, value, block_counts=32)
        >>>
        >>>
        >>> out = native_sparse_attention(query, key, value, platform="triton")
    """
    return _sparse_executor(
        NativeSparseAttention(),
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        softmax_scale=softmax_scale,
        cu_seqlens=cu_seqlens,
        g_cmp=g_cmp,
        g_slc=g_slc,
        platform=platform,
        _cfg=cfg,
    )
