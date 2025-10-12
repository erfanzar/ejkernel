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
        """Provide default configuration with block sizes."""
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
        """Generate candidate configurations for autotuning."""
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
        """Generate GPU-optimized candidate configurations for autotuning."""
        configs = []
        for block_size in [64, 128]:
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
        """Generate TPU-optimized candidate configurations for autotuning."""
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
        """Generate XLA-optimized candidate configurations for autotuning."""
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

    def fwd_with_residuals_gpu(
        self,
        query,
        key,
        value,
        g_cmp=None,
        g_slc=None,
        block_indices=None,
        block_counts=16,
        softmax_scale=None,
        cu_seqlens=None,
        platform=None,
        *,
        cfg: NativeSparseAttentionConfig,
    ):
        """GPU-specific forward pass with residuals using Triton kernels.

        This implementation uses the low-level apply_native_sparse_attention
        with custom VJP, handling the compression and selection logic separately.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            g_cmp: Optional compression gate
            g_slc: Optional selection gate
            block_indices: Explicit block indices or None to compute via compression
            block_counts: Number of blocks per query (default: 16)
            softmax_scale: Attention scaling factor
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            platform: Platform override
            cfg: Configuration object

        Returns:
            Tuple of (output, residuals) for backward pass
        """
        import warnings

        import jax.numpy as jnp

        from ejkernel.kernels._triton.mean_pooling import mean_pooling
        from ejkernel.kernels._triton.native_sparse_attention import native_sparse_attention_gpu_fwd
        from ejkernel.kernels._triton.native_sparse_attention._compression import nsa_compression
        from ejkernel.kernels._triton.native_sparse_attention._triton_impl_fwd import nsa_topk
        from ejkernel.xla_utils import prepare_token_indices

        block_size = cfg.block_size

        if softmax_scale is None:
            softmax_scale = key.shape[-1] ** -0.5

        if cu_seqlens is not None:
            assert query.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"

        group_size = query.shape[2] // key.shape[2]
        assert group_size % 16 == 0, f"Group size must be a multiple of 16 in NSA, got {group_size}"

        token_indices = None
        if cu_seqlens is not None:
            token_indices = prepare_token_indices(cu_seqlens)

        k_cmp = mean_pooling(key, block_size, cu_seqlens)
        v_cmp = mean_pooling(value, block_size, cu_seqlens)
        o_cmp = None

        if g_cmp is not None:
            o_cmp, lse_cmp = nsa_compression(
                query=query,
                key=k_cmp,
                value=v_cmp,
                block_size=block_size,
                softmax_scale=softmax_scale,
                cu_seqlens=cu_seqlens,
            )
            if block_indices is not None:
                warnings.warn("`block_indices` will be ignored when `g_cmp` is provided", stacklevel=1)

            block_indices = nsa_topk(
                q=query,
                k=k_cmp,
                lse=lse_cmp,
                block_counts=block_counts,
                block_size=block_size,
                softmax_scale=softmax_scale,
                cu_seqlens=cu_seqlens,
            )

        assert block_indices is not None, "if `g_cmp` is not passed, `block_indices` must be provided."

        o_slc, residual = native_sparse_attention_gpu_fwd(
            query=query,
            key=key,
            value=value,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
            softmax_scale=softmax_scale,
            cu_seqlens=cu_seqlens,
            token_indices=token_indices,
        )

        o = o_slc
        if g_slc is not None:
            o = o_slc * jnp.expand_dims(g_slc, -1)

        if o_cmp is not None and g_cmp is not None:
            o = o + o_cmp * jnp.expand_dims(g_cmp, -1)

        full_residual = (residual, block_indices, block_counts, block_size, softmax_scale, cu_seqlens, token_indices)
        return o, full_residual

    def vjp_gpu(
        self,
        residuals,
        output,
        dO,
        *args,
        g_cmp=None,
        g_slc=None,
        block_indices=None,
        block_counts=16,
        softmax_scale=None,
        cu_seqlens=None,
        platform=None,
        **kwargs,
    ):
        """GPU-specific backward pass using Triton kernels.

        Args:
            residuals: Residuals from forward pass
            output: Output from forward pass
            dO: Gradient of loss with respect to output
            (remaining args same as fwd_with_residuals_gpu)

        Returns:
            Tuple of gradients (dq, dk, dv, ...)
        """
        from ejkernel.kernels._triton.native_sparse_attention import (
            native_sparse_attention_gpu_bwd,
        )

        (
            inner_residual,
            block_indices_saved,
            block_counts_saved,
            block_size_saved,
            softmax_scale_saved,
            cu_seqlens_saved,
            token_indices_saved,
        ) = residuals

        dq, dk, dv = native_sparse_attention_gpu_bwd(
            block_indices=block_indices_saved,
            block_counts=block_counts_saved,
            block_size=block_size_saved,
            softmax_scale=softmax_scale_saved,
            cu_seqlens=cu_seqlens_saved,
            token_indices=token_indices_saved,
            residual=inner_residual,
            do=dO,
        )

        return dq, dk, dv


_sparse_executor: Executor[NativeSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True),
        tuner=Tuner(warmup=2, iters=5),
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
    softmax_scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    *,
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
