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


"""Block-sparse attention module with automatic optimization.

This module implements block-sparse attention, which applies attention only to
predefined blocks of the attention matrix, significantly reducing computational
cost for long sequences while maintaining important attention patterns.

The block-sparse pattern is defined by a mask builder function that determines
which blocks should be computed. This is particularly useful for document-level
attention, local attention patterns, and sparse attention architectures.
"""

from __future__ import annotations

import typing

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

from ..base import detect_platform
from .configs import BlockSparseAttentionConfig

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


class BlockSparseAttention(Kernel[BlockSparseAttentionConfig, Array]):
    """Block-sparse attention kernel with custom optimization logic.

    Implements attention computation over sparse block patterns, computing attention
    only for specified blocks rather than the full attention matrix. This reduces
    computational complexity from O(N^2) to O(N * B) where B is the average number
    of blocks per row.

    Features:
        - Configurable sparse block patterns via mask builder
        - Support for causal masking and sliding windows
        - Automatic platform/backend selection
        - Optional autotuning for optimal block sizes
        - Gradient support for training with custom VJP
        - Logit soft capping with tanh activation for numerical stability (Gemma-2 style)
        - Separate forward/backward block sizes for performance tuning

    The mask builder function defines which blocks to compute, enabling patterns like:
        - Local attention (nearby tokens only)
        - Global + local (attending to special tokens + local context)
        - Strided patterns (every nth block)
        - Custom patterns based on document structure

    Example:
        >>> from ejkernel.modules.operations import BlockSparseAttention
        >>> from ejkernel.modules import create_default_executor
        >>>
        >>> executor = create_default_executor()
        >>> attn = BlockSparseAttention()
        >>>
        >>>
        >>> def local_mask(q_idx, k_idx, q_size, k_size, window):
        ...
        ...     pass
        >>>
        >>> output = executor(
        ...     attn,
        ...     query, key, value,
        ...     mask_builder=local_mask,
        ...     chunk_size=128
        ... )
    """

    def __init__(self):
        """Initialize BlockSparseAttention module."""
        super().__init__(op_id="blocksparse_attention")

    def get_impl(self, cfg: BlockSparseAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for block-sparse attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        return kernel_registry.get(
            algorithm="blocksparse_attention",
            platform=detect_platform("blocksparse_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        q_positions: Int[Array, "batch seq_len"] | None = None,
        kv_positions: Int[Array, "batch kv_len"] | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        bias: Float[Array, "batch num_heads seq_len head_dim"] | None = None,
        attention_mask: Bool[Array, "batch num_heads seq_len head_dim"]
        | Bool[Array, "batch 1 seq_len head_dim"]
        | Int[Array, "batch num_heads seq_len head_dim"]
        | Int[Array, "batch 1 seq_len head_dim"]
        | None = None,
        sequence_parallelism_mesh_axis_name: str | None = None,
        logit_soft_cap: float | None = None,
        qkv_layouts: tuple["SparseMask"] | None = None,
        softmax_scale: float | None = None,
        mask_builder: typing.Callable[[int, int, int, int, int], "Mask"]
        | typing.Callable[[], "SparseMask"]
        | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        debug: bool = False,
        platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: BlockSparseAttentionConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        """Execute block-sparse attention with the given configuration.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, kv_num_heads, kv_len, head_dim]
            value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim]
            q_segment_ids: Segment IDs for queries to handle multiple sequences [batch, seq_len]
            kv_segment_ids: Segment IDs for keys/values [batch, kv_len]
            softmax_aux: Auxiliary values added to attention scores (e.g., for attention sinks)
            logit_soft_cap: Optional soft cap value to bound attention logits
            softmax_scale: Scaling factor for attention scores (default: 1/sqrt(head_dim))
            mask_builder: Function that builds the sparse mask pattern. Takes (q_idx, k_idx,
                q_size, k_size, window_size) and returns a Mask object
            sliding_window: Window size for local attention, int for symmetric or (left, right) tuple
            chunk_size: Overall chunk size (alternative to separate query/key chunk sizes)
            causal: Whether to apply causal masking (default: True)
            fused_backward: Use fused backward pass for improved gradient computation
            platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
            cfg: Configuration object specifying platform/backend and kernel parameters

        Returns:
            Attention output tensor [batch, seq_len_q, num_heads, head_dim]

        Note:
            The mask_builder function is critical for defining sparsity patterns.
            It should return a mask indicating which blocks to compute.
        """
        if platform is not None:
            cfg = BlockSparseAttentionConfig(
                q_blocksize=cfg.q_blocksize,
                kv_blocksize=cfg.kv_blocksize,
                bwd_q_blocksize=cfg.bwd_q_blocksize,
                bwd_kv_blocksize=cfg.bwd_kv_blocksize,
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
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            softmax_aux=softmax_aux,
            logit_soft_cap=logit_soft_cap,
            bias=bias,
            attention_mask=attention_mask,
            sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
            qkv_layouts=qkv_layouts,
            q_blocksize=cfg.q_blocksize,
            kv_blocksize=cfg.kv_blocksize,
            bwd_q_blocksize=cfg.bwd_q_blocksize,
            bwd_kv_blocksize=cfg.bwd_kv_blocksize,
            softmax_scale=softmax_scale,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            fused_backward=fused_backward,
            debug=debug,
        )

    def heuristic_cfg(self, inv: Invocation[BlockSparseAttentionConfig, Array]) -> BlockSparseAttentionConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return BlockSparseAttentionConfig(
            q_blocksize=128,
            kv_blocksize=128,
            bwd_q_blocksize=128,
            bwd_kv_blocksize=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
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

        block_configs = [(256, 256), (512, 512)]

        candidates = []
        for q_block, kv_block in block_configs:
            candidates.append(
                BlockSparseAttentionConfig(
                    q_blocksize=q_block,
                    kv_blocksize=kv_block,
                    bwd_q_blocksize=q_block * 2,
                    bwd_kv_blocksize=kv_block * 2,
                    num_warps=4,
                    num_stages=2,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def candidate_cfgs_gpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate GPU-optimized candidate configurations for autotuning.

        GPU/Triton kernels benefit from medium blocks with more warps and stages.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of GPU-optimized candidate configurations
        """
        configs = []
        for q_block in [256, 512]:
            for kv_block in [256, 512]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3]:
                        configs.append(
                            BlockSparseAttentionConfig(
                                q_blocksize=q_block,
                                kv_blocksize=kv_block,
                                bwd_q_blocksize=q_block // 2,
                                bwd_kv_blocksize=kv_block // 2,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                platform="triton",
                                backend="gpu",
                            )
                        )
        return configs

    def candidate_cfgs_tpu(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate TPU-optimized candidate configurations for autotuning.

        TPU/Pallas kernels benefit from larger blocks with fewer stages.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of TPU-optimized candidate configurations
        """
        configs = []
        for q_block in [512, 1024]:
            for kv_block in [512, 1024]:
                configs.append(
                    BlockSparseAttentionConfig(
                        q_blocksize=q_block,
                        kv_blocksize=kv_block,
                        bwd_q_blocksize=q_block // 2,
                        bwd_kv_blocksize=kv_block // 2,
                        num_warps=4,
                        num_stages=1,
                        platform="pallas",
                        backend="tpu",
                    )
                )
        return configs

    def candidate_cfgs_xla(self, inv: Invocation[BlockSparseAttentionConfig, Array]):
        """Generate XLA-optimized candidate configurations for autotuning.

        XLA implementations use medium blocks with moderate parallelism.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Iterable of XLA-optimized candidate configurations
        """
        configs = []
        for q_block in [256, 512]:
            for kv_block in [256, 512]:
                configs.append(
                    BlockSparseAttentionConfig(
                        q_blocksize=q_block,
                        kv_blocksize=kv_block,
                        bwd_q_blocksize=q_block * 2,
                        bwd_kv_blocksize=kv_block * 2,
                        num_warps=4,
                        num_stages=2,
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
        q_segment_ids=None,
        kv_segment_ids=None,
        q_positions=None,
        kv_positions=None,
        softmax_aux=None,
        bias=None,
        attention_mask=None,
        sequence_parallelism_mesh_axis_name=None,
        logit_soft_cap=None,
        qkv_layouts=None,
        softmax_scale=None,
        mask_builder=None,
        sliding_window=None,
        chunk_size=None,
        causal=True,
        fused_backward=False,
        debug=False,
        platform=None,
        *,
        cfg: BlockSparseAttentionConfig,
    ):
        """GPU-specific forward pass with residuals using Triton kernels.

        Args:
            query: Query tensor [batch, num_heads, seq_len, head_dim]
            key: Key tensor [batch, kv_num_heads, kv_len, head_dim]
            value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim]
            (remaining args same as run method)
            cfg: Configuration object specifying kernel parameters

        Returns:
            Tuple of (output, residuals) where residuals are needed for backward pass
        """
        import jax.numpy as jnp

        from ejkernel.kernels._triton.blocksparse_attention import _blocksparse_attention_bhtd_fwd

        qlen = query.shape[2]
        kvlen = key.shape[2]

        if mask_builder is not None and qkv_layouts is None:
            qkv_layouts = mask_builder()

        if sliding_window is None:
            window_left = window_right = -1
        else:
            window_left, window_right = sliding_window

        if softmax_scale is None:
            softmax_scale = query.shape[-1] ** -0.5

        if q_positions is None:
            q_positions = jnp.arange(0, qlen).reshape(1, -1).repeat(query.shape[0], 0)
        if kv_positions is None:
            kv_positions = jnp.arange(0, kvlen).reshape(1, -1).repeat(key.shape[0], 0)

        if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
            from ejkernel.xla_utils import mask_to_segment_ids

            inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
            if q_segment_ids is None:
                q_segment_ids = inferred_q_seg
            if kv_segment_ids is None:
                kv_segment_ids = inferred_kv_seg

        if q_segment_ids is None:
            q_segment_ids = jnp.ones_like(q_positions)
        if kv_segment_ids is None:
            kv_segment_ids = jnp.ones_like(kv_positions)

        if qkv_layouts is None:
            from ejkernel.kernels._triton.blocksparse_attention._mask import create_sparsity_mask

            qkv_layouts = create_sparsity_mask(
                q_blocksize=cfg.q_blocksize,
                kv_blocksize=cfg.kv_blocksize,
                kv_positions=kv_positions,
                kv_segment_ids=kv_segment_ids,
                q_positions=q_positions,
                q_segment_ids=q_segment_ids,
                causal=causal,
                window_left=window_left,
                window_right=window_right,
            )
        output, residuals = _blocksparse_attention_bhtd_fwd(
            query=query,
            key=key,
            value=value,
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            qkv_layouts=qkv_layouts,
            softmax_scale=softmax_scale,
            softmax_aux=softmax_aux,
            bias=bias,
            apply_load_balance=True,
            sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
            window_left=window_left,
            window_right=window_right,
            causal=causal,
            q_blocksize=cfg.q_blocksize,
            kv_blocksize=cfg.kv_blocksize,
            bwd_kv_blocksize=cfg.bwd_kv_blocksize,
            bwd_q_blocksize=cfg.bwd_q_blocksize,
            logit_soft_cap=logit_soft_cap,
            debug=debug if isinstance(debug, bool) else False,
        )

        return output, residuals

    def vjp_gpu(
        self,
        residuals,
        output,
        dO,
        *args,
        q_segment_ids=None,
        kv_segment_ids=None,
        q_positions=None,
        kv_positions=None,
        softmax_aux=None,
        bias=None,
        attention_mask=None,
        sequence_parallelism_mesh_axis_name=None,
        logit_soft_cap=None,
        softmax_scale=None,
        mask_builder=None,
        sliding_window=None,
        chunk_size=None,
        causal=True,
        fused_backward=False,
        debug=False,
        platform=None,
        cfg=None,
        **kwargs,
    ):
        """GPU-specific backward pass using Triton kernels.

        Args:
            residuals: Residuals saved from forward pass
            output: Output from forward pass
            dO: Gradient of loss with respect to output
            (remaining args same as run method)

        Returns:
            Tuple of gradients (dq, dk, dv, ...) for all differentiable inputs
        """
        from ejkernel.kernels._triton.blocksparse_attention import _blocksparse_attention_gpu_bwd

        if sliding_window is None:
            window_left = window_right = -1
        else:
            window_left, window_right = sliding_window

        return _blocksparse_attention_gpu_bwd(
            softmax_scale=softmax_scale,
            apply_load_balance=True,
            sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
            window_left=window_left,
            window_right=window_right,
            causal=causal,
            q_blocksize=cfg.q_blocksize,
            kv_blocksize=cfg.kv_blocksize,
            bwd_kv_blocksize=cfg.bwd_kv_blocksize,
            bwd_q_blocksize=cfg.bwd_q_blocksize,
            logit_soft_cap=logit_soft_cap,
            debug=debug if isinstance(debug, bool) else False,
            res=residuals,
            dout=dO,
        )


_executor: Executor[BlockSparseAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(allow_autotune=True),
        tuner=Tuner(warmup=2, iters=5),
    )
)


def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len head_dim"] | None = None,
    attention_mask: Bool[Array, "batch num_heads seq_len head_dim"]
    | Bool[Array, "batch 1 seq_len head_dim"]
    | Int[Array, "batch num_heads seq_len head_dim"]
    | Int[Array, "batch 1 seq_len head_dim"]
    | None = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logit_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    mask_builder: typing.Callable[[int, int, int, int, int], "Mask"] | typing.Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
    debug: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    *,
    cfg: BlockSparseAttentionConfig | None = None,
) -> Float[Array, "batch kv_num_heads kv_len vhead_dim"]:
    """Execute block-sparse attention with automatic optimization.

    Performs efficient attention computation over sparse block patterns, significantly
    reducing memory and computation compared to dense attention while maintaining
    flexibility through custom mask builders.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim]
        key: Key tensor [batch, kv_num_heads, kv_len, head_dim]
        value: Value tensor [batch, kv_num_heads, kv_len, vhead_dim]
        q_segment_ids: Optional segment IDs for queries [batch, seq_len]
        kv_segment_ids: Optional segment IDs for keys/values [batch, kv_len]
        softmax_aux: Optional auxiliary attention values (e.g., attention sinks)
        logit_soft_cap: Optional soft capping for attention logits
        query_chunk_size: Query chunk size for block tiling (default: 128)
        key_chunk_size: Key chunk size for block tiling (default: 128)
        softmax_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
        mask_builder: Callable defining sparse pattern. Signature:
            (q_idx, k_idx, q_size, k_size, window) -> Mask
        sliding_window: Window size for local attention (int or (left, right) tuple)
        chunk_size: Alternative to separate query_chunk_size/key_chunk_size
        causal: Apply causal masking (default: True)
        fused_backward: Use fused backward pass (default: False)
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")

    Returns:
        Attention output [batch, kv_num_heads, kv_len, vhead_dim]

    Example:
        >>> from ejkernel.modules.operations import blocksparse_attention
        >>>
        >>>
        >>> output = blocksparse_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> def local_plus_global(q_idx, k_idx, q_size, k_size, window):
        ...
        ...     return create_local_global_mask(q_idx, k_idx, window)
        >>>
        >>> output = blocksparse_attention(
        ...     query, key, value,
        ...     mask_builder=local_plus_global,
        ...     sliding_window=256
        ... )
        >>>
        >>>
        >>> output = blocksparse_attention(
        ...     query, key, value,
        ...     platform="triton"
        ... )

    Note:
        Block-sparse attention is particularly effective for:
        - Long document processing where full attention is prohibitive
        - Architectures with specific attention patterns (e.g., Longformer)
        - Scenarios where custom sparsity patterns are needed
    """

    return _executor(
        BlockSparseAttention(),
        query=query,
        key=key,
        value=value,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        q_positions=q_positions,
        kv_positions=kv_positions,
        softmax_aux=softmax_aux,
        logit_soft_cap=logit_soft_cap,
        bias=bias,
        attention_mask=attention_mask,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        mask_builder=mask_builder,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
        causal=causal,
        fused_backward=fused_backward,
        debug=debug,
        platform=platform,
        _cfg=cfg,
    )
