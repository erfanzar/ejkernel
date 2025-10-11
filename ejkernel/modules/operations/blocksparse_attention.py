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
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask
    from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask


class BlockSparseAttention(Kernel[KernelConfig, Array]):
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

    def get_impl(self, cfg: KernelConfig):
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
        q_blocksize: int | None = None,
        kv_blocksize: int | None = None,
        bwd_q_blocksize: int | None = None,
        bwd_kv_blocksize: int | None = None,
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
        cfg: KernelConfig,
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
            q_blocksize=q_blocksize,
            kv_blocksize=kv_blocksize,
            bwd_q_blocksize=bwd_q_blocksize,
            bwd_kv_blocksize=bwd_kv_blocksize,
            softmax_scale=softmax_scale,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            fused_backward=fused_backward,
            debug=debug,
        )

    def heuristic_cfg(self, inv: Invocation[KernelConfig, Array]) -> KernelConfig:
        """Provide default configuration based on invocation context.

        Selects optimal block sizes based on sequence length and head dimension.

        Args:
            inv: Invocation object with arguments and metadata

        Returns:
            Default configuration with block sizes
        """

        return KernelConfig(
            block_q=128,
            block_k=128,
            block_d=64,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[KernelConfig, Array]):
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

        block_configs = [(128, 128)]

        candidates = []
        for block_q, block_k in block_configs:
            candidates.append(
                KernelConfig(
                    block_q=block_q,
                    block_k=block_k,
                    block_d=None,
                    num_warps=None,
                    num_stages=None,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_executor = create_default_executor()


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
    q_blocksize: int | None = None,
    kv_blocksize: int | None = None,
    bwd_q_blocksize: int | None = None,
    bwd_kv_blocksize: int | None = None,
    softmax_scale: float | None = None,
    mask_builder: typing.Callable[[int, int, int, int, int], "Mask"] | typing.Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
    debug: bool = False,
    platform: typing.Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
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
        q_blocksize=q_blocksize,
        kv_blocksize=kv_blocksize,
        bwd_q_blocksize=bwd_q_blocksize,
        bwd_kv_blocksize=bwd_kv_blocksize,
        softmax_scale=softmax_scale,
        mask_builder=mask_builder,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
        causal=causal,
        fused_backward=fused_backward,
        debug=debug,
        platform=platform,
    )
