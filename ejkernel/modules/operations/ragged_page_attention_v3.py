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


"""Ragged Page Attention module with automatic optimization.

This module implements ragged page attention, combining the benefits of both
ragged (variable-length) sequence processing and paged KV cache management.
This approach is particularly efficient for serving scenarios where sequences
have variable lengths and KV cache is organized in fixed-size pages.

Ragged page attention addresses key challenges in LLM inference:
    - Variable-length sequences without padding overhead
    - Efficient memory management through paged KV cache
    - Dynamic batching with different sequence lengths
    - Memory sharing for beam search and prefix caching

Key Concepts:
    Ragged Layout: Sequences are concatenated without padding, with start
        locations tracking where each sequence begins
    Pages: Fixed-size blocks holding portions of KV cache
    Block Tables: Mapping from logical sequence positions to physical pages

The combination provides:
    - Zero padding overhead (ragged layout)
    - Flexible memory allocation (paged cache)
    - Efficient batching of variable-length sequences
    - Support for dynamic sequence management

Memory Layout:
    Queries: [total_tokens, num_heads, head_dim] (ragged, no padding)
    KV Cache: [num_pages, page_size, num_heads, head_dim] (paged)

Mathematical Foundation:
    For token i in sequence s:
        start_idx = query_start_loc[s]
        end_idx = query_start_loc[s + 1]
        output[i] = attention(Q[start_idx:end_idx], K[pages[s]], V[pages[s]])

This is the most memory-efficient attention variant for serving workloads.
"""

from __future__ import annotations

import os
from typing import Literal

from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int32

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
from .configs import RaggedPageAttentionv3Config


class RaggedPageAttentionv3(Kernel[RaggedPageAttentionv3Config, tuple[Array, Array]]):
    """Ragged Page Attention with custom optimization logic.

    Combines ragged (variable-length) sequence processing with paged KV cache
    management for maximum memory efficiency in serving workloads.

    Features:
        - Zero padding overhead through ragged layout
        - Efficient paged KV cache management
        - Support for variable context lengths per sequence
        - Sliding window attention for long contexts
        - Logit soft capping for numerical stability
        - Attention sink mechanism for improved long-context performance
        - Configurable block sizes and memory limits
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    This implementation is particularly efficient for:
        - LLM serving with dynamic batching
        - Variable-length inference workloads
        - Memory-constrained deployment
        - Scenarios requiring efficient KV cache sharing

    The ragged layout eliminates padding overhead while paged cache
    enables flexible memory management and sharing.
    """

    def __init__(self):
        """Initialize Ragged Page Attention module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="ragged_page_attention")

    def create_shard_map_wrapper(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_scale: float = 1.0,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RaggedPageAttentionv3Config | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed ragged page attention.

        This method creates a sharded computation wrapper that distributes ragged page
        attention across multiple devices using JAX's shard_map. It enables efficient
        parallel execution while handling variable-length sequences and paged KV cache.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
                All sequences concatenated without padding.
            keys: Key tensor [total_tokens, num_kv_heads, head_dim] for cache updates.
            values: Value tensor [total_tokens, num_kv_heads, head_dim] for cache updates.
            kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
                Contains interleaved keys and values in paged format.
            kv_lens: Context lengths for each sequence [max_num_seqs].
                Indicates how many tokens are valid in each sequence's KV cache.
            block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
                Maps logical pages to physical page indices in kv_cache.
            query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
                query_start_loc[i] to query_start_loc[i+1] defines sequence i's token range.
            distribution: Distribution parameters [3] containing:
                [num_seqs, pages_per_seq, page_size] for kernel execution.
            softmax_scale: Scaling factor applied to attention scores before softmax.
                Default is 1.0, typically set to 1/sqrt(head_dim).
            sliding_window: Optional window size for sliding window attention.
                If None, uses full attention. Otherwise limits attention to sliding_window tokens.
            logits_soft_cap: Optional soft capping value for attention logits.
                Helps prevent extreme values and improves numerical stability.
            q_scale: Optional scaling factor for queries in quantized attention.
            k_scale: Optional scaling factor for keys in quantized attention.
            v_scale: Optional scaling factor for values in quantized attention.
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
                Controls memory usage on TPU accelerators.
            platform: Target platform ("triton", "pallas", "cuda", "xla", "auto").
                If None, uses platform from cfg or auto-detection.
            cfg: Kernel configuration containing block sizes and tuning parameters.
                If None, uses default configuration.
            mesh: JAX device mesh defining the device topology for sharding.
                Must be provided for shard_map execution.
            in_specs: Tuple of PartitionSpec objects defining input tensor sharding.
                Must match the number of input arguments (queries, keys, values, kv_cache,
                kv_lens, block_tables, query_start_loc, distribution).
            out_specs: PartitionSpec defining output tensor sharding.
                Must be provided for shard_map execution.
            check_vma: Whether to check virtual memory alignment in shard_map.
                Default is False.

        Returns:
            Tuple containing:
                - shard_map_fn: The shard_mapped attention function ready for execution
                - call_args: Tuple of arguments to pass to shard_map_fn

        Raises:
            AssertionError: If mesh, in_specs, or out_specs is None.
            AssertionError: If length of in_specs doesn't match number of call arguments.

        Note:
            The shard_map wrapper enables efficient parallel execution of ragged page
            attention across multiple devices while maintaining the ragged layout and
            paged cache structure. This is essential for scaling to large batch sizes
            and long contexts in distributed serving scenarios.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_ragged_page_attn(
            queries: Float[Array, "total_tokens num_q_heads head_dim"],
            keys: Float[Array, "total_tokens num_kv_heads head_dim"],
            values: Float[Array, "total_tokens num_kv_heads head_dim"],
            kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
            kv_lens: Int32[Array, "max_num_seqs"],
            block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
            query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
            distribution: Int32[Array, "3"],
        ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
            return self.run(
                queries=queries,
                keys=keys,
                values=values,
                kv_cache=kv_cache,
                kv_lens=kv_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                distribution=distribution,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                vmem_limit_bytes=vmem_limit_bytes,
                cfg=cfg,
            )

        call_args = (
            queries,
            keys,
            values,
            kv_cache,
            kv_lens,
            block_tables,
            query_start_loc,
            distribution,
        )
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_ragged_page_attn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, call_args

    def get_impl(self, cfg: RaggedPageAttentionv3Config):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged page attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_page_attention", cfg.platform)
        return kernel_registry.get("ragged_page_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        queries: Float[Array, "total_tokens num_q_heads head_dim"],
        keys: Float[Array, "total_tokens num_kv_heads head_dim"],
        values: Float[Array, "total_tokens num_kv_heads head_dim"],
        kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
        kv_lens: Int32[Array, "max_num_seqs"],
        block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
        query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
        distribution: Int32[Array, "3"],
        softmax_scale: float = 1.0,
        sliding_window: int | None = None,
        logits_soft_cap: float | None = None,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
        vmem_limit_bytes: int | None = None,
        *,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: RaggedPageAttentionv3Config,
    ) -> Float[Array, "total_tokens num_q_heads head_dim"]:
        """Execute ragged page attention over variable-length sequences.

        Computes attention where queries are in ragged (concatenated) format
        and KV cache is organized in pages, providing maximum memory efficiency
        for serving workloads with variable-length sequences.

        Args:
            queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
                All sequences concatenated without padding. The start position of
                each sequence is defined by query_start_loc.
            keys: Key tensor [total_tokens, num_kv_heads, head_dim] used for
                updating the KV cache with new tokens.
            values: Value tensor [total_tokens, num_kv_heads, head_dim] used for
                updating the KV cache with new tokens.
            kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
                Contains interleaved keys and values in paged format. Physical pages
                are mapped via block_tables.
            kv_lens: Context lengths for each sequence [max_num_seqs].
                Specifies how many tokens are valid in each sequence's KV cache.
            block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
                Maps logical page indices to physical page indices in kv_cache.
                Flattened from shape [max_num_seqs, pages_per_seq].
            query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
                query_start_loc[i] to query_start_loc[i+1] defines the token range
                for sequence i in the ragged queries tensor.
            distribution: Distribution parameters [3].
            softmax_scale: Scaling factor applied to attention scores before softmax.
                Default is 1.0, typically set to 1/sqrt(head_dim) for stability.
            sliding_window: Optional window size for sliding window attention.
                If None, uses full attention over all context. If set, limits
                attention to the last sliding_window tokens.
            logits_soft_cap: Optional soft capping value for attention logits.
                Applies tanh-based soft capping to prevent extreme values and
                improve numerical stability.
            q_scale: Optional scaling factor for queries in quantized attention.
                Used when queries are quantized to lower precision.
            k_scale: Optional scaling factor for keys in quantized attention.
                Used when keys are quantized to lower precision.
            v_scale: Optional scaling factor for values in quantized attention.
                Used when values are quantized to lower precision.
            vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
                Controls VMEM usage on TPU accelerators for large head dimensions.
            platform: Optional platform override ("triton", "pallas", "cuda", "xla", "auto").
                If provided, overrides the platform specified in cfg.
            cfg: Kernel configuration object containing:
                - num_kv_pages_per_block: Number of KV pages processed per block
                - num_queries_per_block: Number of query tokens processed per block
                - num_warps: Number of warps for Triton kernels
                - num_stages: Number of pipeline stages for Triton kernels
                - platform: Target platform
                - backend: Backend specification

        Returns:
            Attention output [total_tokens, num_q_heads, head_dim] in ragged format.
            The output maintains the same ragged layout as the input queries.

        Note:
            The ragged format eliminates all padding overhead by concatenating sequences
            without padding. Combined with paged KV cache, this provides the most
            memory-efficient attention implementation for serving workloads with
            variable-length sequences. The paged cache also enables memory sharing
            across sequences for features like beam search and prefix caching.

        Example:
            >>> import jax.numpy as jnp
            >>>
            >>> queries = jnp.ones((25, 32, 128))
            >>> keys = jnp.ones((25, 8, 128))
            >>> values = jnp.ones((25, 8, 128))
            >>> kv_cache = jnp.zeros((100, 16, 16, 2, 128))
            >>> kv_lens = jnp.array([10, 15])
            >>> block_tables = jnp.arange(200).reshape(-1)
            >>> query_start_loc = jnp.array([0, 10, 25])
            >>> distribution = jnp.array([2, 100, 16])
            >>>
            >>> kernel = RaggedPageAttentionv3()
            >>> cfg = RaggedPageAttentionv3Config(platform="auto")
            >>> out = kernel.run(
            ...     queries, keys, values, kv_cache, kv_lens,
            ...     block_tables, query_start_loc, distribution,
            ...     softmax_scale=0.0883883476483184,
            ...     cfg=cfg
            ... )
            >>> out.shape
        """

        if platform is not None:
            cfg = RaggedPageAttentionv3Config(
                num_kv_pages_per_block=cfg.num_kv_pages_per_block,
                num_queries_per_block=cfg.num_queries_per_block,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            queries=queries,
            keys=keys,
            values=values,
            kv_cache=kv_cache,
            kv_lens=kv_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            distribution=distribution,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            vmem_limit_bytes=vmem_limit_bytes,
            chunk_prefill_size=cfg.chunk_prefill_size,
            num_kv_pages_per_block=cfg.num_kv_pages_per_block,
            num_queries_per_block=cfg.num_queries_per_block,
        )

    def heuristic_cfg(self, inv: Invocation[RaggedPageAttentionv3Config, Array]) -> RaggedPageAttentionv3Config:
        """Provide default configuration optimized for ragged page attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration with conservative block sizes suitable for
            typical ragged attention workloads with variable sequence lengths
        """
        return RaggedPageAttentionv3Config(
            num_kv_pages_per_block=None,
            num_queries_per_block=None,
            num_warps=4,
            num_stages=1,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate candidate configurations for autotuning.

        Creates configurations optimized for ragged attention scenarios with
        various batch sizes and sequence lengths.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Ragged attention performance depends on the distribution of sequence
            lengths and the page size. Candidates are chosen to work well across
            common serving scenarios.
        """

        return [
            RaggedPageAttentionv3Config(
                num_kv_pages_per_block=None,
                num_queries_per_block=None,
                num_warps=None,
                num_stages=None,
                platform="auto",
                backend="any",
            )
        ]

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate candidate configurations for autotuning on GPU (Triton backend).

        Produces a set of kernel configurations optimized for GPU execution using
        the Triton compiler. The configurations explore different block sizes and
        parallelism settings to find the optimal performance for the given workload.

        Args:
            inv: Invocation object containing the input arguments and metadata.
                This can be inspected to create workload-specific configurations
                based on sequence lengths, head dimensions, batch size, etc.

        Returns:
            List of RaggedPageAttentionv3Config objects to evaluate during autotuning.
            Each configuration represents a different combination of kernel parameters
            that may perform well on GPU hardware.

        Note:
            Currently returns a single default configuration that delegates to XLA
            backend with automatic parameter selection. In future versions, this may
            return multiple configurations exploring different:
            - num_queries_per_block: Controls query batch size (e.g., 16, 32, 64, 128)
            - num_kv_pages_per_block: Controls KV page batch size (e.g., 1, 2, 4, 8)
            - num_warps: Parallelism level for Triton (e.g., 4, 8, 16)
            - num_stages: Pipeline depth for async memory operations (e.g., 1, 2, 3)

            These parameters significantly affect performance and should be tuned
            based on the specific GPU architecture and workload characteristics.
        """

        return [
            RaggedPageAttentionv3Config(
                num_kv_pages_per_block=None,
                num_queries_per_block=None,
                num_warps=None,
                num_stages=None,
                platform="xla",
                backend="any",
            )
        ]

    def candidate_cfgs_tpu(self, inv: Invocation[RaggedPageAttentionv3Config, Array]):
        """Generate candidate configurations for autotuning on TPU (Pallas backend).

        Heuristics:
        - For small head_dim, larger BLOCK_M is fine (64-128).
        - For large head_dim (>=160), prefer smaller BLOCK_M (32-64).
        - More KV pages per block helps small page_size (<=32).
        - Constrain S_block = page_size * num_kv_pages_per_block <= 256 to keep tiles reasonable.
        """

        return [
            RaggedPageAttentionv3Config(
                num_kv_pages_per_block=None,
                num_queries_per_block=None,
                num_warps=None,
                num_stages=None,
                platform="pallas",
                backend="tpu",
            )
        ]

    candidate_cfgs_shard_map_tpu = candidate_cfgs_tpu
    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu


_ragged_page_attention_executor: Executor[RaggedPageAttentionv3Config, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-page-attentionv3"),
    )
)


def ragged_page_attentionv3(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    /,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    vmem_limit_bytes: int | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RaggedPageAttentionv3Config | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Execute ragged page attention v3 with automatic optimization and optional sharding.

    This is the main entry point for ragged page attention v3, which combines variable-length
    (ragged) sequence processing with paged KV cache management. It provides the most
    memory-efficient attention implementation for LLM serving workloads by eliminating
    padding overhead while enabling flexible memory management through paged caching.

    The function automatically selects and caches optimal kernel configurations based on
    the input shapes and hardware platform. It supports both single-device execution and
    distributed execution via JAX's shard_map when mesh and partition specs are provided.

    Key Features:
        - Zero padding overhead through ragged layout
        - Efficient paged KV cache with flexible memory allocation
        - Automatic kernel selection and autotuning
        - Support for sliding window attention
        - Logit soft capping for numerical stability
        - Quantization support (q_scale, k_scale, v_scale)
        - Multi-platform support (Triton/Pallas/CUDA/XLA)
        - Distributed execution via shard_map
        - Persistent configuration caching

    Args:
        queries: Ragged query tensor [total_tokens, num_q_heads, head_dim].
            All sequences concatenated without padding. Sequence boundaries
            are defined by query_start_loc.
        keys: Key tensor [total_tokens, num_kv_heads, head_dim] for updating
            the KV cache with new tokens.
        values: Value tensor [total_tokens, num_kv_heads, head_dim] for updating
            the KV cache with new tokens.
        kv_cache: Paged KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded].
            Contains interleaved keys and values in paged format. The cache is
            organized as fixed-size pages that are dynamically allocated.
        kv_lens: Context lengths for each sequence [max_num_seqs].
            Indicates how many tokens are valid in each sequence's KV cache.
        block_tables: Block table mapping [max_num_seqs_times_pages_per_seq].
            Maps logical page indices to physical page indices in kv_cache.
            Flattened from shape [max_num_seqs, pages_per_seq].
        query_start_loc: Start indices for each sequence [max_num_seqs_plus_1].
            query_start_loc[i] to query_start_loc[i+1] defines the token range
            for sequence i in the ragged queries tensor. The last element equals
            total_tokens.
        distribution: Distribution parameters [3] containing:
            [num_seqs, pages_per_seq, page_size] for kernel execution.
        softmax_scale: Scaling factor applied to attention scores before softmax.
            Default is 1.0, typically set to 1/sqrt(head_dim) for numerical stability.
        sliding_window: Optional window size for sliding window attention.
            If None, uses full attention over all context. If set to a positive integer,
            limits attention to the last sliding_window tokens for each query.
        logits_soft_cap: Optional soft capping value for attention logits.
            Applies soft capping as: logits_soft_cap * tanh(logits / logits_soft_cap)
            to prevent extreme values and improve numerical stability.
        q_scale: Optional scaling factor for queries in quantized attention.
            Used to dequantize queries when they are stored in lower precision.
        k_scale: Optional scaling factor for keys in quantized attention.
            Used to dequantize keys when they are stored in lower precision.
        v_scale: Optional scaling factor for values in quantized attention.
            Used to dequantize values when they are stored in lower precision.
        vmem_limit_bytes: Memory limit for vector memory in bytes (TPU-specific).
            Controls VMEM usage on TPU accelerators, particularly important for
            large head dimensions (e.g., 256).
        platform: Target platform for kernel execution.
            One of "triton" (GPU), "pallas" (TPU), "cuda" (GPU), "xla" (fallback),
            or "auto" (automatic detection). If None, uses platform from cfg or
            auto-detection.
        cfg: Optional kernel configuration object. If None, uses automatic configuration
            selection with autotuning. Can specify:
            - num_kv_pages_per_block: Number of KV pages processed per block
            - num_queries_per_block: Number of query tokens processed per block
            - num_warps: Number of warps for Triton kernels (GPU-specific)
            - num_stages: Number of pipeline stages for Triton kernels (GPU-specific)
            - platform: Target platform
            - backend: Backend specification
        mesh: Optional JAX device mesh for distributed execution.
            If provided along with in_specs and out_specs, executes using shard_map
            for multi-device parallelism.
        in_specs: Optional tuple of PartitionSpec objects defining input tensor sharding.
            Must be provided if mesh is specified. Should contain specs for all 8 inputs:
            (queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc, distribution).
        out_specs: Optional PartitionSpec defining output tensor sharding.
            Must be provided if mesh is specified.

    Returns:
        Tuple containing:
            - Attention output [total_tokens, num_q_heads, head_dim]: Attention-weighted
              combination of values in ragged format, maintaining the same layout as queries.
            - Updated KV cache [num_pages, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded]:
              The KV cache with new key-value pairs incorporated.

    Raises:
        ValueError: If no suitable kernel implementation is found for the platform.
        AssertionError: If mesh is provided without in_specs or out_specs.
        AssertionError: If in_specs length doesn't match number of input arguments.

    Note:
        Performance Characteristics:
            - Ragged layout eliminates padding overhead, saving memory proportional to
              sequence length variance
            - Paged cache enables memory sharing for beam search and prefix caching
            - Automatic configuration caching avoids re-tuning for similar workloads
            - Sliding window attention reduces complexity from O(n²) to O(n*w) where w
              is the window size

        Memory Layout:
            - Queries: Ragged format [total_tokens, ...] with no padding between sequences
            - KV Cache: Paged format [num_pages, page_size, ...] with page-level granularity
            - Block Tables: Maps logical sequence pages to physical cache pages

        Distributed Execution:
            When mesh, in_specs, and out_specs are provided, the function uses JAX's
            shard_map to distribute computation across devices. This is essential for:
            - Large batch sizes that don't fit on a single device
            - Long contexts requiring memory distribution
            - Multi-node inference serving

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> from ejkernel.modules.operations import ragged_page_attentionv3
        >>>
        >>>
        >>> queries = jnp.ones((25, 32, 128), dtype=jnp.bfloat16)
        >>> keys = jnp.ones((25, 8, 128), dtype=jnp.bfloat16)
        >>> values = jnp.ones((25, 8, 128), dtype=jnp.bfloat16)
        >>> kv_cache = jnp.zeros((100, 16, 16, 2, 128), dtype=jnp.bfloat16)
        >>> kv_lens = jnp.array([10, 15], dtype=jnp.int32)
        >>> block_tables = jnp.arange(200, dtype=jnp.int32)
        >>> query_start_loc = jnp.array([0, 10, 25], dtype=jnp.int32)
        >>> distribution = jnp.array([2, 100, 16], dtype=jnp.int32)
        >>>
        >>>
        >>> output, updated_cache = ragged_page_attentionv3(
        ...     queries, keys, values, kv_cache, kv_lens,
        ...     block_tables, query_start_loc, distribution,
        ...     softmax_scale=1.0 / jnp.sqrt(128.0),
        ...     sliding_window=2048,
        ...     logits_soft_cap=30.0,
        ... )
        >>> output.shape
        >>> updated_cache.shape
        >>>
        >>>
        >>> devices = jax.devices()
        >>> mesh = Mesh(devices, axis_names=('data',))
        >>> P = PartitionSpec
        >>> output, updated_cache = ragged_page_attentionv3(
        ...     queries, keys, values, kv_cache, kv_lens,
        ...     block_tables, query_start_loc, distribution,
        ...     mesh=mesh,
        ...     in_specs=(P('data'), P('data'), P('data'), P(None), P('data'), P('data'), P('data'), P(None)),
        ...     out_specs=P('data'),
        ... )

    See Also:
        RaggedPageAttentionv3: The kernel class implementing the attention operation.
        RaggedPageAttentionv3Config: Configuration class for kernel parameters.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _ragged_page_attention_executor(
        RaggedPageAttentionv3(),
        queries=queries,
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        vmem_limit_bytes=vmem_limit_bytes,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
