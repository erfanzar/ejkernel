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

import math
import os
from typing import Literal

from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    FwdParams,
    Invocation,
    Kernel,
    Tuner,
)
from ejkernel.ops.config.persistent import PersistentCache

from ..base import detect_platform
from .configs import RaggedDecodeAttentionConfig


class RaggedDecodeAttention(Kernel[RaggedDecodeAttentionConfig, Array]):
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

    def get_impl(self, cfg: RaggedDecodeAttentionConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend preferences

        Returns:
            Callable kernel implementation for ragged decode attention

        Raises:
            ValueError: If no matching implementation is found for the configuration
        """
        platform = detect_platform("ragged_decode_attention", cfg.platform, maybe_pallas=True)
        return kernel_registry.get("ragged_decode_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        sequence_start: Int[Array, "batch"],
        sequence_end: Int[Array, "batch"],
        softmax_scale: float | None = None,
        sliding_window: tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: RaggedDecodeAttentionConfig,
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
            sliding_window: Optional (left, right) window sizes for local attention
            logits_soft_cap: Optional soft cap to bound attention logits
            softmax_aux: Optional attention sink logits for improved long-context performance
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object containing block_size parameter

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
            cfg = RaggedDecodeAttentionConfig(
                fwd_params=cfg.fwd_params,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
            fwd_params=cfg.fwd_params,
        )

    def heuristic_cfg(self, inv: Invocation[RaggedDecodeAttentionConfig, Array]) -> RaggedDecodeAttentionConfig:
        """Provide default configuration optimized for decode attention.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default KernelConfig with conservative block sizes suitable for
            typical decode scenarios (small query sizes, variable KV lengths)
        """
        return RaggedDecodeAttentionConfig(
            fwd_params=FwdParams(
                blocksize_heads=16,
                num_key_splits=16,
                kv_blocksize=128,
                num_warps=4,
                num_stages=2,
            ),
            platform="auto",
            backend="any",
        )

    def candidate_cfgs_gpu(self, inv: Invocation[RaggedDecodeAttentionConfig, Array]):
        """GPU/Triton candidates for ragged decode attention (bigger blocks + higher warps).

        - Explores kv_blocksize up to 256 (when split_len allows)
        - Tries blocksize_heads in {4, 8, 16} if grouped-heads permit
        - Warps up to 8 (depending on kv_block/head_dim)
        - Stages in {1, 2, 3} (kept low; smem-guarded)
        - Prefers split_len near {128, 256, 512}; ensures split_len % kv_blocksize == 0
        """
        q = inv.kwargs["query"]
        k = inv.kwargs["key"]
        v = inv.kwargs["value"]

        seq_len = int(k.shape[1])
        num_q_heads = int(q.shape[1])
        num_kv_heads = int(k.shape[2])
        head_dim = int(q.shape[-1])
        dtype = q.dtype

        assert num_kv_heads == int(v.shape[2])
        assert head_dim == int(k.shape[-1]) == int(v.shape[-1])
        assert num_q_heads % num_kv_heads == 0, "q_heads must be divisible by kv_heads"

        grouped_heads = num_q_heads // num_kv_heads

        preferred_split_lens = (64, 128, 256, 512)

        def best_splits(n: int, targets=preferred_split_lens, min_len=32, max_len=8192):
            divs = set()
            r = int(math.sqrt(n))
            for d in range(1, r + 1):
                if n % d == 0:
                    divs.add(d)
                    divs.add(n // d)
            valid = []
            for s in sorted(divs):
                sl = n // s
                if min_len <= sl <= max_len:
                    valid.append((s, sl))

            def score(sl):
                return min(abs(sl - t) for t in targets)

            valid.sort(key=lambda x: (score(x[1]), -x[1]))
            return valid

        split_candidates = best_splits(seq_len)

        head_opts = [h for h in (4, 8) if h <= grouped_heads] or [min(grouped_heads, 4)]

        kv_block_opts = [64, 128]

        smem_limit = int(os.getenv("EJKERNEL_TRITON_SMEM_LIMIT", str(99 * 1024)))
        elem_bytes = 2 if dtype in (jnp.float16, jnp.bfloat16) else 4

        def next_pow2_ge(x, min_val=16):
            return max(min_val, 1 << math.ceil(math.log2(max(1, x))))

        block_headdim = next_pow2_ge(head_dim, 16)

        def smem_est_bytes(block_heads: int, block_k: int, num_stages: int) -> int:
            kv_bytes = 2 * block_k * block_headdim * elem_bytes
            q_bytes = int(0.25 * block_heads * block_headdim * elem_bytes)
            stage_factor = 1.0 + 0.5 * max(0, num_stages - 2)
            fudge = 2.5
            return int((kv_bytes + q_bytes) * stage_factor * fudge)

        def warp_options(block_heads: int, block_k: int) -> list[int]:
            opts = [2, 4]
            if head_dim >= 128 or block_k >= 128:
                opts.append(8)
            return opts

        def stage_options(block_k: int) -> list[int]:
            return [1] if block_k <= 64 else [1, 2]

        seeds = []
        if seq_len % 64 == 0:
            seeds.append((seq_len // 64, 64))
        for s, sl in split_candidates[:6]:
            if (s, sl) not in seeds:
                seeds.append((s, sl))

        max_candidates = int(os.getenv("EJKERNEL_RDA_MAX_CANDIDATES", "32"))
        configs: list[RaggedDecodeAttentionConfig] = []
        seen = set()

        def try_add(H, K, s, sl):
            if K > sl or sl % K != 0:
                return False
            for W in warp_options(H, K):
                for S in stage_options(K):
                    if smem_est_bytes(H, K, S) > smem_limit:
                        continue
                    key = (H, K, s, W, S)
                    if key in seen:
                        continue
                    seen.add(key)
                    configs.append(
                        RaggedDecodeAttentionConfig(
                            fwd_params=FwdParams(
                                blocksize_heads=H,
                                num_key_splits=s,
                                kv_blocksize=K,
                                num_warps=W,
                                num_stages=S,
                            ),
                            platform="pallas",
                            backend="gpu",
                        )
                    )
                    if len(configs) >= max_candidates:
                        return True
            return False

        for s, sl in seeds:
            if try_add(4 if 4 in head_opts else head_opts[0], 64, s, sl):
                return configs

        for s, sl in seeds:
            for H in head_opts:
                for K in kv_block_opts:
                    if try_add(H, K, s, sl):
                        return configs

        for s, sl in split_candidates[:12]:
            for H in head_opts:
                for K in kv_block_opts:
                    if try_add(H, K, s, sl):
                        return configs

        if not configs:
            H = min(4, grouped_heads) if grouped_heads >= 4 else grouped_heads
            s = max(1, seq_len // 64)
            s = s if seq_len % s == 0 else 1
            sl = seq_len // s
            K = 64 if sl % 64 == 0 else (32 if sl % 32 == 0 else 16)
            try_add(H, K, s, sl)

        return configs

    def candidate_cfgs(self, inv: Invocation[RaggedDecodeAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Creates multiple configurations optimized for different decode scenarios,
        from small batches with short contexts to larger batches with longer contexts.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            Decode attention typically has small query dimensions (batch size),
            so candidates focus on optimizing block sizes.
        """
        block_configs = [
            (128, 4, 1),
            (256, 4, 1),
            (512, 8, 2),
        ]

        candidates = []
        for block_size, num_warps, num_stages in block_configs:
            candidates.append(
                RaggedDecodeAttentionConfig(
                    fwd_params=FwdParams(
                        blocksize_heads=16,
                        num_key_splits=16,
                        kv_blocksize=block_size,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    ),
                    platform="auto",
                    backend="any",
                )
            )

        return candidates

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_heads head_dim"],
        key: Float[Array, "batch seq_len num_kv_heads head_dim"],
        value: Float[Array, "batch seq_len num_kv_heads head_dim"],
        sequence_start: Int[Array, "batch"],
        sequence_end: Int[Array, "batch"],
        softmax_scale: float | None = None,
        sliding_window: tuple[int, int] | None = None,
        logits_soft_cap: float | None = None,
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: RaggedDecodeAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: PartitionSpec | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed execution.

        Creates a wrapper function that applies shard_map to distribute the ragged decode attention
        computation across devices according to the provided sharding specifications.

        Args:
            query: Query tensor [batch, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_kv_heads, head_dim]
            value: Value tensor [batch, seq_len, num_kv_heads, head_dim]
            sequence_start: Start indices for valid KV range per sequence [batch]
            sequence_end: End indices for valid KV range per sequence [batch]
            softmax_scale: Scaling factor for attention scores
            sliding_window: Optional (left, right) window sizes for local attention
            logits_soft_cap: Optional soft cap to bound attention logits
            softmax_aux: Optional attention sink logits
            platform: Platform to use for execution
            cfg: Configuration for the kernel
            mesh: JAX mesh for distributed execution
            in_specs: Partition specifications for input tensors
            out_specs: Partition specifications for output tensor
            check_vma: Whether to check for valid memory access patterns

        Returns:
            Tuple of (shard_map_fn, call_args) where shard_map_fn is the wrapped
            function and call_args are the arguments to pass to it.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapper(
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_aux,
        ):
            return self.run(
                query=query,
                key=key,
                value=value,
                sequence_start=sequence_start,
                sequence_end=sequence_end,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                softmax_aux=softmax_aux,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
            )

        shard_map_fn = shard_map(
            _wrapper,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        call_args = (
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_aux,
        )

        return shard_map_fn, call_args


_ragged_decode_attention_executor: Executor[RaggedDecodeAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("ragged-decode-attention", cfg_type=RaggedDecodeAttentionConfig),
    )
)


def ragged_decode_attention(
    query: Float[Array, "batch num_heads head_dim"] | Float[Array, "batch 1 num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: RaggedDecodeAttentionConfig | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: PartitionSpec | None = None,
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
        sliding_window: Optional (left, right) window sizes for local attention
        logits_soft_cap: Optional soft cap for attention logits (improves stability)
        softmax_aux: Optional attention sink values for long-context handling
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional config override (block_size is set via cfg)

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim]

    Example:
        >>>
        >>> out = ragged_decode_attention(q, k, v, starts, ends)
        >>>
        >>>
        >>> from ejkernel.modules.operations.configs import RaggedDecodeAttentionConfig
        >>> cfg = RaggedDecodeAttentionConfig(block_size=128)
        >>> out = ragged_decode_attention(
        ...     q, k, v, starts, ends,
        ...     sliding_window=(256, 256),
        ...     cfg=cfg
        ... )
        >>>
        >>>
        >>> out = ragged_decode_attention(
        ...     q, k, v, starts, ends,
        ...     logits_soft_cap=50.0,
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

    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"
    was4d = query.ndim == 4
    if was4d:
        query = query[:, -1, :, :]
    out = _ragged_decode_attention_executor(
        RaggedDecodeAttention(),
        query=query,
        key=key,
        value=value,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
    if was4d:
        out = jnp.expand_dims(out, 1)
    return out
