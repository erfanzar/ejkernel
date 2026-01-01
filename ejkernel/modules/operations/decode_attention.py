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

"""Paged decode attention module with automatic platform selection.

This operation wraps the vLLM-style decode attention kernel implemented in
`ejkernel.kernels` and provides a high-level API consistent with other
`ejkernel.modules.operations` entry points.
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
from .configs import DecodeAttentionConfig


class DecodeAttention(Kernel[DecodeAttentionConfig, tuple[Array, Array]]):
    """vLLM-style paged decode attention returning both output and LSE.

    This kernel performs decode-phase attention over a paged KV cache, where each
    query attends to its full context stored across multiple memory pages. The kernel
    returns both the attention output and the log-sum-exp (LSE) values for numerical
    stability and debugging.
    """

    def __init__(self):
        """Initialize the decode attention kernel."""
        super().__init__(op_id="decode_attention")

    def create_shard_map_wrapper(
        self,
        query: Float[Array, "batch num_q_heads head_dim"],
        key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
        value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
        req_to_tokens: Int32[Array, "batch max_pages"],
        seq_lens: Int32[Array, "batch"],
        *,
        softmax_scale: float | None = None,
        num_kv_splits: int | None = None,
        page_size: int = 1,
        logits_soft_cap: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: DecodeAttentionConfig | None = None,
        mesh: Mesh | None = None,
        in_specs: tuple[PartitionSpec, ...] | None = None,
        out_specs: tuple[PartitionSpec, PartitionSpec] | None = None,
        check_vma: bool = False,
    ):
        """Create a shard_map wrapper for distributed execution.

        Args:
            query: Query vectors [batch, num_q_heads, head_dim]
            key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
            value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
            req_to_tokens: Page table mapping [batch, max_pages]
            seq_lens: Sequence lengths in tokens [batch]
            softmax_scale: Attention scaling factor (default: 1/√head_dim)
            num_kv_splits: Number of KV splits for parallel processing (Triton only)
            page_size: Size of each memory page in tokens (default: 1)
            logits_soft_cap: Optional logit capping value
            platform: Target platform (triton/pallas/cuda/xla/auto)
            cfg: Optional kernel configuration
            mesh: JAX mesh for distributed execution
            in_specs: Input partition specifications
            out_specs: Output partition specifications
            check_vma: Whether to check virtual memory address validity

        Returns:
            Tuple of (shard_map_fn, call_args) for distributed execution.

        Raises:
            AssertionError: If mesh, in_specs, or out_specs are not provided, or if
                           in_specs length doesn't match the number of call arguments.
        """
        assert mesh is not None, "mesh must be provided for shard_map execution"
        assert in_specs is not None, "in_specs must be provided for shard_map execution"
        assert out_specs is not None, "out_specs must be provided for shard_map execution"

        def _wrapped_decode_attention(
            query: Float[Array, "batch num_q_heads head_dim"],
            key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
            value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
            req_to_tokens: Int32[Array, "batch max_pages"],
            seq_lens: Int32[Array, "batch"],
        ) -> tuple[
            Float[Array, "batch num_q_heads head_dim"],
            Float[Array, "batch num_q_heads"],
        ]:
            return self.run(
                query=query,
                key_buffer=key_buffer,
                value_buffer=value_buffer,
                req_to_tokens=req_to_tokens,
                seq_lens=seq_lens,
                softmax_scale=softmax_scale,
                num_kv_splits=num_kv_splits,
                page_size=page_size,
                logits_soft_cap=logits_soft_cap,
                platform=platform,
                cfg=cfg or self.heuristic_cfg(None),
            )

        call_args = (query, key_buffer, value_buffer, req_to_tokens, seq_lens)
        assert len(in_specs) == len(call_args), f"in_specs length {len(in_specs)} != call_args length {len(call_args)}"
        shard_map_fn = shard_map(
            _wrapped_decode_attention,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        return shard_map_fn, call_args

    def get_impl(self, cfg: DecodeAttentionConfig):
        """Get the platform-specific kernel implementation.

        Args:
            cfg: Kernel configuration specifying platform and backend.

        Returns:
            Platform-specific kernel implementation callable.
        """
        platform = detect_platform("decode_attention", cfg.platform)
        return kernel_registry.get("decode_attention", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch num_q_heads head_dim"],
        key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
        value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
        req_to_tokens: Int32[Array, "batch max_pages"],
        seq_lens: Int32[Array, "batch"],
        *,
        softmax_scale: float | None = None,
        num_kv_splits: int | None = None,
        page_size: int = 1,
        logits_soft_cap: float | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        cfg: DecodeAttentionConfig,
    ) -> tuple[
        Float[Array, "batch num_q_heads head_dim"],
        Float[Array, "batch num_q_heads"],
    ]:
        """Execute paged decode attention.

        Args:
            query: Query vectors [batch, num_q_heads, head_dim]
            key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
            value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
            req_to_tokens: Page table mapping logical to physical pages [batch, max_pages]
            seq_lens: Context length per sequence [batch]
            softmax_scale: Attention scaling factor (default: 1/√head_dim)
            num_kv_splits: Number of KV splits for parallelization (Triton-specific)
            page_size: Memory page size in tokens (default: 1)
            logits_soft_cap: Optional logit capping for numerical stability
            platform: Override platform selection (triton/pallas/cuda/xla/auto)
            cfg: Kernel configuration object

        Returns:
            Tuple of (attention_output, lse):
                - attention_output: [batch, num_q_heads, head_dim]
                - lse: Log-sum-exp values [batch, num_q_heads] (float32)
        """
        if platform is not None:
            cfg = DecodeAttentionConfig(
                num_kv_splits=cfg.num_kv_splits,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )

        if num_kv_splits is None:
            num_kv_splits = cfg.num_kv_splits

        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key_buffer=key_buffer,
            value_buffer=value_buffer,
            req_to_tokens=req_to_tokens,
            seq_lens=seq_lens,
            softmax_scale=softmax_scale,
            num_kv_splits=int(num_kv_splits),
            page_size=int(page_size),
            logits_soft_cap=logits_soft_cap,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )

    def heuristic_cfg(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]]):
        """Generate default heuristic configuration.

        Args:
            inv: Invocation context (unused).

        Returns:
            Default configuration with 16 KV splits and automatic platform selection.
        """
        del inv
        return DecodeAttentionConfig(
            num_kv_splits=16,
            num_warps=None,
            num_stages=None,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[DecodeAttentionConfig, tuple[Array, Array]]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation context (unused).

        Returns:
            Empty list (no autotuning candidates currently defined).
        """
        del inv
        return []


_decode_attention_executor: Executor[DecodeAttentionConfig, tuple[Array, Array]] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=2, iters=10),
        persistent=PersistentCache("decode-attention"),
    )
)


def decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    value_buffer: Float[Array, "total_tokens num_kv_heads head_dim"],
    req_to_tokens: Int32[Array, "batch max_pages"],
    seq_lens: Int32[Array, "batch"],
    /,
    *,
    softmax_scale: float | None = None,
    num_kv_splits: int | None = None,
    page_size: int = 1,
    logits_soft_cap: float | None = None,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    mesh: Mesh | None = None,
    in_specs: tuple[PartitionSpec | None, ...] | None = None,
    out_specs: tuple[PartitionSpec, PartitionSpec] | None = None,
    cfg: DecodeAttentionConfig | None = None,
) -> tuple[
    Float[Array, "batch num_q_heads head_dim"],
    Float[Array, "batch num_q_heads"],
]:
    """Execute vLLM-style paged decode attention.

    This function performs decode-phase attention where each query token (one per sequence)
    attends to its full context stored in a paged KV cache. The KV cache is organized into
    pages, and each sequence has a page table mapping logical pages to physical memory.

    Args:
        query: Query vectors [batch, num_q_heads, head_dim]
        key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
        value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
        req_to_tokens: Page table mapping [batch, max_pages], maps logical page indices
                      to physical page IDs for each sequence
        seq_lens: Context length per sequence [batch]
        softmax_scale: Attention scaling factor (default: 1/√head_dim)
        num_kv_splits: Number of KV splits for parallelization (Triton only, default: 16)
        page_size: Size of each memory page in tokens (default: 1)
        logits_soft_cap: Optional logit capping value for numerical stability
        platform: Target platform (triton/pallas/cuda/xla/auto)
        cfg: Optional kernel configuration
        mesh: Optional JAX mesh for distributed execution
        in_specs: Optional input partition specifications for distributed execution
        out_specs: Optional output partition specifications for distributed execution

    Returns:
        Tuple of (attention_output, lse):
            - attention_output: [batch, num_q_heads, head_dim]
            - lse: Log-sum-exp values [batch, num_q_heads] (float32)

    Example:
        ```python
        output, lse = decode_attention(
            query=q,
            key_buffer=k_buffer,
            value_buffer=v_buffer,
            req_to_tokens=page_table,
            seq_lens=sequence_lengths,
            page_size=16,
        )
        ```

    Note:
        The key/value buffers are organized as [num_pages * page_size, num_kv_heads, head_dim],
        where num_pages * page_size = total_tokens. The req_to_tokens tensor maps logical
        page indices to physical page IDs in this buffer.
    """
    method = None
    if mesh is not None and in_specs is not None and out_specs is not None:
        method = "shard_map"

    return _decode_attention_executor(
        DecodeAttention(),
        query=query,
        key_buffer=key_buffer,
        value_buffer=value_buffer,
        req_to_tokens=req_to_tokens,
        seq_lens=seq_lens,
        softmax_scale=softmax_scale,
        num_kv_splits=num_kv_splits,
        page_size=page_size,
        logits_soft_cap=logits_soft_cap,
        platform=platform,
        method=method,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        _cfg=cfg,
    )
