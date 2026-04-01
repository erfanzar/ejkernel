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


"""DeepSeek Sparse Attention (DSA) module with automatic optimization.

Implements DeepSeek-V3.2's sparse attention on top of MLA. Follows the same
tensor conventions as flash_mla: compressed KV latent with on-the-fly
projection via w_kc/w_vc, plus optional RoPE via b_q/b_k.

The DSA-specific additions are the Lightning Indexer inputs:
  - query_index: Lightweight indexer query projections (reuses MLA's q_lora)
  - key_index: Indexer key projections (from hidden states, shared across heads)
  - index_weights: Learned per-head aggregation weights

Architecture:
    Phase 1 — Lightning Indexer: Cheap FP8-friendly scorer that selects
    top-k KV tokens per query using ReLU activation + learned head weights.
    Phase 2 — Sparse MLA: Standard MLA attention with non-selected tokens
    masked to -inf before softmax.

References:
    - DeepSeek-V3.2: https://arxiv.org/abs/2512.02556
    - DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
"""

from __future__ import annotations

import os
from typing import Literal

from jaxtyping import Array, Float

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
from .configs import DeepSeekAttentionConfig


class DeepSeekAttention(Kernel[DeepSeekAttentionConfig, Array]):
    """DeepSeek Sparse Attention with MLA + Lightning Indexer.

    Combines MLA attention (compressed KV latent, on-the-fly projection)
    with a lightweight learned indexer that dynamically selects top-k
    KV tokens per query position.
    """

    def __init__(self):
        super().__init__(op_id="deepseek_attn")

    def get_impl(self, cfg: DeepSeekAttentionConfig):
        """Get kernel implementation from registry based on configuration.

        Args:
            cfg: Configuration specifying platform and backend preferences.

        Returns:
            Callable kernel implementation for DSA (Pallas/TPU or XLA fallback).
        """
        platform = detect_platform("deepseek_attn", cfg.platform)
        return kernel_registry.get("deepseek_attn", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len q_heads q_head_dim"],
        key_value: Float[Array, "batch seq_len kv_lora_rank"],
        w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
        w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
        query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
        key_index: Float[Array, "batch seq_len index_head_dim"],
        index_weights: Float[Array, "batch seq_len index_heads"],
        index_topk: int = 2048,
        softmax_scale: float | None = None,
        index_softmax_scale: float | None = None,
        b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        causal: bool = True,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto", "cute"] | None = None,
        *,
        cfg: DeepSeekAttentionConfig,
    ) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
        """Execute DeepSeek Sparse Attention.

        Args:
            query: Query tensor [batch, seq_len, q_heads, q_head_dim].
            key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
            w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
            w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
            query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
            key_index: Indexer keys [batch, seq_len, index_head_dim].
            index_weights: Per-head weights [batch, seq_len, index_heads].
            index_topk: Tokens to select (default: 2048).
            softmax_scale: Attention scale.
            index_softmax_scale: Indexer scale.
            b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
            b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
            causal: Causal masking (default: True).
            platform: Optional platform override.
            cfg: Kernel configuration.

        Returns:
            Attention output [batch, seq_len, q_heads, v_head_dim].
        """
        if platform is not None:
            cfg = DeepSeekAttentionConfig(
                index_topk=cfg.index_topk,
                block_q=cfg.block_q,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                platform=platform,
                backend=Backend.ANY if platform == "xla" else cfg.backend,
            )
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key_value=key_value,
            w_kc=w_kc,
            w_vc=w_vc,
            query_index=query_index,
            key_index=key_index,
            index_weights=index_weights,
            index_topk=index_topk if index_topk != 2048 else cfg.index_topk,
            softmax_scale=softmax_scale,
            index_softmax_scale=index_softmax_scale,
            b_q=b_q,
            b_k=b_k,
            causal=causal,
        )

    def heuristic_cfg(self, inv: Invocation[DeepSeekAttentionConfig, Array]) -> DeepSeekAttentionConfig:
        """Generate default configuration with balanced block sizes.

        Uses ``index_topk=2048`` and 128x128 query/key blocks as a safe
        starting point before autotuning has profiled the workload.

        Args:
            inv: Invocation containing input tensors and metadata.

        Returns:
            Default ``DeepSeekAttentionConfig`` suitable for most sequence lengths.
        """
        return DeepSeekAttentionConfig(
            index_topk=2048,
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[DeepSeekAttentionConfig, Array]):
        """Generate candidate configurations for autotuning.

        Explores the cross-product of ``index_topk`` (1024, 2048, 4096) and
        query/key block sizes (64, 128, 256).  Larger ``index_topk`` retains
        more KV tokens and increases accuracy at the cost of compute; larger
        blocks improve MXU utilisation but increase per-tile memory.

        Args:
            inv: Invocation containing input tensors and metadata.

        Returns:
            List of ``DeepSeekAttentionConfig`` candidates to benchmark.
        """
        candidates = []
        for index_topk in [1024, 2048, 4096]:
            for block_q, block_k in [(64, 64), (128, 128), (256, 256)]:
                candidates.append(
                    DeepSeekAttentionConfig(
                        index_topk=index_topk,
                        block_q=block_q,
                        block_k=block_k,
                        num_warps=4,
                        num_stages=2,
                        platform="auto",
                        backend="any",
                    )
                )
        return candidates


_dsa_executor: Executor[DeepSeekAttentionConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("dsa"),
    )
)


def deepseek_attn(
    query: Float[Array, "batch seq_len q_heads q_head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads qk_nope_head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads v_head_dim"],
    query_index: Float[Array, "batch seq_len index_heads index_head_dim"],
    key_index: Float[Array, "batch seq_len index_head_dim"],
    index_weights: Float[Array, "batch seq_len index_heads"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    /,
    *,
    index_topk: int = 2048,
    softmax_scale: float | None = None,
    index_softmax_scale: float | None = None,
    causal: bool = True,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto", "cute"] | None = None,
    cfg: DeepSeekAttentionConfig | None = None,
) -> Float[Array, "batch seq_len q_heads v_head_dim"]:
    """Execute DeepSeek Sparse Attention with automatic optimization.

    DSA = MLA + Lightning Indexer. Same tensor conventions as flash_mla:
      - key_value: Compressed KV latent [batch, seq_len, kv_lora_rank]
      - w_kc/w_vc: On-the-fly K/V projection weights
      - b_q/b_k: Optional RoPE bias

    Plus DSA-specific Lightning Indexer inputs:
      - query_index: Indexer queries (derived from MLA's q_lora)
      - key_index: Indexer keys (from hidden states, shared across heads)
      - index_weights: Learned per-head aggregation weights

    Args:
        query: Query tensor [batch, seq_len, q_heads, q_head_dim].
        key_value: Compressed KV latent [batch, seq_len, kv_lora_rank].
        w_kc: Key projection [kv_lora_rank, kv_heads, qk_nope_head_dim].
        w_vc: Value projection [kv_lora_rank, kv_heads, v_head_dim].
        query_index: Indexer queries [batch, seq_len, index_heads, index_head_dim].
        key_index: Indexer keys [batch, seq_len, index_head_dim].
        index_weights: Per-head weights [batch, seq_len, index_heads].
        b_q: Optional query RoPE [batch, seq_len, qk_rope_head_dim].
        b_k: Optional key RoPE [batch, seq_len, qk_rope_head_dim].
        index_topk: Tokens to select per query (default: 2048).
        softmax_scale: Attention scale (default: effective_head_dim^-0.5).
        index_softmax_scale: Indexer scale (default: index_head_dim^-0.5).
        causal: Causal masking (default: True).
        platform: Platform override ("triton", "pallas", "cuda", "xla").
        cfg: Optional configuration override.

    Returns:
        Attention output [batch, seq_len, q_heads, v_head_dim].

    Example:
        >>> output = deepseek_attn(
        ...     query, key_value, w_kc, w_vc,
        ...     query_index, key_index, index_weights,
        ...     index_topk=2048, causal=True,
        ... )
    """
    return _dsa_executor(
        DeepSeekAttention(),
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        query_index=query_index,
        key_index=key_index,
        index_weights=index_weights,
        index_topk=index_topk,
        softmax_scale=softmax_scale,
        index_softmax_scale=index_softmax_scale,
        b_q=b_q,
        b_k=b_k,
        causal=causal,
        platform=platform,
        _cfg=cfg,
    )
