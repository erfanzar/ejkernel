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


"""Multi-head Latent Attention (MLA) module with automatic optimization.

This module implements Multi-head Latent Attention, a memory-efficient attention
variant that uses low-rank compression for key-value pairs. MLA reduces the KV cache
size by projecting keys and values through a low-rank bottleneck while maintaining
attention quality.

The key innovation is compressing the KV representations:
    1. Keys and values are projected to a low-rank space (kv_lora_rank)
    2. Compressed representations are stored efficiently
    3. Full-rank keys/values are reconstructed on-the-fly using learned weights

This is particularly beneficial for:
    - Long context inference where KV cache dominates memory
    - Multi-query or grouped-query attention patterns
    - Deployment scenarios with memory constraints
"""

from __future__ import annotations

import os
from typing import Literal

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
from .configs import FlashMLAConfig


class FlashMLA(Kernel[FlashMLAConfig, Array]):
    """Flash Multi-head Latent Attention with custom optimization logic.

    Combines flash attention's memory efficiency with MLA's low-rank KV compression.
    This implementation uses tiling and on-the-fly decompression to achieve both
    reduced memory footprint and computational efficiency.

    Features:
        - Low-rank KV compression via w_kc and w_vc weight matrices
        - Optional RoPE bias for positional encoding (b_q, b_k)
        - Flash attention-style tiling for memory efficiency
        - Support for causal masking and variable-length sequences
        - Multiple platform support (Triton/Pallas/CUDA/XLA)

    The compression scheme:
        - key_value: Compressed KV tensor [batch, seq_len, kv_lora_rank]
        - w_kc, w_vc: Decompression weights [kv_lora_rank, kv_heads, head_dim]
        - Keys/values are reconstructed as: key = key_value @ w_kc
    """

    def __init__(self):
        """Initialize Flash MLA module.

        Sets up the kernel with the operation identifier for registry lookup
        and configuration management.
        """
        super().__init__(op_id="flash_mla")

    def get_impl(self, cfg: FlashMLAConfig):
        """Get kernel implementation from registry.

        Args:
            cfg: Configuration specifying platform and backend

        Returns:
            Callable kernel implementation for flash MLA

        Raises:
            ValueError: If no matching implementation is found
        """
        platform = detect_platform("flash_mla", cfg.platform)
        return kernel_registry.get("flash_mla", platform=platform, backend=cfg.backend)

    def run(
        self,
        query: Float[Array, "batch seq_len q_heads head_dim"],
        key_value: Float[Array, "batch seq_len kv_lora_rank"],
        w_kc: Float[Array, "kv_lora_rank kv_heads head_dim"],
        w_vc: Float[Array, "kv_lora_rank kv_heads head_dim"],
        b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
        platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        *,
        cfg: FlashMLAConfig,
    ) -> Float[Array, "batch seq_len q_heads head_dim"]:
        """Execute flash multi-head latent attention.

        Args:
            query: Query tensor [batch, seq_len, q_heads, head_dim]
            key_value: Compressed key-value tensor [batch, seq_len, kv_lora_rank]
            w_kc: Key decompression weights [kv_lora_rank, kv_heads, head_dim]
            w_vc: Value decompression weights [kv_lora_rank, kv_heads, head_dim]
            b_q: Optional query RoPE bias [batch, seq_len, qk_rope_head_dim]
            b_k: Optional key RoPE bias [batch, seq_len, qk_rope_head_dim]
            softmax_scale: Optional scaling factor for attention scores
            causal: Whether to apply causal masking (default: False)
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            platform: Optional platform override ("triton", "pallas", "cuda", "xla")
            cfg: Kernel configuration object

        Returns:
            Attention output [batch, seq_len, q_heads, head_dim]

        Note:
            The kv_lora_rank determines the compression ratio. Lower ranks
            save more memory but may reduce quality. Typical values: 64-256.
        """

        if platform is not None:
            cfg = FlashMLAConfig(
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
            b_q=b_q,
            b_k=b_k,
            softmax_scale=softmax_scale,
            causal=causal,
            cu_seqlens=cu_seqlens,
        )

    def heuristic_cfg(self, inv: Invocation[FlashMLAConfig, Array]) -> FlashMLAConfig:
        """Provide default configuration with block sizes.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration optimized for MLA's low-rank decompression
            and on-the-fly reconstruction requirements
        """
        return FlashMLAConfig(
            block_q=128,
            block_k=128,
            num_warps=4,
            num_stages=2,
            platform="auto",
            backend="any",
        )

    def candidate_cfgs(self, inv: Invocation[FlashMLAConfig, Array]):
        """Generate candidate configurations for autotuning.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            List of candidate configurations to benchmark during autotuning

        Note:
            MLA performance depends on the compression rank and decompression
            overhead. Candidates balance memory efficiency with compute cost.
        """
        block_configs = [
            (64, 64, 4, 1),
            (128, 128, 4, 2),
            (256, 256, 8, 2),
        ]

        candidates = []
        for block_q, block_k, num_warps, num_stages in block_configs:
            candidates.append(
                FlashMLAConfig(
                    block_q=block_q,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    platform="auto",
                    backend="any",
                )
            )

        return candidates


_mla_executor: Executor[FlashMLAConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=True,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("mla"),
    )
)


def mla_attention(
    query: Float[Array, "batch seq_len q_heads head_dim"],
    key_value: Float[Array, "batch seq_len kv_lora_rank"],
    w_kc: Float[Array, "kv_lora_rank kv_heads head_dim"],
    w_vc: Float[Array, "kv_lora_rank kv_heads head_dim"],
    b_q: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    b_k: Float[Array, "batch seq_len qk_rope_head_dim"] | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    /,
    *,
    softmax_scale: float | None = None,
    causal: bool = False,
    platform: Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
    cfg: FlashMLAConfig | None = None,
) -> Float[Array, "batch seq_len q_heads head_dim"]:
    """Execute flash multi-head latent attention with automatic optimization.

    MLA uses low-rank compression for key-value pairs to reduce memory
    and computation while maintaining attention quality.

    Args:
        query: Query tensor [batch, seq_len, q_heads, head_dim]
        key_value: Compressed key-value tensor [batch, seq_len, kv_lora_rank]
        w_kc: Key compression weights [kv_lora_rank, kv_heads, head_dim]
        w_vc: Value compression weights [kv_lora_rank, kv_heads, head_dim]
        b_q: Query RoPE bias [batch, seq_len, qk_rope_head_dim]
        b_k: Key RoPE bias [batch, seq_len, qk_rope_head_dim]
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        cu_seqlens: Cumulative sequence lengths for variable-length sequences
        platform: Specific platform to use ("triton", "pallas", "cuda", or "xla")
        cfg: Optional kernel configuration override

    Returns:
        Attention output with same shape as query

    Example:
        >>>
        >>> out = mla_attention(query, key_value, w_kc, w_vc)
        >>>
        >>>
        >>> out = mla_attention(query, key_value, w_kc, w_vc, causal=True)
        >>>
        >>>
        >>> out = mla_attention(query, key_value, w_kc, w_vc, b_q=q_rope, b_k=k_rope)
            >>>
        >>>
        >>> out = mla_attention(..., platform="triton")
    """
    return _mla_executor(
        FlashMLA(),
        query=query,
        key_value=key_value,
        w_kc=w_kc,
        w_vc=w_vc,
        b_q=b_q,
        b_k=b_k,
        softmax_scale=softmax_scale,
        causal=causal,
        cu_seqlens=cu_seqlens,
        platform=platform,
        _cfg=cfg,
    )
