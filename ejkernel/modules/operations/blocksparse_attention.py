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


"""Attention module with automatic optimization."""

from __future__ import annotations

import typing

from jaxtyping import Array, Float, Int

from ejkernel.kernels._registry import kernel_registry
from ejkernel.ops import Invocation, Kernel

from ..base import KernelConfig, create_default_executor, detect_platform

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


class BlockSparseAttention(Kernel[KernelConfig, Array]):
    def __init__(self):
        super().__init__(op_id="block_sparse_attention")

    def get_impl(self, cfg: KernelConfig):
        return kernel_registry.get(
            algorithm="block_sparse_attention",
            platform=detect_platform("block_sparse_attention", cfg.platform),
            backend=cfg.backend,
        )

    def run(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        q_segment_ids: Int[Array, "batch seq_len"] | None = None,
        kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
        softmax_aux: Float[Array, "..."] | None = None,
        logit_soft_cap: float | None = None,
        query_chunk_size: int = 128,
        key_chunk_size: int = 128,
        softmax_scale: float | None = None,
        mask_builder: typing.Callable[[int, int, int, int, int], Mask] | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        chunk_size: int | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        *,
        cfg: KernelConfig,
    ) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
        impl = self.get_impl(cfg)
        return impl(
            query=query,
            key=key,
            value=value,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=softmax_aux,
            logit_soft_cap=logit_soft_cap,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            softmax_scale=softmax_scale,
            mask_builder=mask_builder,
            sliding_window=sliding_window,
            chunk_size=chunk_size,
            causal=causal,
            fused_backward=fused_backward,
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

        block_configs = [
            (128, 128),
            (128, 256),
            (256, 128),
            (256, 256),
        ]

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


def block_sparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
    logit_soft_cap: float | None = None,
    query_chunk_size: int = 128,
    key_chunk_size: int = 128,
    softmax_scale: float | None = None,
    mask_builder: typing.Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch kv_num_heads kv_len vhead_dim"]:
    return _executor(
        BlockSparseAttention(),
        query=query,
        key=key,
        value=value,
        q_segment_ids=q_segment_ids,
        kv_segment_ids=kv_segment_ids,
        softmax_aux=softmax_aux,
        logit_soft_cap=logit_soft_cap,
        query_chunk_size=query_chunk_size,
        key_chunk_size=key_chunk_size,
        softmax_scale=softmax_scale,
        mask_builder=mask_builder,
        sliding_window=sliding_window,
        chunk_size=chunk_size,
        causal=causal,
        fused_backward=fused_backward,
    )
