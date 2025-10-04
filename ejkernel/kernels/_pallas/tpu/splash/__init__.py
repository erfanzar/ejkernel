# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import typing

import jax
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_kernel import (
    BlockSizes,
    QKVLayout,
    SegmentIds,
    make_masked_mha_reference,
    make_masked_mqa_reference,
    make_splash_mha,
    make_splash_mha_single_device,
    make_splash_mqa,
    make_splash_mqa_single_device,
)
from jax.experimental.pallas.ops.tpu.splash_attention.splash_attention_mask import (
    CausalMask,
    FullMask,
    LocalMask,
    Mask,
    MultiHeadMask,
    NumpyMask,
    make_causal_mask,
    make_local_attention_mask,
    make_random_mask,
)
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit
from ejkernel.kernels._registry import Backend, Platform, kernel_registry


@kernel_registry.register("splash", Platform.PALLAS, Backend.TPU)
@ejit(static_argnames=("query_chunk_size", "key_chunk_size", "sm_scale", "mask_builder"))
def splash(
    q: Float[Array, "batch num_heads seq_len head_dim"],
    k: Float[Array, "batch num_heads kv_len head_dim"],
    v: Float[Array, "batch num_heads kv_len head_dim"],
    q_mask: Int[Array, "batch seq_len"] | None = None,
    kv_mask: Int[Array, "batch kv_len"] | None = None,
    query_chunk_size: int = 128,
    key_chunk_size: int = 128,
    sm_scale: float | None = None,
    mask_builder: typing.Callable[[int, int, int], Mask] | None = None,
) -> Float[Array, "batch num_heads seq_len head_dim"]:
    if mask_builder is None:

        def mask_builder(q_len: int, kv_len: int, num_heads: int) -> Mask:
            if q_len == kv_len:
                return CausalMask((q_len, kv_len))
            else:
                return FullMask((q_len, kv_len))

    block_sizes = BlockSizes(
        block_q=min(query_chunk_size, q.shape[1]),
        block_kv_compute=min(key_chunk_size, v.shape[1]),
        block_kv=min(key_chunk_size, v.shape[1]),
        block_q_dkv=min(query_chunk_size, q.shape[1]),
        block_kv_dkv=min(key_chunk_size, v.shape[1]),
        block_kv_dkv_compute=min(key_chunk_size, v.shape[1]),
        block_q_dq=min(query_chunk_size, q.shape[1]),
        block_kv_dq=min(key_chunk_size, v.shape[1]),
    )
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    if q.shape[-2] != 1:
        output_shape = (*q.shape[:-1], v.shape[-1])
        num_reps = q.shape[1] // k.shape[1]
        q = q.reshape((*q.shape[:-3], k.shape[-3], num_reps, q.shape[-2], q.shape[-1]))
        fn = jax.vmap(
            jax.vmap(
                make_splash_mqa_single_device(
                    mask=MultiHeadMask([mask_builder(q.shape[-2], k.shape[-2], ox) for ox in range(q.shape[-3])]),
                    block_sizes=block_sizes,
                ),
                in_axes=(0, 0, 0, None),
            ),
            in_axes=(0, 0, 0, 0),
        )
        m = None
        if kv_mask is not None:
            m = SegmentIds(q_mask, kv_mask)
        out = fn(q * sm_scale, k, v, m).reshape(output_shape)
    return out


__all__ = (
    "BlockSizes",
    "CausalMask",
    "FullMask",
    "LocalMask",
    "Mask",
    "MultiHeadMask",
    "NumpyMask",
    "QKVLayout",
    "SegmentIds",
    "make_causal_mask",
    "make_local_attention_mask",
    "make_masked_mha_reference",
    "make_masked_mqa_reference",
    "make_random_mask",
    "make_splash_mha",
    "make_splash_mha_single_device",
    "make_splash_mqa",
    "make_splash_mqa_single_device",
    "splash",
)
