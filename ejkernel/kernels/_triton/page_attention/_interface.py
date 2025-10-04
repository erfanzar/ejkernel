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


import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, strides_from_shape, triton_call

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import _paged_attn_kernel, _paged_attn_v2_reduce_kernel


@kernel_registry.register("page_attention", Platform.TRITON, Backend.GPU)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    *,
    mask_value: float = -2.381976426469702e38,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    if pages_per_compute_block is not None:
        raise NotImplementedError("pages_per_compute_block is not supported in Triton implementation")
    if megacore_mode is not None:
        raise NotImplementedError("megacore_mode is not supported in Triton implementation")
    if not inline_seq_dim:
        raise NotImplementedError("inline_seq_dim=False is not supported in Triton implementation")
    if attn_logits_soft_cap is not None:
        raise NotImplementedError("attn_logits_soft_cap is not supported in Triton implementation")

    num_seqs = query.shape[0]
    num_kv_heads = key_cache.shape[1]
    kv_block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    query_group_size = query.shape[1] // num_kv_heads

    if attn_scale is None:
        attn_scale = 1.0 / (head_size**0.5)

    if max_context_len is None:
        max_context_len = int(context_lens.max())

    # Compute padded group size
    if query_group_size == 1:
        padded_group_size = 1
    elif query_group_size < 16:
        padded_group_size = 16
    else:
        padded_group_size = 1 << (query_group_size - 1).bit_length()  # next power of 2

    assert head_size in (16, 32, 64, 128, 256, 512), f"head_size={head_size}"
    assert padded_group_size == 1 or kv_block_size >= 16, f"kv_block_size={kv_block_size}"

    # Determine partitioning strategy
    num_sms = 108  # Default for A100, can be made dynamic
    if num_splits == 0:
        if num_seqs * num_kv_heads > 2 * num_sms:
            num_splits = 1
            if max_context_len >= 4096:
                partition_size = max(256, kv_block_size)
                num_splits = cdiv(max_context_len, partition_size)
        else:
            partition_size = max(256, kv_block_size)
            num_splits = cdiv(max_context_len, partition_size)
            if max_context_len <= 1024 or kv_block_size >= 256:
                num_splits = 1
    elif num_splits > 1:
        partition_size = cdiv(max_context_len, num_splits)
        partition_size = 1 << (partition_size - 1).bit_length()  # next power of 2

    # Compute strides using eformer utility
    stride_bt0, stride_bt1 = strides_from_shape(block_tables.shape)
    stride_q0, stride_q1, stride_q2 = strides_from_shape(query.shape)
    stride_kv0, stride_kv1, stride_kv2, stride_kv3 = strides_from_shape(key_cache.shape)

    if num_splits == 1:
        # Single-pass attention
        out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)

        def grid(meta):
            return (num_seqs, num_kv_heads, 1)

        # out strides same as query for single-pass
        stride_o0 = stride_q0
        stride_o1 = stride_q1
        stride_o2 = stride_q2
        stride_o3 = stride_q2
        stride_o4 = stride_q2

        metaparams = dict(
            grid=grid,
            attn_scale=attn_scale,
            stride_bt0=stride_bt0,
            stride_bt1=stride_bt1,
            stride_q0=stride_q0,
            stride_q1=stride_q1,
            stride_q2=stride_q2,
            stride_kv0=stride_kv0,
            stride_kv1=stride_kv1,
            stride_kv2=stride_kv2,
            stride_kv3=stride_kv3,
            stride_o0=stride_o0,
            stride_o1=stride_o1,
            stride_o2=stride_o2,
            stride_o3=stride_o3,
            stride_o4=stride_o4,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PADDED_QUERY_GROUP_SIZE=padded_group_size,
            NUM_KV_HEADS=num_kv_heads,
            KV_BLOCK_SIZE=kv_block_size,
            PARTITION_SIZE=0,
        )

        dummy_m_shape = jax.ShapeDtypeStruct((num_seqs, num_kv_heads, 1, query_group_size), jnp.float32)
        dummy_l_shape = jax.ShapeDtypeStruct((num_seqs, num_kv_heads, 1, query_group_size), jnp.float32)

        *_, out = triton_call(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            kernel=_paged_attn_kernel,
            out_shape=(dummy_m_shape, dummy_l_shape, out_shape),
            **metaparams,
        )

    else:
        # Multi-pass attention with reduction
        tmp_out_shape = jax.ShapeDtypeStruct(
            (num_seqs, num_kv_heads, num_splits, query_group_size, head_size),
            query.dtype,
        )
        m_i_shape = jax.ShapeDtypeStruct(
            (num_seqs, num_kv_heads, num_splits, query_group_size),
            jnp.float32,
        )
        l_i_shape = m_i_shape

        def grid(meta):
            return (num_seqs, num_kv_heads, num_splits)

        assert (partition_size >= kv_block_size) and (partition_size % kv_block_size == 0), (
            f"partition_size={partition_size}, kv_block_size={kv_block_size}"
        )

        metaparams = dict(
            grid=grid,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PADDED_QUERY_GROUP_SIZE=padded_group_size,
            NUM_KV_HEADS=num_kv_heads,
            KV_BLOCK_SIZE=kv_block_size,
            PARTITION_SIZE=partition_size,
        )

        # Compute tmp_out strides: [num_seqs, num_kv_heads, num_splits, query_group_size, head_size]
        stride_tmp0, stride_tmp1, stride_tmp2, stride_tmp3, stride_tmp4 = strides_from_shape(
            (num_seqs, num_kv_heads, num_splits, query_group_size, head_size)
        )

        metaparams["attn_scale"] = attn_scale
        metaparams["stride_bt0"] = stride_bt0
        metaparams["stride_bt1"] = stride_bt1
        metaparams["stride_q0"] = stride_q0
        metaparams["stride_q1"] = stride_q1
        metaparams["stride_q2"] = stride_q2
        metaparams["stride_kv0"] = stride_kv0
        metaparams["stride_kv1"] = stride_kv1
        metaparams["stride_kv2"] = stride_kv2
        metaparams["stride_kv3"] = stride_kv3
        metaparams["stride_o0"] = stride_tmp0
        metaparams["stride_o1"] = stride_tmp1
        metaparams["stride_o2"] = stride_tmp2
        metaparams["stride_o3"] = stride_tmp3
        metaparams["stride_o4"] = stride_tmp4

        # First pass: compute partial outputs
        # Kernel signature has inputs first, then outputs: q, k, v, context_lens, block_tables, m_i_ptr, l_i_ptr, out_ptr
        m_i, l_i, tmp_out = triton_call(
            query,
            key_cache,
            value_cache,
            context_lens,
            block_tables,
            kernel=_paged_attn_kernel,
            out_shape=(m_i_shape, l_i_shape, tmp_out_shape),
            **metaparams,
        )

        # Second pass: reduce partial outputs
        out_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)

        def reduce_grid(meta):
            return (num_seqs, num_kv_heads)

        next_num_splits = 1 << (num_splits - 1).bit_length()  # next power of 2

        reduce_metaparams = dict(
            grid=reduce_grid,
            max_num_partitions=num_splits,
            stride_o0=stride_q0,
            stride_o1=stride_q1,
            stride_o2=stride_q2,
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            NUM_KV_HEADS=num_kv_heads,
            PARTITION_SIZE=partition_size,
            NUM_PARTITIONS=next_num_splits,
        )

        out = triton_call(
            m_i,
            l_i,
            tmp_out,
            context_lens,
            kernel=_paged_attn_v2_reduce_kernel,
            out_shape=out_shape,
            **reduce_metaparams,
        )

    return out
