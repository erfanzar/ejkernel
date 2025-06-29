# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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


from functools import partial

import jax
import triton
from eformer.callib import triton_call
from jax import numpy as jnp

from ...utils import cdiv, get_stride, is_hip, next_power_of_2
from ._triton_impl_fwd import _fwd_grouped_kernel_stage1, _fwd_kernel_stage1, _fwd_kernel_stage2


def compute_attention_logits_and_values(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    slot_mapping: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float,
):
    """
    Compute attention logits and values for decode phase (stage 1).

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        key_cache: Cached key states
        value_cache: Cached value states
        slot_mapping: Mapping from requests to token positions
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation
        softmax_scale: Scaling factor for attention scores
        page_size: Size of memory pages
        logit_cap: Maximum value for attention logits

    Returns:
        Attention logits and intermediate values
    """
    # Set block size based on hardware
    block_size = 64 if not is_hip() else 8

    key_dim = key_cache.shape[-1]
    value_dim = value_cache.shape[-1]

    batch_size, num_query_heads = query.shape[0], query.shape[1]

    # Calculate grouped query attention ratio
    key_value_group_size = query.shape[1] // key_cache.shape[-2]

    # Set number of warps based on grouping
    num_warps = 4
    if key_value_group_size != 1:
        num_warps = 1 if is_hip() else 2

    # Calculate block dimensions (must be powers of 2 for Triton)
    key_block_dim = triton.next_power_of_2(key_dim)
    value_block_dim = triton.next_power_of_2(value_dim)

    kernel_params = dict(
        KVGroups=key_value_group_size,
        BLOCK_DMODEL=key_block_dim,
        BLOCK_DV=value_block_dim,
        BLOCK_N=block_size,
        NUM_KV_SPLITS=num_key_value_splits,
        PAGE_SIZE=page_size,
        BLOCK_DIMK=key_dim,
        BLOCK_DIMV=value_dim,
        LOGIT_CAP=logit_cap,
    )

    # Output shape includes space for attention scores
    output_shape = (query.shape[0], query.shape[1], num_key_value_splits, value_cache.shape[-1] + 1)

    (attention_output,) = triton_call(
        query,
        key_cache,
        value_cache,
        softmax_scale,
        slot_mapping,
        batch_sequence_lengths,
        get_stride(slot_mapping, 0),
        get_stride(query, 0),
        get_stride(query, 1),
        get_stride(key_cache, -3),
        get_stride(key_cache, -2),
        get_stride(value_cache, -3),
        get_stride(value_cache, -2),
        get_stride(output_shape, 0),
        get_stride(output_shape, 1),
        get_stride(output_shape, 2),
        kernel=_fwd_kernel_stage1,
        out_shape=[jax.ShapeDtypeStruct(output_shape, query.dtype)],
        grid=lambda META: (batch_size, num_query_heads, num_key_value_splits),
        name="egjpu::decode_attention::_fwd_kernel_stage1",
        num_warps=num_warps,
        num_stages=2,
        **kernel_params,
    )
    return attention_output


def compute_grouped_attention_logits_and_values(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    slot_mapping: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float,
):
    """
    Compute attention logits and values for grouped query attention (stage 1).

    Optimized version for grouped query attention where multiple query heads
    share the same key-value heads.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        key_cache: Cached key states
        value_cache: Cached value states
        slot_mapping: Mapping from requests to token positions
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation
        softmax_scale: Scaling factor for attention scores
        page_size: Size of memory pages
        logit_cap: Maximum value for attention logits

    Returns:
        Attention logits and intermediate values for grouped attention
    """
    block_size = 32
    key_dim = key_cache.shape[-1]
    value_dim = value_cache.shape[-1]

    # Adjust block size for HIP backend with large dimensions
    if is_hip() and key_dim >= 576:
        block_size = 16

    # Special handling for common dimension sizes
    if key_dim == 576:
        key_block_dim = 512
        positional_encoding_block_dim = 64
    elif key_dim == 288:
        key_block_dim = 256
        positional_encoding_block_dim = 32
    else:
        key_block_dim = next_power_of_2(key_dim)
        positional_encoding_block_dim = 0

    value_block_dim = next_power_of_2(value_dim)

    batch_size, num_query_heads = query.shape[0], query.shape[1]
    key_value_group_size = query.shape[1] // key_cache.shape[-2]

    head_block_size = 16

    # Adjust stages for HIP backend
    num_stages = 2
    if is_hip():
        num_stages = 1

    output_shape = (query.shape[0], query.shape[1], num_key_value_splits, value_cache.shape[-1] + 1)

    kernel_params = dict(
        KVGroups=key_value_group_size,
        Qheads=num_query_heads,
        BLOCK_DMODEL=key_block_dim,
        BLOCK_DPE=positional_encoding_block_dim,
        BLOCK_DV=value_block_dim,
        BLOCK_N=block_size,
        BLOCK_H=head_block_size,
        NUM_KV_SPLITS=num_key_value_splits,
        PAGE_SIZE=page_size,
        LOGIT_CAP=logit_cap,
        BLOCK_DIMK=key_dim,
        BLOCK_DIMV=value_dim,
    )

    (attention_output,) = triton_call(
        query,
        key_cache,
        value_cache,
        softmax_scale,
        slot_mapping,
        batch_sequence_lengths,
        get_stride(slot_mapping, 0),
        get_stride(query, 0),
        get_stride(query, 1),
        get_stride(key_cache, -3),
        get_stride(key_cache, -2),
        get_stride(value_cache, -3),
        get_stride(value_cache, -2),
        get_stride(output_shape, 0),
        get_stride(output_shape, 1),
        get_stride(output_shape, 2),
        kernel=_fwd_grouped_kernel_stage1,
        out_shape=[jax.ShapeDtypeStruct(output_shape, jnp.float32)],
        grid=lambda META: (
            batch_size,
            cdiv(num_query_heads, min(head_block_size, key_value_group_size)),
            num_key_value_splits,
        ),
        name="egjpu::decode_attention::_fwd_grouped_kernel_stage1",
        num_warps=4,
        num_stages=num_stages,
        **kernel_params,
    )
    return attention_output


def apply_softmax_and_reduce_values(
    attention_logits: jax.Array,
    query: jax.Array,
    value_cache: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
):
    """
    Apply softmax to attention logits and reduce values (stage 2).

    Args:
        attention_logits: Raw attention logits from stage 1
        query: Query tensor (used for output shape)
        value_cache: Cached value states
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation

    Returns:
        Final attention output after softmax and value reduction
    """
    batch_size, num_heads, _ = query.shape
    value_dim = value_cache.shape[-1]
    value_block_dim = next_power_of_2(value_dim)

    kernel_params = dict(
        NUM_KV_SPLITS=num_key_value_splits,
        BLOCK_DV=value_block_dim,
        BLOCK_DIMV=value_dim,
    )

    output_shape = [jax.ShapeDtypeStruct(query.shape, query.dtype)]

    (final_output,) = triton_call(
        attention_logits,
        batch_sequence_lengths,
        get_stride(attention_logits, 0),
        get_stride(attention_logits, 1),
        get_stride(attention_logits, 2),
        get_stride(query, 0),
        get_stride(query, 1),
        kernel=_fwd_kernel_stage2,
        out_shape=output_shape,
        grid=lambda META: (batch_size, num_heads),
        name="egjpu::decode_attention::_fwd_kernel_stage2",
        num_warps=4,
        num_stages=2,
        **kernel_params,
    )
    return final_output


def decode_attention_standard(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    slot_mapping: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
):
    """
    Standard decode attention for multi-head attention.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        key_cache: Cached key states
        value_cache: Cached value states
        slot_mapping: Mapping from requests to token positions
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation
        softmax_scale: Scaling factor for attention scores
        page_size: Size of memory pages
        logit_cap: Maximum value for attention logits

    Returns:
        Attention output tensor
    """
    attention_logits = compute_attention_logits_and_values(
        query,
        key_cache,
        value_cache,
        slot_mapping,
        batch_sequence_lengths,
        num_key_value_splits,
        softmax_scale,
        page_size,
        logit_cap,
    )

    return apply_softmax_and_reduce_values(
        attention_logits,
        query,
        value_cache,
        batch_sequence_lengths,
        num_key_value_splits,
    )


def decode_attention_grouped(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    slot_mapping: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
    softmax_scale: float,
    page_size: int,
    logit_cap: float = 0.0,
):
    """
    Grouped query attention decode for efficient inference.

    Used when multiple query heads share the same key-value heads,
    which is common in models like Llama-2.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        key_cache: Cached key states
        value_cache: Cached value states
        slot_mapping: Mapping from requests to token positions
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation
        softmax_scale: Scaling factor for attention scores
        page_size: Size of memory pages
        logit_cap: Maximum value for attention logits

    Returns:
        Attention output tensor
    """
    attention_logits = compute_grouped_attention_logits_and_values(
        query,
        key_cache,
        value_cache,
        slot_mapping,
        batch_sequence_lengths,
        num_key_value_splits,
        softmax_scale,
        page_size,
        logit_cap,
    )

    return apply_softmax_and_reduce_values(
        attention_logits,
        query,
        value_cache,
        batch_sequence_lengths,
        num_key_value_splits,
    )


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def decode_attention(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    slot_mapping: jax.Array,
    batch_sequence_lengths: jax.Array,
    num_key_value_splits: int,
    softmax_scale: float | None = None,
    page_size: int = 1,
    logit_cap: float = 0.0,
):
    """
    Main decode attention function that automatically selects the appropriate implementation.

    Chooses between standard multi-head attention and grouped query attention
    based on the ratio of query heads to key-value heads.

    Args:
        query: Query tensor [batch, num_heads, head_dim]
        key_cache: Cached key states
        value_cache: Cached value states
        slot_mapping: Mapping from requests to token positions
        batch_sequence_lengths: Length of each sequence in batch
        num_key_value_splits: Number of splits for key-value computation
        softmax_scale: Scaling factor for attention scores (typically 1/sqrt(head_dim))
        page_size: Size of memory pages for paged attention
        logit_cap: Maximum value for attention logits to prevent overflow

    Returns:
        Attention output tensor [batch, num_heads, head_dim]
    """

    key_value_group_size = query.shape[1] // value_cache.shape[-2]
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if key_value_group_size == 1:
        return decode_attention_standard(
            query,
            key_cache,
            value_cache,
            slot_mapping,
            batch_sequence_lengths,
            num_key_value_splits,
            softmax_scale,
            page_size,
            logit_cap,
        )
    else:
        # Grouped query attention (multiple query heads per key-value head)
        return decode_attention_grouped(
            query,
            key_cache,
            value_cache,
            slot_mapping,
            batch_sequence_lengths,
            num_key_value_splits,
            softmax_scale,
            page_size,
            logit_cap,
        )
