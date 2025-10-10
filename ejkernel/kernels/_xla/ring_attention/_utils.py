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


import chex
import jax
import jax.lax as lax
from jax import numpy as jnp
from jaxtyping import DTypeLike


def _chunk_attention_bias(
    query_chunk_size: int,
    key_chunk_size: int,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    deterministic: bool,
    attn_dropout: chex.Array | None,
    pdrop: float,
    causal_block_size: int | None,
    dtype: DTypeLike,
    query_chunk_idx: int,
    key_chunk_idx: int,
    sliding_window: int | tuple[int, int] | None = None,
    attention_sink_size: int = 0,
):
    """Computes the attention bias for a chunk of the input.

    Args:
            query_chunk_size: Size of query chunks.
            key_chunk_size: Size of key chunks.
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            q_segment_ids: tp.Optional query segment ids array of shape (batch, q_len).
            kv_segment_ids: tp.Optional key/value segment ids array of shape (batch, kv_len).
            deterministic: Whether to apply dropout.
            attn_dropout: Dropout mask.
            pdrop: Dropout probability.
            causal_block_size: Size of causal blocks.
            dtype: dtype of the computation.
            query_chunk_idx: Index of the query chunk.
            key_chunk_idx: Index of the key chunk.
            sliding_window: Size of sliding window for local attention. Can be int or tuple (left_window, right_window).
            attention_sink_size: Number of initial tokens to always attend to (attention sink).

    Returns:
            Attention bias for the chunk.
    """
    query_offset = query_chunk_idx * query_chunk_size
    key_offset = key_chunk_idx * key_chunk_size
    chunk_bias = jnp.zeros((1, 1, 1, 1), dtype=dtype)
    neg_inf = jnp.array(-jnp.inf, dtype=dtype)
    zero = jnp.array(0.0, dtype=dtype)
    if bias is not None:
        chunk_bias = lax.dynamic_slice(
            bias,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *bias.shape[:2],
                min(bias.shape[-2], query_chunk_size),
                min(bias.shape[-1], key_chunk_size),
            ),
        )

    if q_segment_ids is not None and kv_segment_ids is not None:
        q_seg_chunk = lax.dynamic_slice(
            q_segment_ids,
            start_indices=(0, query_offset),
            slice_sizes=(q_segment_ids.shape[0], query_chunk_size),
        )
        kv_seg_chunk = lax.dynamic_slice(
            kv_segment_ids,
            start_indices=(0, key_offset),
            slice_sizes=(kv_segment_ids.shape[0], key_chunk_size),
        )

        segment_mismatch_mask = ~jnp.equal(q_seg_chunk[:, :, None], kv_seg_chunk[:, None, :])

        q_or_kv_is_zero = (q_seg_chunk[:, :, None] == 0) | (kv_seg_chunk[:, None, :] == 0)

        segment_ids_mask = segment_mismatch_mask | q_or_kv_is_zero

        segment_ids_mask = segment_ids_mask[:, None]

        segment_ids_bias = jnp.where(segment_ids_mask, neg_inf, zero)

        chunk_bias = chunk_bias + segment_ids_bias

    if causal_block_size is not None:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        query_idx += query_offset
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        key_idx += key_offset

        causal_mask_value = jnp.where(key_idx > query_idx, neg_inf, zero)

        chunk_bias = chunk_bias + causal_mask_value.reshape(1, 1, *causal_mask_value.shape)

    if sliding_window is not None:
        query_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(query_chunk_size, 1), dimension=0)
        query_idx += query_offset
        key_idx = lax.broadcasted_iota(dtype=jnp.int32, shape=(1, key_chunk_size), dimension=1)
        key_idx += key_offset

        if isinstance(sliding_window, tuple):
            left_window, right_window = sliding_window
        else:
            left_window = right_window = sliding_window

        pos_diff = query_idx - key_idx
        window_mask = (pos_diff >= -right_window) & (pos_diff <= left_window)

        if attention_sink_size > 0:
            sink_mask = key_idx < attention_sink_size
            window_mask = window_mask | sink_mask

        window_mask_value = jnp.where(~window_mask, neg_inf, zero)

        chunk_bias = chunk_bias + window_mask_value.reshape(1, 1, *window_mask_value.shape)

    if not deterministic and pdrop > 0.0:
        attn_dropout_slice = lax.dynamic_slice(
            attn_dropout,
            start_indices=(0, 0, query_offset, key_offset),
            slice_sizes=(
                *attn_dropout.shape[:2],
                min(attn_dropout.shape[-2], query_chunk_size),
                min(attn_dropout.shape[-1], key_chunk_size),
            ),
        )
        chunk_bias = chunk_bias + jnp.where(attn_dropout_slice, neg_inf, zero)
    return chunk_bias.astype(dtype)


def below_or_on_diag(r: int, r_blk_size: int, c: int, c_blk_size: int, causal_block_size: int):
    """Checks if the element at (r, c) is below or on the diagonal.

    Args:
            r: Row index.
            r_blk_size: Block size of the row.
            c: Column index.
            c_blk_size: Block size of the column.
            causal_block_size: Size of causal blocks.

    Returns:
            True if the element is below or on the diagonal, False otherwise.
    """
    causal_block_size_q = max(causal_block_size, r_blk_size)
    causal_block_size_k = max(causal_block_size, c_blk_size)
    r = jax.lax.div(r, causal_block_size_q // r_blk_size)
    c = jax.lax.div(c, causal_block_size_k // c_blk_size)
    return ((r + 1) * causal_block_size_q - 1) > (c * causal_block_size_k)
