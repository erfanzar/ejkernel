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
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import FwdParams


def create_attention_mask(
    batch_size: int,
    q_len: int,
    kv_len: int,
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    sliding_window: tuple[int, int] | None = None,
    num_sinks: int = 0,
) -> Float[Array, "batch q_len 1 kv_len"]:
    """Creates a comprehensive attention mask with ragged sequences, sliding window, and sinks.

    Args:
        batch_size: Batch size
        q_len: Query sequence length
        kv_len: Key/value sequence length
        sequence_start: Start indices for each sequence
        sequence_end: End indices for each sequence
        sliding_window: Optional (left, right) window size for local attention
        num_sinks: Number of attention sink tokens (always attendable)

    Returns:
        Boolean mask of shape [batch, q_len, 1, kv_len]
    """

    kv_positions = jnp.arange(kv_len)[None, None, :]

    start_expanded = sequence_start[:, None, None]
    end_expanded = sequence_end[:, None, None]

    ragged_mask = (kv_positions >= start_expanded) & (kv_positions < end_expanded)

    if sliding_window is not None:
        window_left, window_right = sliding_window

        if q_len == 1:
            q_positions = end_expanded - 1

            window_mask = (kv_positions >= q_positions - window_left) & (kv_positions <= q_positions + window_right)

            ragged_mask = ragged_mask & window_mask
        else:
            q_positions = jnp.arange(q_len)[None, :, None] + start_expanded
            window_mask = (kv_positions >= q_positions - window_left) & (kv_positions <= q_positions + window_right)
            ragged_mask = ragged_mask[:, None, :] & window_mask

    if num_sinks > 0:
        sink_positions = jnp.arange(kv_len)[None, None, :]

        sink_mask = (sink_positions >= start_expanded) & (sink_positions < start_expanded + num_sinks)
        ragged_mask = ragged_mask | sink_mask

    if ragged_mask.ndim == 3:
        ragged_mask = ragged_mask[:, None, None, :]
    elif ragged_mask.ndim == 4:
        ragged_mask = ragged_mask[:, :, None, :]

    return ragged_mask


def apply_logits_soft_cap(scores: Float[Array, "... seq_len"], logits_soft_cap: float) -> Float[Array, "... seq_len"]:
    """Applies soft capping to attention logits.

    Args:
        scores: Attention scores
        logits_soft_cap: Soft capping value

    Returns:
        Soft-capped scores
    """
    return jnp.tanh(scores / logits_soft_cap) * logits_soft_cap


def apply_attention_sinks_block(
    scores: Float[Array, "batch q_len heads block_size"],
    sink_scores: Float[Array, "heads num_sinks"] | None = None,
    num_sinks: int = 0,
    block_offset: int = 0,
) -> Float[Array, "batch q_len heads block_size"]:
    """Applies attention sink biases to scores for a specific block.

    Args:
        scores: Attention scores for this block [B, Q, H, block_size]
        sink_scores: Optional learned biases for sink tokens [H, num_sinks] or [num_sinks]
        num_sinks: Number of sink tokens
        block_offset: Offset of this block in the full sequence

    Returns:
        Scores with sink biases applied if this block contains sinks
    """
    if num_sinks == 0 or sink_scores is None:
        return scores

    _batch_size, _q_len, heads, block_size_val = scores.shape

    if sink_scores.ndim == 1:
        sink_scores = jnp.broadcast_to(sink_scores[None, :], (heads, num_sinks))

    block_positions = jnp.arange(block_size_val) + block_offset

    is_sink_position = block_positions < num_sinks
    sink_indices = jnp.minimum(block_positions, num_sinks - 1)

    block_sink_biases = sink_scores[:, sink_indices]

    block_sink_biases = jnp.where(is_sink_position[None, :], block_sink_biases, 0.0)

    block_sink_biases = block_sink_biases[None, None, :, :]

    return scores + block_sink_biases


def flash_attention_block(
    carry: tuple[Array, Array, Array],
    block_inputs: tuple[Array, Array, Array, Array, int],
    softmax_scale: float,
    logits_soft_cap: float | None = None,
    sink_scores: Array | None = None,
    num_sinks: int = 0,
) -> tuple[tuple[Array, Array, Array], None]:
    """Enhanced flash attention block with soft cap and sinks.

    Args:
        carry: Tuple of (output, max_logits, normalizer)
        block_inputs: Tuple of (queries, keys_block, values_block, mask_block, block_offset)
        softmax_scale: Scaling factor for attention
        logits_soft_cap: Optional soft capping value
        sink_scores: Optional attention sink biases
        num_sinks: Number of sink tokens

    Returns:
        Updated carry tuple
    """
    o_prev, m_prev, l_prev = carry
    q, k_block, v_block, mask_block, block_offset = block_inputs

    _batch_size, _q_len, q_heads, _head_dim = q.shape
    _, _block_size, kv_heads, _ = k_block.shape

    if kv_heads < q_heads:
        assert q_heads % kv_heads == 0, f"Query heads {q_heads} must be divisible by KV heads {kv_heads}"
        repeat_factor = q_heads // kv_heads
        k_block = jnp.repeat(k_block, repeat_factor, axis=2)
        v_block = jnp.repeat(v_block, repeat_factor, axis=2)

    scores = jnp.einsum("...qhd,...khd->...qhk", q * softmax_scale, k_block)

    if logits_soft_cap is not None:
        scores = apply_logits_soft_cap(scores, logits_soft_cap)

    if sink_scores is not None and num_sinks > 0:
        scores = apply_attention_sinks_block(scores, sink_scores, num_sinks, block_offset)

    mask_expanded = jnp.broadcast_to(mask_block, scores.shape)

    scores = jnp.where(mask_expanded, scores, -jnp.inf)

    m_curr = jnp.max(scores, axis=-1, keepdims=True)
    m_new = jnp.maximum(m_prev, m_curr)

    exp_scores = jnp.exp(scores - m_new)
    exp_scores = jnp.where(mask_expanded, exp_scores, 0.0)

    l_curr = jnp.sum(exp_scores, axis=-1, keepdims=True)
    correction_prev = jnp.exp(m_prev - m_new)
    l_new = correction_prev * l_prev + l_curr

    l_new_safe = jnp.where(l_new == 0, 1.0, l_new)

    attn_weights = exp_scores / l_new_safe
    o_curr = jnp.einsum("...qhk,...khd->...qhd", attn_weights, v_block)

    o_new = (correction_prev * l_prev * o_prev + l_curr * o_curr) / l_new_safe

    o_new = o_new.astype(o_prev.dtype)
    m_new = m_new.astype(m_prev.dtype)
    l_new = l_new.astype(l_prev.dtype)

    return (o_new, m_new, l_new), None


def ragged_flash_attention_xla(
    query: Float[Array, "batch q_len num_heads head_dim"],
    key: Float[Array, "batch kv_len num_heads head_dim"],
    value: Float[Array, "batch kv_len num_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch q_len num_heads head_dim"]:
    """Enhanced XLA-compatible ragged flash attention with sliding window, soft cap, and sinks.

    Args:
        query: Query tensor [B, Q, H, D]
        key: Key tensor [B, K, H, D]
        value: Value tensor [B, K, H, D]
        sequence_start: Start indices for each sequence
        sequence_end: End indices for each sequence
        softmax_scale: Optional scaling factor for attention
        block_size: Size of blocks for chunked computation
        sliding_window: Optional (left, right) window for local attention
        logits_soft_cap: Optional soft capping for logits
        softmax_aux: Optional attention sink biases [H, num_sinks] or [num_sinks]

    Returns:
        Attention output [B, Q, H, D]
    """
    batch_size, q_len, num_heads, head_dim = query.shape
    _, kv_len, kv_heads, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim)

    num_sinks = 0
    sink_scores = None
    if softmax_aux is not None:
        if softmax_aux.ndim == 1:
            num_sinks = softmax_aux.shape[0]
        elif softmax_aux.ndim == 2:
            num_sinks = softmax_aux.shape[-1]
        sink_scores = softmax_aux

    mask = create_attention_mask(
        batch_size, q_len, kv_len, sequence_start, sequence_end, sliding_window=sliding_window, num_sinks=num_sinks
    )

    num_blocks = (kv_len + block_size - 1) // block_size

    output_init = jnp.zeros_like(query, dtype=query.dtype)
    max_logits_init = jnp.full((batch_size, q_len, num_heads, 1), -jnp.inf, dtype=query.dtype)
    normalizer_init = jnp.zeros((batch_size, q_len, num_heads, 1), dtype=query.dtype)

    pad_len = num_blocks * block_size - kv_len
    if pad_len > 0:
        key = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")
        value = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode="constant")

        if mask.ndim == 4:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, pad_len)), mode="constant")
        elif mask.ndim == 5:
            mask = jnp.pad(mask, ((0, 0), (0, 0), (0, 0), (0, 0), (0, pad_len)), mode="constant")

    key_blocks = key.reshape(batch_size, num_blocks, block_size, kv_heads, head_dim)
    value_blocks = value.reshape(batch_size, num_blocks, block_size, kv_heads, head_dim)

    if mask.ndim == 4:
        mask_blocks = mask.reshape(batch_size, q_len, 1, num_blocks, block_size)
    else:
        mask_blocks = mask.reshape(batch_size, q_len, mask.shape[2], num_blocks, block_size)
    mask_blocks = jnp.transpose(mask_blocks, (0, 3, 1, 2, 4))

    def scan_fn(carry, inputs):
        (block_idx,) = inputs
        k_block = key_blocks[:, block_idx]
        v_block = value_blocks[:, block_idx]
        m_block = mask_blocks[:, block_idx]

        block_offset = block_idx * block_size

        return flash_attention_block(
            carry,
            (query, k_block, v_block, m_block, block_offset),
            softmax_scale,
            logits_soft_cap=logits_soft_cap,
            sink_scores=sink_scores,
            num_sinks=num_sinks,
        )

    (output, _, _), _ = lax.scan(scan_fn, (output_init, max_logits_init, normalizer_init), (jnp.arange(num_blocks),))

    return output


def ragged_decode_mqa_xla(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Enhanced XLA-compatible ragged MQA decoding.

    Args:
        query: Query tensor [B, H_q, D]
        key: Key tensor [B, S, H_kv, D]
        value: Value tensor [B, S, H_kv, D]
        sequence_start: Start indices for each sequence
        sequence_end: End indices for each sequence
        softmax_scale: Optional scaling factor
        block_size: Block size for computation
        sliding_window: Optional sliding window parameters
        logits_soft_cap: Optional soft capping for logits
        softmax_aux: Optional attention sink biases

    Returns:
        Output tensor [B, H_q, D]
    """
    batch_size, num_heads_q, head_dim = query.shape
    _, _seq_len, num_heads_kv, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim)

    group_size = num_heads_q // num_heads_kv
    query = query.reshape(batch_size, num_heads_kv, group_size, head_dim)

    query = jnp.transpose(query, (1, 0, 2, 3))
    key = jnp.transpose(key, (2, 0, 1, 3))
    value = jnp.transpose(value, (2, 0, 1, 3))

    def process_kv_head(q_group, k_head, v_head):
        k_head = k_head[:, :, None, :]
        v_head = v_head[:, :, None, :]
        q_group = q_group[:, None, :, :]

        output = ragged_flash_attention_xla(
            q_group,
            k_head,
            v_head,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=fwd_params.kv_blocksize,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )

        return output[:, 0, :, :]

    outputs = jax.vmap(process_kv_head)(query, key, value)

    outputs = jnp.transpose(outputs, (1, 0, 2, 3))
    return outputs.reshape(batch_size, num_heads_q, head_dim)


@ejit(static_argnames=["block_size", "softmax_scale", "logits_soft_cap", "sliding_window"])
def inner_decode_xla(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "..."] | None = None,
) -> chex.Array:
    """Enhanced JIT-compiled XLA implementation of ragged MQA Flash Attention.

    Args:
        query: Query tensor, optionally with leading singleton dimension
        key: Key tensor [B, S, H_kv, D]
        value: Value tensor [B, S, H_kv, D]
        sequence_start: Sequence start indices
        sequence_end: Sequence end indices
        softmax_scale: Scaling factor for attention logits
        block_size: Block size for attention computation
        sliding_window: Optional (left, right) window for local attention
        logits_soft_cap: Optional soft capping for logits (e.g., 50.0)
        softmax_aux: Optional attention sink biases [H, num_sinks] or [num_sinks]
                     First few tokens become "attention sinks" with learnable biases

    Returns:
        Output tensor with same batch/head structure as query

    Examples:

        output = inner_decode_xla(query, key, value, start, end)


        output = inner_decode_xla(
            query, key, value, start, end,
            sliding_window=(128, 0)
        )


        output = inner_decode_xla(
            query, key, value, start, end,
            logits_soft_cap=50.0
        )


        sink_biases = jnp.ones(4) * 0.1
        output = inner_decode_xla(
            query, key, value, start, end,
            softmax_aux=sink_biases
        )
    """
    batch_size = query.shape[0]
    num_heads_q = query.shape[-2]
    head_dim = query.shape[-1]

    out_shape = (batch_size, 1, num_heads_q, head_dim)
    if query.ndim == 3:
        query = jnp.expand_dims(query, 1)
        out_shape = (batch_size, num_heads_q, head_dim)

    if query.shape[1] == 1:
        query = query[:, 0]
        output = ragged_decode_mqa_xla(
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=block_size,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )
    else:
        _, _seq_len_q, _, _ = query.shape
        _, _seq_len_kv, num_heads_kv, _ = key.shape

        if num_heads_kv != num_heads_q:
            repeat_factor = num_heads_q // num_heads_kv
            key = jnp.repeat(key, repeat_factor, axis=2)
            value = jnp.repeat(value, repeat_factor, axis=2)

        output = ragged_flash_attention_xla(
            query,
            key,
            value,
            sequence_start,
            sequence_end,
            softmax_scale=softmax_scale,
            block_size=block_size,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=softmax_aux,
        )

    return jnp.reshape(output, out_shape)
