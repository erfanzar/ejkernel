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


import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float, Int

from ejkernel.callib import ejit


@ejit(static_argnames=["sliding_window", "softmax_scale", "logit_soft_cap"])
def ragged_decode_attention_impl(
    query: Float[Array, "batch num_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    sliding_window: tuple[int, int] | None = None,
    logit_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
) -> Float[Array, "batch num_heads head_dim"]:
    """
    XLA implementation of ragged decode attention using standard JAX operations.

    This function computes attention for a batch of sequences with variable lengths,
    supporting MQA/GQA, sliding windows, logit soft capping, and attention sinks.

    Args:
        query: Query tensor [batch, num_heads, head_dim].
        key: Key tensor [batch, seq_len, num_kv_heads, head_dim].
        value: Value tensor [batch, seq_len, num_kv_heads, head_dim].
        sequence_start: Start indices [batch].
        sequence_end: End indices [batch].
        softmax_scale: Attention score scaling factor.
        sliding_window: Optional (left, right) window sizes.
        logit_soft_cap: Optional soft capping value.
        softmax_aux: Optional attention sink logits.

    Returns:
        Output tensor [batch, num_heads, head_dim].
    """
    batch_size, num_heads, head_dim = query.shape
    _, seq_len, num_kv_heads, _ = key.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim)

    num_groups = num_heads // num_kv_heads
    query = query.reshape(batch_size, num_kv_heads, num_groups, head_dim)

    scores = jnp.einsum("bkhd,bskd->bkhs", query * softmax_scale, key)

    if logit_soft_cap is not None:
        scores = logit_soft_cap * jnp.tanh(scores / logit_soft_cap)

    positions = jnp.arange(seq_len)

    start = sequence_start[:, None, None, None]
    end = sequence_end[:, None, None, None]
    sequence_mask = (positions >= start) & (positions < end)

    if sliding_window is not None:
        left_window, right_window = sliding_window
        query_pos = sequence_end[:, None, None, None] - 1

        left_bound = jnp.where(left_window >= 0, query_pos - left_window, jnp.int32(0))
        right_bound = jnp.where(right_window >= 0, query_pos + right_window, jnp.int32(seq_len - 1))

        window_mask = (positions >= left_bound) & (positions <= right_bound)
        sequence_mask = sequence_mask & window_mask

    scores = jnp.where(sequence_mask, scores, jnp.finfo(scores.dtype).min)

    if softmax_aux is not None:
        if softmax_aux.ndim == 1:
            num_sinks = softmax_aux.shape[0]
            sinks = softmax_aux.reshape(1, 1, 1, num_sinks)
            sinks = jnp.broadcast_to(sinks, (batch_size, num_kv_heads, num_groups, num_sinks))
        else:
            num_sinks = softmax_aux.shape[1]
            sinks = softmax_aux.reshape(1, num_kv_heads, 1, num_sinks)
            sinks = jnp.broadcast_to(sinks, (batch_size, num_kv_heads, num_groups, num_sinks))

        combined_scores = jnp.concatenate([scores, sinks], axis=-1)

        attention_weights = jax.nn.softmax(combined_scores, axis=-1)

        attention_weights = attention_weights[..., :seq_len]
    else:
        attention_weights = jax.nn.softmax(scores, axis=-1)

    output = jnp.einsum("bkhs,bskd->bkhd", attention_weights, value)

    output = output.reshape(batch_size, num_heads, head_dim)

    return output
