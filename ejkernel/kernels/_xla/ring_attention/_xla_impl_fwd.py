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


from functools import partial

import chex
import jax
import jax.lax as lax
from einops import rearrange
from jax import numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray

from ._utils import _chunk_attention_bias, below_or_on_diag


def _blockwise_attention_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    carry,
    q_chunk_idx_start: int,
    k_chunk_idx_start: int,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
    softmax_scale: float | None,
    causal_block_size: int | None,
    query_chunk_size: int,
    key_chunk_size: int,
    deterministic: bool,
    dropout_rng: PRNGKeyArray | None,
    pdrop: float,
    dtype: DTypeLike,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """Forward pass for blockwise attention.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            carry: Tuple of intermediate values from the previous iteration.
            q_chunk_idx_start: Start index of the query chunk.
            k_chunk_idx_start: Start index of the key chunk.
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            q_segment_ids: tp.Optional query segment ids array of shape (batch, q_len).
            kv_segment_ids: tp.Optional key/value segment ids array of shape (batch, kv_len).
            softmax_scale: softmax_scale for softmax or depth ** -0.5.
            causal_block_size: Size of causal blocks.
            query_chunk_size: Size of query chunks.
            key_chunk_size: Size of key chunks.
            deterministic: Whether to apply dropout.
            dropout_rng: PRNG key for dropout.
            pdrop: Dropout probability.
            dtype: dtype of the computation.
            policy: Checkpoint policy.
            precision: Precision of the computation.
            prevent_cse: Whether to prevent common subexpression elimination.
            sliding_window: Size of sliding window for local attention. Can be int or tuple (left_window, right_window).
            logits_soft_cap: Soft cap value for logits to prevent overflow.
            attention_sink_size: Number of initial tokens to always attend to (attention sink).
            causal: If True, applies causal masking.

    Returns:
            A tuple containing the numerator, denominator, and max score arrays.
    """
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

    numerator, denominator, max_score = carry
    numerator = numerator.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    numerator = jnp.moveaxis(numerator, 1, 0)
    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))

    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    softmax_scale = (
        jnp.sqrt(query.shape[-1]).astype(jnp.float32) if softmax_scale is None else jnp.float32(1 / softmax_scale)
    )
    if not deterministic and pdrop > 0.0:
        attn_dropout_rng, dropout_rng = jax.random.split(dropout_rng)
        attn_dropout = jax.random.bernoulli(attn_dropout_rng, pdrop, (batch, num_heads, q_len, kv_len))
    else:
        attn_dropout = None
    _chunk_bias_fn = partial(
        _chunk_attention_bias,
        query_chunk_size,
        key_chunk_size,
        bias,
        q_segment_ids,
        kv_segment_ids,
        deterministic,
        attn_dropout,
        pdrop,
        causal_block_size if causal else None,
        dtype,
        sliding_window=sliding_window,
        attention_sink_size=attention_sink_size,
    )

    def scan_attention(_, scan):
        q_chunk, numerator_chunk, denominator_chunk, max_score_chunk, q_chunk_idx = scan

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan

            numerator_chunk, denominator_chunk, prev_max_score_chunk = carry

            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / softmax_scale

            if logits_soft_cap is not None:
                attn_weights = jnp.tanh(attn_weights / logits_soft_cap) * logits_soft_cap

            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk

            valid = jnp.isfinite(attn_weights)

            masked_logits = jnp.where(valid, attn_weights, -jnp.inf)

            if softmax_aux is not None:
                if softmax_aux.ndim == 1:
                    sinks = softmax_aux.reshape(1, 1, 1, -1)
                    sinks = jnp.broadcast_to(sinks, (batch, num_heads, 1, softmax_aux.shape[0]))
                elif softmax_aux.ndim == 2:
                    sinks = softmax_aux.reshape(1, num_heads, 1, -1)
                    sinks = jnp.broadcast_to(sinks, (batch, num_heads, 1, softmax_aux.shape[-1]))
                else:
                    raise ValueError(f"softmax_aux must be 1D or 2D, got {softmax_aux.ndim}D")

                sinks = jnp.broadcast_to(sinks, (batch, num_heads, query_chunk_size, sinks.shape[-1]))

                combined_weights = jnp.concatenate([attn_weights, sinks], axis=-1)

                max_score_chunk = jnp.maximum(prev_max_score_chunk, jnp.max(combined_weights, axis=-1))
                max_score_chunk = lax.stop_gradient(max_score_chunk)
                combined_exp_weights = jnp.exp(combined_weights - max_score_chunk[..., None]).astype(jnp.float32)

                exp_weights = combined_exp_weights[..., : attn_weights.shape[-1]]
                exp_values = jnp.einsum(
                    "bhqk,bkhd->bqhd", exp_weights, value_chunk.astype(jnp.float32), precision=precision
                )

                corr_raw = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_raw = jnp.where(jnp.isfinite(max_score_chunk), corr_raw, jnp.array(1.0, corr_raw.dtype))
                correction = rearrange(corr_raw, "b h query -> b query h")[..., None]
                numerator_chunk = numerator_chunk * correction + exp_values
                corr_denom = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_denom = jnp.where(
                    jnp.isfinite(max_score_chunk),
                    corr_denom,
                    jnp.array(1.0, denominator_chunk.dtype),
                )
                denominator_chunk = denominator_chunk * corr_denom + combined_exp_weights.sum(axis=-1)
            else:
                local_max = jnp.max(masked_logits, axis=-1)
                max_score_chunk = jnp.maximum(prev_max_score_chunk, local_max)
                max_score_chunk = lax.stop_gradient(max_score_chunk)
                exp_weights = jnp.where(valid, jnp.exp(attn_weights - max_score_chunk[..., None]), 0.0).astype(
                    jnp.float32
                )
                exp_values = jnp.einsum(
                    "bhqk,bkhd->bqhd", exp_weights, value_chunk.astype(jnp.float32), precision=precision
                )
                corr_raw = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_raw = jnp.where(jnp.isfinite(max_score_chunk), corr_raw, jnp.array(1.0, corr_raw.dtype))
                correction = rearrange(corr_raw, "b h query -> b query h")[..., None]
                numerator_chunk = numerator_chunk * correction + exp_values
                corr_denom = jnp.exp(prev_max_score_chunk - max_score_chunk)
                corr_denom = jnp.where(
                    jnp.isfinite(max_score_chunk),
                    corr_denom,
                    jnp.array(1.0, denominator_chunk.dtype),
                )
                denominator_chunk = denominator_chunk * corr_denom + exp_weights.sum(axis=-1)

            return (
                numerator_chunk,
                denominator_chunk,
                max_score_chunk,
            ), None

        def skip_upper_half(carry, args):
            _key_chunk, _value_chunk, k_chunk_idx = args
            should_run = jnp.array(True)
            if causal_block_size is not None:
                should_run = below_or_on_diag(
                    q_chunk_idx_start + q_chunk_idx,
                    query_chunk_size,
                    k_chunk_idx_start + k_chunk_idx,
                    key_chunk_size,
                    causal_block_size,
                )
            return jax.lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (carry, None),
                carry,
                args,
            )

        (numerator_chunk, denominator_chunk, max_score_chunk), _ = lax.scan(
            skip_upper_half,
            init=(numerator_chunk, denominator_chunk, max_score_chunk),
            xs=(key, value, jnp.arange(0, num_kv)),
        )
        denom = rearrange(denominator_chunk, "b h query -> b query h")[..., None]

        output_chunk = jnp.where(denom > 0, numerator_chunk / denom, 0.0).astype(dtype)
        return (), (output_chunk, numerator_chunk, denominator_chunk, max_score_chunk)

    _, (_, numerator, denominator, max_score) = lax.scan(
        scan_attention,
        init=(),
        xs=(query, numerator, denominator, max_score, jnp.arange(0, num_q)),
    )

    numerator = jnp.moveaxis(numerator, 1, 0)
    numerator = numerator.reshape((batch, q_len, num_heads, dim_per_head))
    denominator, max_score = map(lambda x: rearrange(x, "n b h c -> b h n c"), (denominator, max_score))
    denominator = denominator.reshape((batch, num_heads, q_len))
    max_score = max_score.reshape((batch, num_heads, q_len))

    return numerator, denominator, max_score


def _ring_attention_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    q_segment_ids: chex.Array | None,
    kv_segment_ids: chex.Array | None,
    softmax_aux: chex.Array | None,
    axis_name: str | None,
    float32_logits: bool,
    softmax_scale: float | None,
    query_chunk_size: int,
    key_chunk_size: int,
    causal_block_size: int | None,
    deterministic: bool,
    dropout_rng: PRNGKeyArray | None,
    pdrop: float,
    dtype: DTypeLike,
    policy,
    precision: lax.PrecisionLike,
    prevent_cse: bool,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """Forward pass for ring attention.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            q_segment_ids: tp.Optional query segment ids array of shape (batch, q_len).
            kv_segment_ids: tp.Optional key/value segment ids array of shape (batch, kv_len).
            axis_name: Name of the axis to ppermute over.
            float32_logits: Whether to compute logits in float32.
            softmax_scale: softmax_scale for softmax or depth ** -0.5.
            query_chunk_size: Size of query chunks.
            key_chunk_size: Size of key chunks.
            causal_block_size: Size of causal blocks.
            deterministic: Whether to apply dropout.
            dropout_rng: PRNG key for dropout.
            pdrop: Dropout probability.
            dtype: dtype of the computation.
            policy: Checkpoint policy.
            precision: Precision of the computation.
            prevent_cse: Whether to prevent common subexpression elimination.
            sliding_window: Size of sliding window for local attention. Can be int or tuple (left_window, right_window).
            logits_soft_cap: Soft cap value for logits to prevent overflow.
            attention_sink_size: Number of initial tokens to always attend to (attention sink).
            causal: If True, applies causal masking.

    Returns:
            A tuple containing the output array and a tuple of intermediate values.
    """
    if float32_logits:
        query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    numerator = jnp.zeros((batch, q_len, num_heads, dim_per_head)).astype(jnp.float32)
    denominator = jnp.zeros((batch, num_heads, q_len)).astype(jnp.float32)
    axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
    q_block_size, kv_blocksize = (q_len, kv_len)

    def scan_kv_block(carry, idx):
        prev_max_score, numerator, denominator, key, value = carry
        axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
        q_block_idx = axis_idx
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_block_idx = (axis_idx - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_blocksize // key_chunk_size)
        numerator, denominator, max_score = _blockwise_attention_fwd(
            query,
            key,
            value,
            (numerator, denominator, prev_max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=bias,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            query_chunk_size=query_chunk_size,
            key_chunk_size=key_chunk_size,
            causal_block_size=causal_block_size,
            deterministic=deterministic,
            dropout_rng=dropout_rng,
            pdrop=pdrop,
            dtype=dtype,
            policy=policy,
            precision=precision,
            prevent_cse=prevent_cse,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            attention_sink_size=attention_sink_size,
            causal=causal,
        )
        key, value = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])
            if axis_name is not None
            else x,
            (key, value),
        )
        return (max_score, numerator, denominator, key, value), None

    prev_max_score = jnp.full((batch, num_heads, q_len), -jnp.inf).astype(jnp.float32)
    (max_score, numerator, denominator, _, _), _ = lax.scan(
        scan_kv_block,
        init=(prev_max_score, numerator, denominator, key, value),
        xs=jnp.arange(0, axis_size),
    )
    denom_full = rearrange(denominator, "b h query -> b query h")
    max_full = rearrange(max_score, "b h query -> b query h")
    eps = jnp.finfo(jnp.float32).tiny
    me = max_full + jnp.log(jnp.maximum(denom_full, eps))

    delta = max_full - me
    delta = jnp.where(jnp.isfinite(delta), delta, jnp.array(-jnp.inf, dtype=delta.dtype))
    o_scale = jnp.exp(delta)[..., None]
    output = numerator * o_scale

    return output.astype(value.dtype), (
        output,
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        denominator,
        max_score,
    )
