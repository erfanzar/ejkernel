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


def _blockwise_attention_bwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    g: chex.Array,
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
    """Backward pass for blockwise attention.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            g: Gradient of the output.
            carry: Tuple of intermediate values from the forward pass.
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
            sliding_window: Size of sliding window for local attention.
            logits_soft_cap: Soft cap value for logits to prevent overflow.
            causal: If True, applies causal masking.

    Returns:
            A tuple containing the gradients of the query, key, and value arrays.
    """
    batch, q_len, num_heads, dim_per_head = query.shape
    batch, kv_len, num_heads, dim_per_head = key.shape
    batch, kv_len, num_heads, dim_per_head = value.shape
    num_q = q_len // query_chunk_size
    num_kv = kv_len // key_chunk_size
    dq, dk, dv, output, denominator, max_score = carry
    g = g.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dq = dq.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    dk = dk.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    dv = dv.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    output = output.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    g, dq, dk, dv, output = map(lambda x: jnp.moveaxis(x, 1, 0), (g, dq, dk, dv, output))

    denominator = denominator.reshape((batch, num_heads, num_q, query_chunk_size))
    max_score = max_score.reshape((batch, num_heads, num_q, query_chunk_size))
    denominator, max_score = map(lambda x: rearrange(x, "b h n c -> n b h c"), (denominator, max_score))

    query = query.reshape((batch, num_q, query_chunk_size, num_heads, dim_per_head))
    key = key.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    value = value.reshape((batch, num_kv, key_chunk_size, num_heads, dim_per_head))
    query, key, value = map(lambda x: jnp.moveaxis(x, 1, 0), (query, key, value))

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

    def scan_attention(carry, scan):
        dk, dv = carry
        (
            q_chunk,
            dq_chunk,
            g_chunk,
            output_chunk,
            denominator_chunk,
            max_score_chunk,
            q_chunk_idx,
        ) = scan
        dl_part = jnp.einsum("bqhd,bqhd->bhq", g_chunk, output_chunk)[..., None]

        @partial(jax.checkpoint, prevent_cse=prevent_cse, policy=policy)
        def scan_kv_block(carry, scan):
            k_chunk, value_chunk, k_chunk_idx = scan
            dq_chunk = carry
            attn_weights = jnp.einsum("bqhd,bkhd->bhqk", q_chunk, k_chunk, precision=precision) / softmax_scale

            if logits_soft_cap is not None:
                attn_weights_uncapped = attn_weights
                attn_weights = jnp.tanh(attn_weights / logits_soft_cap) * logits_soft_cap

            bias_chunk = _chunk_bias_fn(q_chunk_idx_start + q_chunk_idx, k_chunk_idx_start + k_chunk_idx)
            attn_weights = attn_weights + bias_chunk
            valid = jnp.isfinite(attn_weights)

            exp_logits = jnp.where(valid, jnp.exp(attn_weights - max_score_chunk[..., None]), 0.0)

            denom = denominator_chunk[..., None]
            exp_weights = jnp.where(denom > 0, exp_logits / denom, 0.0)

            ds = jnp.einsum("bqhd,bkhd->bhqk", g_chunk, value_chunk)
            dl = (ds - dl_part) * exp_weights

            if logits_soft_cap is not None:
                sech2 = 1 - jnp.tanh(attn_weights_uncapped / logits_soft_cap) ** 2
                dl = dl * sech2
            dq_chunk = dq_chunk + jnp.einsum("bhqk,bkhd->bqhd", dl, k_chunk) / softmax_scale
            dk_chunk = jnp.einsum("bqhd,bhqk->bkhd", q_chunk, dl) / softmax_scale
            dv_chunk = jnp.einsum("bhqk,bqhd->bkhd", exp_weights, g_chunk)

            return dq_chunk, (
                dk_chunk,
                dv_chunk,
            )

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
            return lax.cond(
                should_run,
                scan_kv_block,
                lambda carry, args: (
                    carry,
                    (
                        jnp.zeros(
                            (batch, key_chunk_size, num_heads, dim_per_head),
                            dtype=jnp.float32,
                        ),
                        jnp.zeros(
                            (batch, key_chunk_size, num_heads, dim_per_head),
                            dtype=jnp.float32,
                        ),
                    ),
                ),
                carry,
                args,
            )

        dq_chunk, (dk_part, dv_part) = lax.scan(
            skip_upper_half,
            init=dq_chunk,
            xs=(key, value, jnp.arange(0, num_kv)),
        )
        return (dk + dk_part, dv + dv_part), dq_chunk

    (dk, dv), dq = lax.scan(
        scan_attention,
        init=(dk, dv),
        xs=(
            query,
            dq,
            g,
            output,
            denominator,
            max_score,
            jnp.arange(0, num_q),
        ),
    )

    dq, dk, dv = map(lambda x: jnp.moveaxis(x, 1, 0), (dq, dk, dv))
    dq = dq.reshape((batch, q_len, num_heads, dim_per_head))
    dk = dk.reshape((batch, kv_len, num_heads, dim_per_head))
    dv = dv.reshape((batch, kv_len, num_heads, dim_per_head))

    return dq, dk, dv


def _ring_attention_bwd(
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
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    attention_sink_size: int,
    causal: bool,
    res,
    g: chex.Array,
):
    """Backward pass for ring attention.

    Args:
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
            sliding_window: Size of sliding window for local attention.
            logits_soft_cap: Soft cap value for logits to prevent overflow.
            res: Tuple of intermediate values from the forward pass.
            g: Gradient of the output.

    Returns:
            A tuple containing the gradients of the inputs.
    """
    del float32_logits
    (
        output,
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        denominator,
        max_score,
    ) = res
    _, q_len, _, _ = query.shape
    _, kv_len, _, _ = key.shape
    axis_size = lax.psum(1, axis_name) if axis_name is not None else 1
    dq = jnp.zeros_like(query, dtype=jnp.float32)
    dk = jnp.zeros_like(key, dtype=jnp.float32)
    dv = jnp.zeros_like(value, dtype=jnp.float32)
    q_block_size, kv_blocksize = (
        q_len,
        kv_len,
    )

    def scan_kv_block(carry, idx):
        dq, dk, dv, key, value = carry
        axis_idx = lax.axis_index(axis_name) if axis_name is not None else 0
        q_block_idx = axis_idx
        q_chunk_idx_start = q_block_idx * (q_block_size // query_chunk_size)
        k_block_idx = (axis_idx - idx) % axis_size
        k_chunk_idx_start = k_block_idx * (kv_blocksize // key_chunk_size)
        dq, dk, dv = _blockwise_attention_bwd(
            query,
            key,
            value,
            g,
            (dq, dk, dv, output, denominator, max_score),
            q_chunk_idx_start,
            k_chunk_idx_start,
            bias=bias,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            softmax_aux=None,
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
        key, value, dk, dv = map(
            lambda x: lax.ppermute(x, axis_name, perm=[(i, (i + 1) % axis_size) for i in range(axis_size)])
            if axis_name is not None
            else x,
            (key, value, dk, dv),
        )
        return (dq, dk, dv, key, value), None

    (dq, dk, dv, key, value), _ = lax.scan(scan_kv_block, init=(dq, dk, dv, key, value), xs=jnp.arange(0, axis_size))
    dq, dk, dv = dq.astype(query.dtype), dk.astype(key.dtype), dv.astype(value.dtype)

    return dq, dk, dv, None, None, None, None
