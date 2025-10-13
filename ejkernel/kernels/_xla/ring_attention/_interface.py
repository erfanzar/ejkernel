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
import jaxtyping
from beartype import beartype
from jax import numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.callib import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _ring_attention_bwd
from ._xla_impl_fwd import _ring_attention_fwd


@partial(jax.custom_vjp, nondiff_argnums=[7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
def _ring_attention(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None = None,
    q_segment_ids: chex.Array | None = None,
    kv_segment_ids: chex.Array | None = None,
    softmax_aux: chex.Array | None = None,
    axis_name: str | None = None,
    float32_logits: bool = True,
    softmax_scale: float | None = None,
    query_chunk_size: int = 512,
    key_chunk_size: int = 512,
    causal_block_size: int | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    pdrop: float = 0.0,
    dtype: DTypeLike = jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    prevent_cse: bool = True,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
):
    """
    Computes ring attention with blockwise transformers.

    Args:
            query: Query array of shape (batch, q_len, num_heads, dim_per_head).
            key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
            value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
            bias: tp.Optional bias array of shape (batch, num_heads, q_len, kv_len).
            q_segment_ids: tp.Optional query segment ids array of shape (batch, q_len).
                If both q_segment_ids and kv_segment_ids are None, no segment masking is applied.
            kv_segment_ids: tp.Optional key/value segment ids array of shape (batch, kv_len).
                If only one of q_segment_ids or kv_segment_ids is provided, it will be used for both.
            softmax_aux: Optional attention sink logits of shape [num_heads, num_sinks] or [num_sinks].
                These are auxiliary logits that participate in softmax normalization but don't
                contribute to output, allowing the model to absorb probability mass.
            axis_name: Name of the axis to ppermute over.
            float32_logits: Whether to compute logits in float32.
            softmax_scale: softmax_scale for softmax or depth ** -0.5.
            query_chunk_size: Size of query chunks.
            key_chunk_size: Size of key chunks.
            causal_block_size: Size of causal blocks for efficient causal masking. If None and causal=True,
                defaults to query_chunk_size for block-level causal attention.
            deterministic: Whether to apply dropout.
            dropout_rng: PRNG key for dropout.
            pdrop: Dropout probability.
            dtype: dtype of the computation.
            policy: Checkpoint policy.
            precision: Precision of the computation.
            prevent_cse: Whether to prevent common subexpression elimination.
            sliding_window: Size of sliding window for local attention. Can be int for symmetric window
                or tuple (left_window, right_window) for asymmetric window.
            logits_soft_cap: Soft cap value for logits to prevent overflow.
            attention_sink_size: Number of initial tokens to always attend to (StreamingLLM-style attention sink).
            causal: If True, applies causal masking where each position can only attend to previous positions.
                Uses causal_block_size for efficient blockwise causal computation.

    Returns:
            Output array of shape (batch, q_len, num_heads, dim_per_head).
    """

    if q_segment_ids is None and kv_segment_ids is not None:
        q_segment_ids = kv_segment_ids
    elif kv_segment_ids is None and q_segment_ids is not None:
        kv_segment_ids = q_segment_ids

    if causal and causal_block_size is None:
        causal_block_size = query_chunk_size

    y, _ = _ring_attention_fwd(
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        axis_name,
        float32_logits,
        softmax_scale,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
        deterministic,
        dropout_rng,
        pdrop,
        dtype,
        policy,
        precision,
        prevent_cse,
        sliding_window,
        logits_soft_cap,
        attention_sink_size,
        causal,
    )
    return y


_ring_attention.defvjp(_ring_attention_fwd, _ring_attention_bwd)
_ring_attention = ejit(_ring_attention, static_argnums=(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23))


@kernel_registry.register("ring_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    cache_idx=None,
    attention_mask=None,
    axis_name: str | None = None,
    float32_logits: bool = True,
    softmax_scale: float | None = None,
    query_chunk_size: int = 512,
    key_chunk_size: int = 512,
    causal_block_size: int | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    pdrop: float = 0.0,
    dtype: DTypeLike = jnp.float32,
    policy=jax.checkpoint_policies.nothing_saveable,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    prevent_cse: bool = True,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    attention_sink_size: int = 0,
    causal: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    query_chunk_size = min(query_chunk_size, query.shape[1])
    key_chunk_size = min(key_chunk_size, key.shape[1])

    if attention_mask is not None:
        if attention_mask.dtype != jnp.bool_:
            attention_mask = attention_mask.astype(jnp.bool_)
        if attention_mask.ndim == 2:
            attention_mask = attention_mask[None, None, ...]
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, ...]

        neg_inf = jnp.array(-jnp.inf, dtype=dtype)
        zero = jnp.array(0.0, dtype=dtype)
        mask_bias = jnp.where(attention_mask, zero, neg_inf)
        bias = mask_bias if bias is None else (bias + mask_bias)

    return _ring_attention(
        query,
        key,
        value,
        bias,
        q_segment_ids,
        kv_segment_ids,
        softmax_aux,
        axis_name,
        float32_logits,
        softmax_scale,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
        deterministic,
        dropout_rng,
        pdrop,
        dtype,
        policy,
        precision,
        prevent_cse,
        sliding_window,
        logits_soft_cap,
        attention_sink_size,
        causal,
    )
