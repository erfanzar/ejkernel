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


"""
Efficient Ring Attention Implementation for Single-Device Execution

This module provides an optimized implementation of ring attention,
originally inspired by the work of Liu et al. (2023)
([https://arxiv.org/abs/2310.01889](https://arxiv.org/abs/2310.01889)).
It incorporates the following enhancements:

- Single-Device Focus: Adapted for efficient execution on a single device,
  removing the need for parallel communication primitives.
- Enhanced JIT Compatibility: Streamlined for smoother integration with
  JAX's Just-In-Time (JIT) compilation.
- Performance Optimizations:  Includes code optimizations for improved speed
  and memory usage.

Note: While based on existing implementations, this version offers significant
modifications to enhance its usability and performance in single-device and multi-host
settings.
- also adding softmax softmax_scale option to support custom scales
"""

from functools import partial

import chex
import jax
import jaxtyping
from beartype import beartype
from jax import Array, lax
from jax import numpy as jnp
from jaxtyping import DTypeLike, Float, Int, PRNGKeyArray

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import _ring_flash_attention_bwd_tpu
from ._pallas_impl_fwd import _ring_flash_attention_fwd_tpu
from ._utils import SegmentIds


@partial(
    jax.custom_vjp,
    nondiff_argnums=(4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
)
def _ring_attention(
    query: Array,
    key: Array,
    value: Array,
    attn_bias: Array | None,
    segment_ids: SegmentIds | None,
    softmax_aux: Array | None,
    cache_idx,
    axis_name: str,
    float32_logits,
    query_chunk_size,
    key_chunk_size,
    causal_block_size,
    deterministic: bool,
    dropout_rng: PRNGKeyArray | None,
    pdrop: float,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    attention_sink_size: int,
    policy,
    softmax_scale,
    causal: bool,
) -> chex.Array:
    """
    Computes ring attention using FlashAttention on TPU.
    """
    y, _ = _ring_flash_attention_fwd_tpu(
        query,
        key,
        value,
        attn_bias,
        segment_ids,
        softmax_aux,
        cache_idx,
        axis_name,
        float32_logits,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
        deterministic,
        dropout_rng,
        pdrop,
        sliding_window,
        logits_soft_cap,
        attention_sink_size,
        policy,
        softmax_scale,
        causal,
    )
    return y


_ring_attention.defvjp(_ring_flash_attention_fwd_tpu, _ring_flash_attention_bwd_tpu)


@kernel_registry.register("ring_attention", Platform.PALLAS, Backend.TPU)
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
) -> chex.Array:
    """
    Computes ring attention using FlashAttention on TPU.

    Args:
        query: Query array of shape (batch, q_len, num_heads, dim_per_head).
        key: Key array of shape (batch, kv_len, num_heads, dim_per_head).
        value: Value array of shape (batch, kv_len, num_heads, dim_per_head).
        bias: Optional bias array of shape (batch, num_heads, q_len, kv_len).
        q_segment_ids: Optional query segment ids array of shape (batch, q_len).
        kv_segment_ids: Optional key/value segment ids array of shape (batch, kv_len).
        softmax_aux: Optional attention sink logits of shape [num_heads, num_sinks] or [num_sinks].
        cache_idx: Optional cache index for incremental decoding.
        axis_name: Name of the axis to ppermute over.
        float32_logits: Whether to compute logits in float32.
        softmax_scale: Scale for softmax (default: dim_per_head ** -0.5).
        query_chunk_size: Size of query chunks.
        key_chunk_size: Size of key chunks.
        causal_block_size: Size of causal blocks for block-wise causal masking. If None and causal=True,
            defaults to query_chunk_size for efficient block-level causal attention.
        deterministic: Whether to apply dropout (False = apply dropout).
        dropout_rng: PRNG key for dropout.
        pdrop: Dropout probability.
        sliding_window: Size of sliding window for local attention. Can be int for symmetric
            window or tuple (left_window, right_window) for asymmetric window.
        logits_soft_cap: Soft cap value for logits to prevent overflow (tanh capping).
        attention_sink_size: Number of initial tokens to always attend to (StreamingLLM).
        causal: If True, applies causal masking where each position can only attend to previous positions.
            Uses causal_block_size for efficient blockwise causal computation.
        policy: Checkpoint policy for gradient checkpointing.
        prevent_cse: Whether to prevent common subexpression elimination.

    Returns:
        Output array of shape (batch, q_len, num_heads, dim_per_head).
    """
    del dtype, precision, prevent_cse

    if attention_mask is not None:
        if attention_mask.dtype != jnp.bool_:
            attention_mask = attention_mask.astype(jnp.bool_)
        mask_bias = jnp.where(attention_mask, 0.0, jnp.finfo(jnp.float32).min).astype(jnp.float32)

        if bias is None:
            bias = mask_bias
        else:
            bias = bias + mask_bias

    if q_segment_ids is None and kv_segment_ids is None:
        segment_ids = None
    elif q_segment_ids is not None and kv_segment_ids is None:
        segment_ids = SegmentIds(query=q_segment_ids, kv=q_segment_ids)
    elif q_segment_ids is None and kv_segment_ids is not None:
        segment_ids = SegmentIds(query=kv_segment_ids, kv=kv_segment_ids)
    else:
        segment_ids = SegmentIds(query=q_segment_ids, kv=kv_segment_ids)

    if causal:
        causal_block_size = 1

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    return _ring_attention(
        query,
        key,
        value,
        bias,
        segment_ids,
        softmax_aux,
        cache_idx,
        axis_name,
        float32_logits,
        query_chunk_size,
        key_chunk_size,
        causal_block_size,
        deterministic,
        dropout_rng,
        pdrop,
        sliding_window,
        logits_soft_cap,
        attention_sink_size,
        policy,
        softmax_scale,
        causal,
    )
