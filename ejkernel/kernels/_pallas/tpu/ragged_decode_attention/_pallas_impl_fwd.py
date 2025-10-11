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


import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Float, Int

from ejkernel.callib import ejit


def get_mha_cost_estimate(shape_dtype):
    """Estimates the cost of MHA computation for use with Pallas.

    Args:
        shape_dtype (tuple): Tuple of chex.Array instances (query, key, value, start, end).

    Returns:
        pl.CostEstimate: A rough estimate of compute cost in terms of FLOPs, bytes, etc.
    """
    batch_size, _, num_heads, head_dim = shape_dtype[0].shape
    seq_len = shape_dtype[1].shape[1]

    return pl.CostEstimate(
        flops=batch_size * num_heads * seq_len * (2 * head_dim + seq_len + 2 * head_dim),
        transcendentals=batch_size * num_heads * seq_len,
        bytes_accessed=int(sum(np.prod(s.shape) * s.dtype.itemsize for s in shape_dtype)),
    )


def ragged_flash_attention_kernel(
    s_ref,
    e_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    aux_ref,
    *,
    block_size: int,
    softmax_scale: float,
    sliding_window_left: int,
    sliding_window_right: int,
    logit_soft_cap: float,
    has_softmax_aux: bool,
):
    """Flash Attention kernel for ragged sequences on TPU via Pallas.

    Applies a block-wise attention pattern while respecting per-sequence boundaries.

    Args:
        s_ref: Start indices of each sequence (prefetched).
        e_ref: End indices of each sequence (prefetched).
        q_ref: Query tensor reference.
        k_ref: Key tensor reference.
        v_ref: Value tensor reference.
        o_ref: Output tensor reference (written in-place).
        m_ref: Max logits (intermediate).
        l_ref: Normalization factors (intermediate).
        aux_ref: Auxiliary logits reference (softmax aux/attention sinks).
        block_size (int): Size of blocks to compute attention on.
        softmax_scale (float): Scaling factor applied to attention logits.
        sliding_window_left (int): Left sliding window size (-1 for no limit).
        sliding_window_right (int): Right sliding window size (-1 for no limit).
        logit_soft_cap (float): Soft capping value (0.0 for no capping).
        has_softmax_aux (bool): Whether to use auxiliary logits.
    """
    b, i = pl.program_id(0), pl.program_id(1)

    @pl.when(i == 0)
    def init():
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        o_ref[...] = jnp.zeros_like(o_ref)

    sequence_end = e_ref[b].reshape(1, 1)
    sequence_start = s_ref[b].reshape(1, 1)
    run_index = i * block_size

    @pl.when(run_index < e_ref[b])
    def run():
        q = q_ref[...].astype(jnp.float32)
        k = k_ref[...].astype(jnp.float32)
        v = v_ref[...].astype(jnp.float32)
        m_prev, l_prev = m_ref[...], l_ref[...]

        qk = lax.dot_general(q * softmax_scale, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

        @pl.when(logit_soft_cap > 0.0)
        def apply_soft_cap():
            nonlocal qk
            qk = logit_soft_cap * jnp.tanh(qk / logit_soft_cap)

        @pl.when(has_softmax_aux)
        def add_sinks():
            nonlocal qk
            sinks = aux_ref[...].astype(jnp.float32)

            qk = jnp.concatenate([qk, sinks], axis=-1)

        ranges = (i * block_size) + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)

        mask = (sequence_start <= ranges) & (ranges < sequence_end)

        @pl.when(sliding_window_left >= 0)
        def apply_left_window():
            nonlocal mask
            query_pos = sequence_end - 1
            left_bound = query_pos - sliding_window_left
            mask = mask & (ranges >= left_bound)

        @pl.when(sliding_window_right >= 0)
        def apply_right_window():
            nonlocal mask
            query_pos = sequence_end - 1
            right_bound = query_pos + sliding_window_right
            mask = mask & (ranges <= right_bound)

        sink_cols = lax.broadcasted_iota(jnp.int32, qk.shape, 1) >= block_size
        mask_with_sinks = mask | sink_cols if has_softmax_aux else mask
        qk = jnp.where(mask_with_sinks, qk, jnp.finfo(qk.dtype).min)

        has_seq = jnp.any(mask[..., :block_size], axis=-1)

        has_update = has_seq | has_softmax_aux

        m_raw = qk.max(axis=-1)
        m_curr = jnp.where(has_update, m_raw, m_prev)

        s_curr = jnp.exp(jnp.where(has_update[..., None], qk - m_curr[..., None], -jnp.inf))

        s_curr_seq = s_curr[..., :block_size]
        o_curr_times_l_curr = jnp.dot(s_curr_seq, v)

        sum_s = s_curr.sum(axis=-1)
        has_update2 = (sum_s > 0) | (l_prev > 0)

        m_next = jnp.where(has_update2, jnp.maximum(m_prev, m_curr), m_prev)
        alpha = jnp.where(has_update2, jnp.exp(m_prev - m_next), 0.0)
        beta = jnp.where(has_update2, jnp.exp(m_curr - m_next), 0.0)

        sum_s_b = jax.lax.broadcast_in_dim(sum_s, l_prev.shape, (0,))
        l_next = jnp.where(has_update2, alpha * l_prev + beta * sum_s_b, l_prev)
        l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

        o_new = jnp.where(
            has_update2[..., None],
            (l_prev * alpha)[..., None] * o_ref[...] + beta[..., None] * o_curr_times_l_curr,
            o_ref[...],
        )
        o_ref[...] = (o_new / l_next_safe[..., None]).astype(o_ref.dtype)

        m_ref[...], l_ref[...] = m_next, l_next_safe


def ragged_decode_mqa(
    query_tensor: Float[Array, "batch num_heads head_dim"],
    key_tensor: Float[Array, "batch seq_len head_dim"],
    value_tensor: Float[Array, "batch seq_len head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_aux: Float[Array, "num_heads num_sinks"] | None = None,
    softmax_scale: float | None = 1,
    block_size: int = 256,
    sliding_window_left: int = -1,
    sliding_window_right: int = -1,
    logit_soft_cap: float = 0.0,
    cost_estimate: pl.CostEstimate | None = None,
) -> Float[Array, "batch num_heads head_dim"]:
    """
    Runs ragged MQA decoding using a Flash Attention Pallas kernel.

    Args:
        query_tensor (chex.Array): Query tensor of shape [B, H, D].
        key_tensor (chex.Array): Key tensor of shape [B, S, H, D].
        value_tensor (chex.Array): Value tensor of shape [B, S, H, D].
        sequence_start (chex.Array): Start indices of each sequence [B].
        sequence_end (chex.Array): End indices of each sequence [B].
        softmax_aux (chex.Array | None): Auxiliary logits for attention sinks [B, S].
        softmax_scale (float | None): Optional scale for attention logits.
        block_size (int): Number of tokens processed per block.
        sliding_window_left (int): Left sliding window size (-1 for no limit).
        sliding_window_right (int): Right sliding window size (-1 for no limit).
        logit_soft_cap (float): Soft capping value (0.0 for no capping).
        cost_estimate (pl.CostEstimate | None): Optional cost model for Pallas.

    Returns:
        jax.Array: output array.
    """
    batch_size, num_heads, head_dim = query_tensor.shape

    sequence_start = sequence_start.reshape(batch_size)
    assert sequence_start.shape == (batch_size,)
    assert sequence_start.dtype == jnp.int32

    sequence_end = sequence_end.reshape(batch_size)
    assert sequence_end.shape == (batch_size,)
    assert sequence_end.dtype == jnp.int32

    seq_len = key_tensor.shape[1]

    has_softmax_aux = softmax_aux is not None
    num_sinks = 0
    if softmax_aux is not None:
        if softmax_aux.ndim == 1:
            num_sinks = softmax_aux.shape[0]
            softmax_aux = jnp.broadcast_to(softmax_aux[None, :], (num_heads, num_sinks))
        else:
            num_sinks = softmax_aux.shape[1]
    else:
        softmax_aux = jnp.zeros((num_heads, 1), dtype=jnp.float32)

    out, *_ = pl.pallas_call(
        functools.partial(
            ragged_flash_attention_kernel,
            block_size=block_size,
            softmax_scale=softmax_scale,
            sliding_window_left=sliding_window_left,
            sliding_window_right=sliding_window_right,
            logit_soft_cap=logit_soft_cap,
            has_softmax_aux=has_softmax_aux,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec((None, num_heads, head_dim), lambda b, i, *_: (b, 0, 0)),
                pl.BlockSpec((None, block_size, head_dim), lambda b, i, *_: (b, i, 0)),
                pl.BlockSpec((None, block_size, head_dim), lambda b, i, *_: (b, i, 0)),
                pl.BlockSpec((num_heads, num_sinks if num_sinks > 0 else 1), lambda b, i, *_: (0, 0)),
            ],
            out_specs=[
                pl.BlockSpec((None, num_heads, head_dim), lambda b, i, *_: (b, 0, 0)),
                pl.BlockSpec((None, num_heads), lambda b, i, *_: (b, 0)),
                pl.BlockSpec((None, num_heads), lambda b, i, *_: (b, 0)),
            ],
            grid=(batch_size, (seq_len + block_size - 1) // block_size),
        ),
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel", "arbitrary")),
        out_shape=[
            jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, num_heads), jnp.float32),
            jax.ShapeDtypeStruct((batch_size, num_heads), jnp.float32),
        ],
        cost_estimate=cost_estimate,
    )(sequence_start, sequence_end, query_tensor, key_tensor, value_tensor, softmax_aux)
    return out


@ejit(static_argnames=["block_size", "softmax_scale", "sliding_window", "logit_soft_cap"])
def inner_decode_tpu(
    query_tensor: Float[Array, "batch num_heads head_dim"],
    key_tensor: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value_tensor: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = 1,
    block_size: int = 256,
    sliding_window: tuple[int, int] | None = None,
    logit_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
) -> Float[Array, "batch num_heads head_dim"]:
    """JIT-compiled core implementation of ragged MQA Flash Attention for TPU.

    Args:
        query_tensor (chex.Array): Query tensor, optionally with leading singleton dimension.
        key_tensor (chex.Array): Key tensor of shape [B, S, H, D].
        value_tensor (chex.Array): Value tensor of shape [B, S, H, D].
        sequence_start (chex.Array): Sequence start indices.
        sequence_end (chex.Array): Sequence end indices.
        softmax_scale (float | None): Scaling factor for attention logits.
        block_size (int): Block size to tile attention computation.
        sliding_window (tuple[int, int] | None): Optional (left, right) sliding window sizes.
        logit_soft_cap (float | None): Optional soft capping value for attention logits.
        softmax_aux (chex.Array | None): Optional auxiliary logits for attention sinks.

    Returns:
        chex.Array: Output tensor of shape [B, H, D].
    """
    batch_size = query_tensor.shape[0]
    num_heads_q = query_tensor.shape[-2]
    head_dim = query_tensor.shape[-1]
    _, _, num_heads_kv, _ = key_tensor.shape
    out_shape = (batch_size, 1, num_heads_q, head_dim)
    if query_tensor.ndim == 3:
        query_tensor = jnp.expand_dims(query_tensor, 1)
        out_shape = (batch_size, num_heads_q, head_dim)
    shape_dtype = (query_tensor, key_tensor, value_tensor, sequence_start, sequence_end)
    cost_estimate = get_mha_cost_estimate(shape_dtype)

    sliding_window_left = -1 if sliding_window is None else sliding_window[0]
    sliding_window_right = -1 if sliding_window is None else sliding_window[1]

    logit_soft_cap_val = 0.0 if logit_soft_cap is None else logit_soft_cap

    if softmax_aux is not None:
        if softmax_aux.ndim not in [1, 2]:
            raise ValueError(
                f"softmax_aux must have shape [num_sinks] or [num_heads, num_sinks], got {softmax_aux.shape}"
            )

    query_tensor = query_tensor.reshape(batch_size, num_heads_kv, num_heads_q // num_heads_kv, head_dim)
    key_tensor = jnp.swapaxes(key_tensor, 1, 2)
    value_tensor = jnp.swapaxes(value_tensor, 1, 2)

    o = jax.vmap(
        functools.partial(
            ragged_decode_mqa,
            block_size=block_size,
            cost_estimate=cost_estimate,
            softmax_scale=softmax_scale,
            sliding_window_left=sliding_window_left,
            sliding_window_right=sliding_window_right,
            logit_soft_cap=logit_soft_cap_val,
        ),
        in_axes=(1, 1, 1, None, None, 0)
        if (softmax_aux is not None and softmax_aux.ndim == 2)
        else (1, 1, 1, None, None, None),
        out_axes=1,
    )(query_tensor, key_tensor, value_tensor, sequence_start, sequence_end, softmax_aux)

    return jnp.reshape(o, out_shape)
