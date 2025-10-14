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


"""Flash Attention TPU kernel."""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._utils import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    NUM_LANES,
    NUM_SUBLANES,
    TRANS_B_DIM_NUMBERS,
    _fwd_cost_estimate,
    _verify_block,
    below_or_on_diag,
)


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_sizes,
):
    o, l, m = _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        True,
        causal,
        softmax_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
    )
    return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    block_b = q_tile_ref.shape[0]

    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    for batch_idx in range(block_b):
        kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    softmax_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
        l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[batch_idx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    q_seq_idx = pl.program_id(2)
    if causal:
        should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_k):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]
            q = q_tile_ref[batch_idx]
            k = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]

            s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

            if ab_tile_ref is not None:
                ab = ab_tile_ref[(*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))].astype(jnp.float32)
                s += ab

            if softmax_scale != 1.0:
                s *= softmax_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
                q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1)
                kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1, pl.dslice(start_k, block_k)]
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]
            m_next = jnp.maximum(m_prev, m_curr)

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
            p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

            alpha = jnp.exp(m_prev - m_next)

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)

            def l_broadcast(l):
                return pltpu.repeat(l, head_dim_repeats, 1)

            if rem:
                if head_dim_repeats == 0:

                    def l_broadcast(l):
                        return l[:, :head_dim]
                else:
                    raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
            o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    softmax_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]

    assert kv_seq_len == block_k_major == block_k

    q = q_tile_ref[batch_idx]
    k = k_tile_ref[batch_idx]
    s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

    if ab_tile_ref is not None:
        s += ab_tile_ref[batch_idx].astype(jnp.float32)
    if softmax_scale != 1.0:
        s *= softmax_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
        q_segment_ids = q_segment_ids_tile_ref[batch_idx[0]]
        q_segment_ids = pltpu.repeat(q_segment_ids, repeats, axis=1)
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l

    if m_ref is not None:
        m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

    v = v_tile_ref[batch_idx]
    o_tile_ref[batch_idx] = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32).astype(
        o_tile_ref.dtype
    )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major)

            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index

        return (batch_index, head_index, next_q_index, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        softmax_scale=softmax_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        m = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            ab,
            segment_ids,
            causal=causal,
            softmax_scale=softmax_scale,
            kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o
