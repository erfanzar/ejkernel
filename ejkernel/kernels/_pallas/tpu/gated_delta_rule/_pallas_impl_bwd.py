# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Backward TPU Pallas kernels for Gated Delta Rule (GDR).

Pure Pallas backward with a single reverse-scan kernel that recomputes
all forward intermediates (Neumann series, decay masks) from raw inputs
+ saved per-chunk states, then computes all gradients in one fused pass.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_bwd
from ._pallas_impl_fwd import _chunk_blockspec, _dot, _neumann_inv


def _gdr_bwd_grad_kernel(
    state_pre_ref,
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    d_out_ref,
    d_state_next_ref,
    d_state_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_decay_ref,
):
    """Reverse gradient scan step with full recomputation, all in VMEM.

    Recomputes Neumann series, decay masks, and all forward intermediates
    from raw (q, k, v, beta, decay, state_pre), then computes final
    d_q, d_k, d_v, d_beta, d_decay for a single chunk.
    """
    C = q_ref.shape[2]
    K = q_ref.shape[3]
    V = v_ref.shape[3]
    ones_K = jnp.ones((K, 1), dtype=jnp.float32)
    ones_V = jnp.ones((V, 1), dtype=jnp.float32)
    ones_C = jnp.ones((C, 1), dtype=jnp.float32)

    state_pre = state_pre_ref[0, 0].astype(jnp.float32)
    state_pre = jnp.nan_to_num(state_pre, nan=0.0, posinf=0.0, neginf=0.0)
    q = q_ref[0, 0].astype(jnp.float32)
    k = k_ref[0, 0].astype(jnp.float32)
    v = v_ref[0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, :, 0].astype(jnp.float32)
    decay = decay_ref[0, 0, :, 0].astype(jnp.float32)
    d_out = d_out_ref[0, 0].astype(jnp.float32)
    d_out = jnp.nan_to_num(d_out, nan=0.0, posinf=0.0, neginf=0.0)
    d_state_next = d_state_next_ref[0, 0].astype(jnp.float32)
    d_state_next = jnp.nan_to_num(d_state_next, nan=0.0, posinf=0.0, neginf=0.0)

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)
    upper_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.float32))

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cumsum = _dot(lower_mask, decay)
    g_diff = g_cumsum[:, None] - g_cumsum[None, :]
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn = _neumann_inv(attn_neg, C)
    attn = jnp.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    k_beta_scaled = k_beta * g_cumsum_exp[:, None]
    value_local = _dot(attn, v_beta)
    value_local = jnp.nan_to_num(value_local, nan=0.0, posinf=0.0, neginf=0.0)
    k_cumdecay = _dot(attn, k_beta_scaled)
    k_cumdecay = jnp.nan_to_num(k_cumdecay, nan=0.0, posinf=0.0, neginf=0.0)

    attn_qk_base = _dot(q, k.T)
    attn_qk = attn_qk_base * decay_mask
    attn_qk = jnp.where(lower_mask > 0.0, attn_qk, 0.0)
    attn_qk = jnp.nan_to_num(attn_qk, nan=0.0, posinf=0.0, neginf=0.0)
    q_scaled = q * g_cumsum_exp[:, None]
    v_prime = _dot(k_cumdecay, state_pre)
    v_prime = jnp.nan_to_num(v_prime, nan=0.0, posinf=0.0, neginf=0.0)
    v_new = value_local - v_prime
    v_new = jnp.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
    k_scaled = k * g_diff_state_exp[:, None]

    d_state = d_state_next * g_end_exp
    d_state = jnp.nan_to_num(d_state, nan=0.0, posinf=0.0, neginf=0.0)
    d_g_end = _dot(ones_K.T, _dot(d_state_next * state_pre, ones_V))[0, 0]

    d_k_scaled = _dot(v_new, d_state_next.T)
    d_k_scaled = jnp.nan_to_num(d_k_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    d_v_new = _dot(k_scaled, d_state_next)
    d_v_new = jnp.nan_to_num(d_v_new, nan=0.0, posinf=0.0, neginf=0.0)

    d_attn_qk = _dot(d_out, v_new.T)
    d_attn_qk = jnp.nan_to_num(d_attn_qk, nan=0.0, posinf=0.0, neginf=0.0)
    d_v_new = d_v_new + _dot(attn_qk.T, d_out)
    d_v_new = jnp.nan_to_num(d_v_new, nan=0.0, posinf=0.0, neginf=0.0)

    d_value_local = d_v_new
    d_v_prime = -d_v_new

    d_k_cumdecay = _dot(d_v_prime, state_pre.T)
    d_k_cumdecay = jnp.nan_to_num(d_k_cumdecay, nan=0.0, posinf=0.0, neginf=0.0)
    d_state = d_state + _dot(k_cumdecay.T, d_v_prime)
    d_state = jnp.nan_to_num(d_state, nan=0.0, posinf=0.0, neginf=0.0)

    d_q_scaled = _dot(d_out, state_pre.T)
    d_q_scaled = jnp.nan_to_num(d_q_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    d_state = d_state + _dot(q_scaled.T, d_out)
    d_state = jnp.nan_to_num(d_state, nan=0.0, posinf=0.0, neginf=0.0)

    d_q = d_q_scaled * g_cumsum_exp[:, None]
    d_g_exp = _dot(d_q_scaled * q, ones_K)[:, 0]

    d_k = d_k_scaled * g_diff_state_exp[:, None]
    d_g_diff = _dot(d_k_scaled * k, ones_K)[:, 0]

    d_attn_qk_base = d_attn_qk * decay_mask
    d_attn_qk_base = jnp.where(lower_mask > 0.0, d_attn_qk_base, 0.0)
    d_attn_qk_base = jnp.nan_to_num(d_attn_qk_base, nan=0.0, posinf=0.0, neginf=0.0)
    d_decay_mask_from_qk = d_attn_qk * attn_qk_base

    d_q = d_q + _dot(d_attn_qk_base, k)
    d_k = d_k + _dot(d_attn_qk_base.T, q)

    d_attn = _dot(d_value_local, v_beta.T) + _dot(d_k_cumdecay, k_beta_scaled.T)
    d_attn = jnp.nan_to_num(d_attn, nan=0.0, posinf=0.0, neginf=0.0)
    d_value_beta = _dot(attn.T, d_value_local)
    d_value_beta = jnp.nan_to_num(d_value_beta, nan=0.0, posinf=0.0, neginf=0.0)
    d_key_beta_scaled = _dot(attn.T, d_k_cumdecay)
    d_key_beta_scaled = jnp.nan_to_num(d_key_beta_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    d_key_beta = d_key_beta_scaled * g_cumsum_exp[:, None]
    d_g_exp = d_g_exp + _dot(d_key_beta_scaled * k_beta, ones_K)[:, 0]

    tmp = _dot(attn.T, d_attn)
    d_k_attn = -_dot(tmp, attn.T)
    d_k_attn = d_k_attn * strict_lower
    d_k_attn = jnp.nan_to_num(d_k_attn, nan=0.0, posinf=0.0, neginf=0.0)

    kk = _dot(k_beta, k.T)
    d_kk = d_k_attn * decay_mask
    d_decay_mask = (d_decay_mask_from_qk + d_k_attn * kk) * lower_mask
    d_decay_mask = jnp.nan_to_num(d_decay_mask, nan=0.0, posinf=0.0, neginf=0.0)

    d_key_beta = d_key_beta + _dot(d_kk, k)
    d_k = d_k + _dot(d_kk.T, k_beta)

    d_v = d_value_beta * beta[:, None]
    d_beta = _dot(d_value_beta * v, ones_V)[:, 0]

    d_k = d_k + d_key_beta * beta[:, None]
    d_beta = d_beta + _dot(d_key_beta * k, ones_K)[:, 0]

    d_decay_f = d_decay_mask * decay_mask
    d_decay_f = jnp.nan_to_num(d_decay_f, nan=0.0, posinf=0.0, neginf=0.0)
    d_g_row = _dot(d_decay_f, ones_C)[:, 0]
    d_g_col = _dot(ones_C.T, d_decay_f)[0]
    d_g = d_g_row - d_g_col
    d_g = d_g + d_g_exp * g_cumsum_exp

    d_g_diff_term = d_g_diff * g_diff_state_exp
    d_g_end_total = _dot(d_g_diff_term[None, :], ones_C)[0, 0] + d_g_end * g_end_exp
    d_g = d_g - d_g_diff_term

    d_decay_final = _dot(upper_mask, d_g[:, None])[:, 0] + d_g_end_total

    d_state_ref[0, 0] = d_state.astype(d_state_ref.dtype)
    d_q_ref[0, 0] = d_q.astype(d_q_ref.dtype)
    d_k_ref[0, 0] = d_k.astype(d_k_ref.dtype)
    d_v_ref[0, 0] = d_v.astype(d_v_ref.dtype)
    d_beta_ref[0, 0] = d_beta[:, None].astype(d_beta_ref.dtype)
    d_decay_ref[0, 0] = d_decay_final[:, None].astype(d_decay_ref.dtype)


def _run_bwd_grad_step(
    state_pre,
    q_i,
    k_i,
    v_i,
    beta_i,
    decay_i,
    d_out_i,
    d_state_next,
):
    bsz, num_heads, C, qk_dim = q_i.shape
    v_dim = v_i.shape[-1]
    call = pl.pallas_call(
        _gdr_bwd_grad_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, v_dim)),
                _chunk_blockspec((1, 1, C, 1)),
                _chunk_blockspec((1, 1, C, 1)),
                _chunk_blockspec((1, 1, C, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, v_dim)),
                _chunk_blockspec((1, 1, C, 1)),
                _chunk_blockspec((1, 1, C, 1)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, C, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, C, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, C, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, C, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, C, 1), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(
        state_pre,
        q_i,
        k_i,
        v_i,
        beta_i,
        decay_i,
        d_out_i,
        d_state_next,
    )


def _cast_grad(x, dtype):
    if x is None:
        return None
    return x.astype(dtype) if x.dtype != dtype else x


def _chunk_gdr_bwd(
    chunk_size: int,
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    """Pure Pallas backward for chunked GDR."""
    (
        query,
        key,
        value,
        beta,
        decay,
        state_pre_all,
        _initial_state,
        q_inv_norm,
        k_inv_norm,
        seq_len,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    ) = res
    d_out, d_final_state = g
    input_dtype = query.dtype
    B, H, num_chunks, _C, K_dim = query.shape
    V_dim = value.shape[-1]
    scale = 1.0 / math.sqrt(K_dim)

    if pad_size > 0:
        d_out = jnp.pad(d_out, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
    d_out = d_out.reshape(B, H, num_chunks, chunk_size, V_dim)

    beta_k = beta[:, :, :, :, None]
    decay_k = decay[:, :, :, :, None]

    q_tm = query.transpose(2, 0, 1, 3, 4)
    k_tm = key.transpose(2, 0, 1, 3, 4)
    v_tm = value.transpose(2, 0, 1, 3, 4)
    beta_tm = beta_k.transpose(2, 0, 1, 3, 4)
    decay_tm = decay_k.transpose(2, 0, 1, 3, 4)
    state_pre_tm = state_pre_all.transpose(2, 0, 1, 3, 4)
    d_out_tm = d_out.transpose(2, 0, 1, 3, 4)

    d_final_state = d_final_state.astype(jnp.float32)

    def grad_step(d_state_next, inputs):
        sp_i, q_i, k_i, v_i, b_i, dc_i, do_i = inputs
        d_state_i, d_q_i, d_k_i, d_v_i, d_beta_i, d_decay_i = _run_bwd_grad_step(
            sp_i,
            q_i,
            k_i,
            v_i,
            b_i,
            dc_i,
            do_i,
            d_state_next,
        )
        return d_state_i, (d_q_i, d_k_i, d_v_i, d_beta_i, d_decay_i)

    d_initial_state, grads_tm = lax.scan(
        grad_step,
        d_final_state,
        (state_pre_tm, q_tm, k_tm, v_tm, beta_tm, decay_tm, d_out_tm),
        reverse=True,
    )
    d_q_tm, d_k_tm, d_v_tm, d_beta_tm, d_decay_tm = grads_tm

    total_len = seq_len + pad_size
    d_query = d_q_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_key = d_k_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
    d_value = d_v_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, V_dim)[:, :, :seq_len, :]
    d_beta = d_beta_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, 1)[:, :, :seq_len, 0]
    d_decay = d_decay_tm.transpose(1, 2, 0, 3, 4).reshape(B, H, total_len, 1)[:, :, :seq_len, 0]

    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :] / scale
        k_norm = key.reshape(B, H, total_len, K_dim)[:, :, :seq_len, :]
        d_query = _l2norm_bwd(d_query, q_norm, q_inv_norm.astype(jnp.float32))
        d_key = _l2norm_bwd(d_key, k_norm, k_inv_norm.astype(jnp.float32))

    if decay_was_none:
        d_decay = None
    if initial_state_was_none:
        d_initial_state = None

    return (
        _cast_grad(d_query, input_dtype),
        _cast_grad(d_key, input_dtype),
        _cast_grad(d_value, input_dtype),
        _cast_grad(d_beta, input_dtype),
        _cast_grad(d_decay, input_dtype),
        _cast_grad(d_initial_state, input_dtype),
    )


def _gdr_single_step_bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_prev_ref,
    d_out_ref,
    d_state_next_ref,
    d_q_ref,
    d_k_ref,
    d_v_ref,
    d_beta_ref,
    d_decay_ref,
    d_state_ref,
):
    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0, 0].astype(jnp.float32)[0]
    decay_t = decay_ref[0, 0, 0].astype(jnp.float32)[0]
    state_prev = state_prev_ref[0, 0].astype(jnp.float32)
    d_out_t = d_out_ref[0, 0, 0].astype(jnp.float32)
    d_state_next = d_state_next_ref[0, 0].astype(jnp.float32)

    g_exp = jnp.exp(jnp.clip(decay_t, -20.0, 20.0))
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta_raw = v_t - kv_mem
    delta = delta_raw * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]

    d_s = d_state_next + q_t[:, None] * d_out_t[None, :]
    d_q = jnp.sum(state * d_out_t[None, :], axis=-1)
    d_k = jnp.sum(d_s * delta[None, :], axis=-1)
    d_delta = jnp.sum(d_s * k_t[:, None], axis=0)
    d_beta = jnp.sum(d_delta * delta_raw)
    d_v = d_delta * beta_t
    d_kv_mem = -d_delta * beta_t
    d_state_decayed = d_s + k_t[:, None] * d_kv_mem[None, :]
    d_k = d_k + jnp.sum(state_decayed * d_kv_mem[None, :], axis=-1)
    d_state = d_state_decayed * g_exp
    d_decay = jnp.sum(d_state_decayed * state_prev) * g_exp

    d_q_ref[0, 0, 0] = d_q
    d_k_ref[0, 0, 0] = d_k
    d_v_ref[0, 0, 0] = d_v
    d_beta_ref[0, 0, 0, 0] = d_beta
    d_decay_ref[0, 0, 0, 0] = d_decay
    d_state_ref[0, 0] = d_state


def _run_single_step_backward(
    query,
    key,
    value,
    beta,
    decay,
    recurrent_state,
    d_out,
    d_state_next,
):
    bsz, num_heads, _, qk_dim = query.shape
    v_dim = value.shape[-1]
    beta = beta[..., None]
    decay = decay[..., None]
    call = pl.pallas_call(
        _gdr_single_step_bwd_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, 1, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, qk_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, v_dim), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, 1, 1), jnp.float32),
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(query, key, value, beta, decay, recurrent_state, d_out, d_state_next)


def _single_step_gdr_bwd(
    use_qk_l2norm: bool,
    res: tuple,
    g: tuple[Float[Array, "..."], Float[Array, "..."]],
) -> tuple:
    query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none = res
    d_out, d_final_state = g
    input_dtype = query.dtype
    scale = 1.0 / math.sqrt(query.shape[-1])
    d_query, d_key, d_value, d_beta, d_decay, d_state = _run_single_step_backward(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        d_out.astype(jnp.float32),
        d_final_state.astype(jnp.float32),
    )
    d_beta = d_beta[..., 0]
    d_decay = d_decay[..., 0]
    d_query = d_query * scale
    if use_qk_l2norm:
        q_norm = query / scale
        k_norm = key
        d_query = _l2norm_bwd(d_query, q_norm, q_inv_norm.astype(jnp.float32))
        d_key = _l2norm_bwd(d_key, k_norm, k_inv_norm.astype(jnp.float32))
    if decay_was_none:
        d_decay = None
    return (
        _cast_grad(d_query, input_dtype),
        _cast_grad(d_key, input_dtype),
        _cast_grad(d_value, input_dtype),
        _cast_grad(d_beta, input_dtype),
        _cast_grad(d_decay, input_dtype),
        _cast_grad(d_state, input_dtype),
    )
