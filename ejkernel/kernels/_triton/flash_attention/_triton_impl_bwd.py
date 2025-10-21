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


import math
from typing import Any

import jax
import triton
import triton.language as tl
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int
from triton import Config

from ejkernel.callib import triton_call
from ejkernel.ops import BwdParams, FwdParams
from ejkernel.utils import dtype_index, get_sharding, get_strides

from ._utilities import attention_pack_with_static_shape, calc_bias_strides, padded_load


def config_prune_kernel(
    configs: list[Config],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Config]:
    """Prune autotuning configurations for backward pass kernel.

    Filters out configurations where block dimensions exceed sequence lengths.
    Falls back to small default configs if all configs are pruned.

    Args:
        configs: List of triton autotuning configurations
        named_args: Dictionary with kernel arguments including QSeq and KSeq
        **kwargs: Additional unused arguments

    Returns:
        list[Config]: Valid configurations for the given problem size
    """
    kept_configs = []
    for config in configs:
        largest_m = (
            max(
                config.kwargs["BLOCK_M1"],
                config.kwargs["BLOCK_M2"],
            )
            > named_args["QSeq"]
        )
        largest_n = (
            max(
                config.kwargs["BLOCK_N1"],
                config.kwargs["BLOCK_N2"],
            )
            > named_args["KSeq"]
        )
        if largest_m or largest_n:
            pass
        else:
            kept_configs.append(config)
    if kept_configs:
        return kept_configs
    return [
        Config(
            {
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
            },
            num_warps=4,
            num_stages=0,
        ),
        Config(
            {
                "BLOCK_M1": 32,
                "BLOCK_N1": 32,
                "BLOCK_M2": 32,
                "BLOCK_N2": 32,
            },
            num_warps=2,
            num_stages=0,
        ),
    ]


@triton.autotune(
    configs=[
        Config({"BLOCK_M": 16}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 32}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 64}, num_warps=4, num_stages=0),
        Config({"BLOCK_M": 128}, num_warps=4, num_stages=0),
    ],
    key=["CQSeq", "DRuntime"],
)
@triton.jit
def _attn_bwd_preprocess(
    Po,
    Do,
    stride_oz,
    stride_om,
    stride_oh,
    stride_dez,
    stride_dem,
    stride_deh,
    nheads,
    QSeq,
    max_seqlen_q_rounded,
    cum_seqlens_q,
    headdim,
    CQSeq,
    DRuntime,
    Delta,
    VARLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    """Preprocessing kernel for backward pass gradient computation.

    Computes delta values needed for efficient gradient calculation by
    combining output gradients with output values.

    This kernel runs before the main backward pass to prepare intermediate
    values that are reused across all attention blocks.
    """
    start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // nheads
    off_h = off_zh % nheads
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    if VARLEN:
        start_seqlen_q = tl.load(cum_seqlens_q + off_z)
        actual_seqlen_q = tl.load(cum_seqlens_q + off_z + 1) - start_seqlen_q
        cu_seq_start_q = tl.load(cum_seqlens_q + off_z)
        off_z = 0
    else:
        actual_seqlen_q = QSeq
        cu_seq_start_q = 0

    o_ptrs = (
        Po
        + off_z * stride_oz
        + off_h * stride_oh
        + cu_seq_start_q * stride_om
        + offs_m[:, None] * stride_om
        + offs_d[None, :]
    )
    do_ptrs = (
        Do
        + off_z * stride_dez
        + off_h * stride_deh
        + cu_seq_start_q * stride_dem
        + offs_m[:, None] * stride_dem
        + offs_d[None, :]
    )

    mask = (offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim)
    o = tl.load(o_ptrs, mask=mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask, other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_zh * max_seqlen_q_rounded + offs_m, delta)


@triton.jit
def _attn_bwd_dkdv(
    index_start_m,
    k,
    v,
    dk,
    dv,
    M,
    D,
    offs_m,
    offs_n,
    offs_d,
    q_ptrs,
    bias_ptrs,
    dropout_offs,
    do_ptrs,
    softmax_scale,
    stride_qm,
    stride_bm,
    stride_dom,
    actual_seqlen_q,
    actual_seqlen_k,
    fully_masked_lines,
    headdim,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    BIG_NEG: tl.constexpr = -2147483648
    LN2: tl.constexpr = 1.44269504089

    q_ptrs = q_ptrs + index_start_m * stride_qm
    do_ptrs = do_ptrs + index_start_m * stride_dom
    if BIAS_ON:
        bias_ptrs = bias_ptrs + index_start_m * stride_bm
    if USE_DROPOUT:
        dropout_offs += index_start_m * actual_seqlen_k

    offs_m_curr = index_start_m + offs_m

    q = padded_load(
        q_ptrs,
        offs_m_curr,
        offs_d,
        PA0=PAD_ROWS or HEADS_PADDED,
        PA1=PAD_ROWS or HEADS_PADDED,
        LA0=actual_seqlen_q,
        LA1=headdim,
    )
    me_i = tl.load(M + offs_m_curr)

    if BIAS_ON:
        bias = padded_load(
            bias_ptrs,
            offs_m_curr,
            offs_n,
            PA0=PAD_ROWS or HEADS_PADDED,
            PA1=PAD_ROWS or HEADS_PADDED,
            LA0=actual_seqlen_q,
            LA1=actual_seqlen_k,
        )

    qk = tl.dot(q, tl.trans(k)).to(tl.float32)

    if BIAS_ON:
        if BOOL_BIAS:
            qk = tl.where(bias, qk, BIG_NEG)
        else:
            qk += bias / softmax_scale

    offs_n_causal = offs_n - actual_seqlen_k + actual_seqlen_q
    if MASKED:
        if PAD_COLS:
            if IS_CAUSAL:
                qk = tl.where(
                    tl.minimum(actual_seqlen_q - 1, offs_m_curr)[:, None] >= offs_n_causal[None, :],
                    qk,
                    float("-inf"),
                )
            else:
                qk = tl.where(actual_seqlen_q - 1 >= offs_n_causal[None, :], qk, float("-inf"))
        elif IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= offs_n_causal[None, :], qk, float("-inf"))

    if SLIDING:
        shift = actual_seqlen_k - actual_seqlen_q
        j_aligned = offs_n[None, :] - shift
        i_idx = offs_m_curr[:, None]
        in_window = (j_aligned >= (i_idx - window_left)) & (j_aligned <= (i_idx + window_right))
        qk = tl.where(in_window, qk, float("-inf"))

    if SOFTCAP:
        s = qk * softmax_scale
        x = s / logits_soft_cap
        exp_2x = tl.exp(2.0 * x)
        tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)

        qk_after = (logits_soft_cap * tanh_x) * LN2

        jac = softmax_scale * (1.0 - tanh_x * tanh_x)
    else:
        qk_after = qk * (softmax_scale * LN2)
        jac = softmax_scale

    tl.debug_barrier()

    p = tl.exp2(qk_after - me_i[:, None])

    if MASKED and (fully_masked_lines > 0):
        p = tl.where(offs_m_curr[:, None] < fully_masked_lines, 0, p)

    do = padded_load(
        do_ptrs,
        offs_m_curr,
        offs_d,
        PA0=PAD_ROWS,
        PA1=HEADS_PADDED,
        LA0=actual_seqlen_q,
        LA1=headdim,
    ).to(tl.float32)

    dv += tl.dot(tl.trans(p), do)
    dp = tl.dot(do, tl.trans(v.to(tl.float32)))
    Di = tl.load(D + offs_m_curr)
    ds = (p * (dp - Di[:, None]) * jac).to(q.dtype)
    dk += tl.dot(tl.trans(ds), q)

    return dk, dv


@triton.jit
def _attn_bwd_block_dkdv(
    index_start_n,
    Q,
    K,
    V,
    B,
    Dropout,
    Do,
    Dk,
    Dv,
    M,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dkn,
    stride_dvn,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    """Process a block of K/V positions for gradient computation.

    Iterates through query blocks to accumulate gradients for a specific
    block of key and value positions. Handles causal masking and sliding
    window constraints.

    This is the main workhorse for K/V gradient computation, called for
    each K/V block position in the sequence.
    """
    index_begin_m = max(index_start_n + actual_seqlen_q - actual_seqlen_k, 0) if IS_CAUSAL else 0
    index_begin_m = (index_begin_m // BLOCK_M) * BLOCK_M
    index_end_m = actual_seqlen_q

    fully_masked_lines = (actual_seqlen_q - actual_seqlen_k) if IS_CAUSAL else 0
    if (index_begin_m >= actual_seqlen_q) or (index_start_n >= actual_seqlen_k):
        return

    offs_n = index_start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    dk_ptrs = Dk + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dv_ptrs = Dv + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    do_ptrs = Do + (offs_m[:, None] * stride_dom + offs_d[None, :])
    bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n[None, :]) if BIAS_ON else None
    dropout_offs = Dropout + offs_m[:, None] * actual_seqlen_k + offs_n[None, :] if USE_DROPOUT else None

    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    k = padded_load(k_ptrs, offs_n, offs_d, PA0=PAD_COLS, PA1=HEADS_PADDED, LA0=actual_seqlen_k, LA1=headdim)
    v = padded_load(v_ptrs, offs_n, offs_d, PA0=PAD_COLS, PA1=HEADS_PADDED, LA0=actual_seqlen_k, LA1=headdim)

    fr = max(0, index_start_n + BLOCK_N - 1 + actual_seqlen_q - actual_seqlen_k)
    fb = BLOCK_M * ((min(fr, actual_seqlen_q) + BLOCK_M - 1) // BLOCK_M)
    num_masked_blocks = (fb - index_begin_m) // BLOCK_M if IS_CAUSAL else 0
    index_next_start_m = index_begin_m

    if num_masked_blocks > 0:
        for _ in range(0, num_masked_blocks):
            dk, dv = _attn_bwd_dkdv(
                index_next_start_m,
                k,
                v,
                dk,
                dv,
                M,
                D,
                offs_m,
                offs_n,
                offs_d,
                q_ptrs,
                bias_ptrs,
                dropout_offs,
                do_ptrs,
                softmax_scale,
                stride_qm,
                stride_bm,
                stride_dom,
                actual_seqlen_q,
                actual_seqlen_k,
                fully_masked_lines,
                headdim,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                MASKED=True,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                USE_DROPOUT=USE_DROPOUT,
                PAD_ROWS=True,
                PAD_COLS=PAD_COLS,
                HEADS_PADDED=HEADS_PADDED,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
            )
            index_next_start_m += BLOCK_M

    if index_next_start_m < index_end_m:
        for index_start_m in range(index_next_start_m, index_end_m, BLOCK_M):
            dk, dv = _attn_bwd_dkdv(
                index_start_m,
                k,
                v,
                dk,
                dv,
                M,
                D,
                offs_m,
                offs_n,
                offs_d,
                q_ptrs,
                bias_ptrs,
                dropout_offs,
                do_ptrs,
                softmax_scale,
                stride_qm,
                stride_bm,
                stride_dom,
                actual_seqlen_q,
                actual_seqlen_k,
                fully_masked_lines,
                headdim,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                MASKED=False,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                USE_DROPOUT=USE_DROPOUT,
                PAD_ROWS=True,
                PAD_COLS=PAD_COLS,
                HEADS_PADDED=HEADS_PADDED,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
            )

    if HEADS_PADDED:
        if PAD_COLS:
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < actual_seqlen_k) & (offs_d[None, :] < headdim))
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
    else:
        if PAD_COLS:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < actual_seqlen_k)
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < actual_seqlen_k)
        else:
            tl.store(dk_ptrs, dk)
            tl.store(dv_ptrs, dv)


@triton.jit
def _attn_bwd_dq(
    index_start_n,
    q,
    dq,
    do,
    me_i,
    de_i,
    offs_m,
    offs_n,
    offs_d,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    dropout_offs,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    stride_kn,
    stride_vn,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    MASKED: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_COLS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    BIG_NEG: tl.constexpr = -2147483648
    LN2: tl.constexpr = 1.44269504089

    k_ptrs = k_ptrs + index_start_n * stride_kn
    v_ptrs = v_ptrs + index_start_n * stride_vn
    offs_n_curr = index_start_n + offs_n
    if BIAS_ON:
        bias_ptrs += index_start_n
    if USE_DROPOUT:
        dropout_offs += index_start_n

    k = padded_load(k_ptrs, offs_n_curr, offs_d, PA0=PAD_COLS, PA1=HEADS_PADDED, LA0=actual_seqlen_k, LA1=headdim)
    v = padded_load(v_ptrs, offs_n_curr, offs_d, PA0=PAD_COLS, PA1=HEADS_PADDED, LA0=actual_seqlen_k, LA1=headdim)
    if BIAS_ON:
        bias = padded_load(
            bias_ptrs, offs_m, offs_n_curr, PA0=True, PA1=PAD_COLS, LA0=actual_seqlen_q, LA1=actual_seqlen_k
        )

    qk = tl.dot(q, tl.trans(k))

    if BIAS_ON:
        if BOOL_BIAS:
            qk = tl.where(bias, qk, BIG_NEG)
        else:
            qk += bias / softmax_scale

    offs_n_causal = offs_n_curr - actual_seqlen_k + actual_seqlen_q
    if MASKED:
        if PAD_COLS:
            if IS_CAUSAL:
                qk = tl.where(
                    tl.minimum(actual_seqlen_q - 1, offs_m)[:, None] >= offs_n_causal[None, :],
                    qk,
                    float("-inf"),
                )
            else:
                qk = tl.where(actual_seqlen_q - 1 >= offs_n_causal[None, :], qk, float("-inf"))
        elif IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n_causal[None, :], qk, float("-inf"))

    if SLIDING:
        shift = actual_seqlen_k - actual_seqlen_q
        j_aligned = offs_n_curr[None, :] - shift
        i_idx = offs_m[:, None]
        in_window = (j_aligned >= (i_idx - window_left)) & (j_aligned <= (i_idx + window_right))
        qk = tl.where(in_window, qk, float("-inf"))

    if SOFTCAP:
        s = qk * softmax_scale
        x = s / logits_soft_cap
        exp_2x = tl.exp(2.0 * x)
        tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)

        qk_after = (logits_soft_cap * tanh_x) * LN2

        jac = softmax_scale * (1.0 - tanh_x * tanh_x)
    else:
        qk_after = qk * (softmax_scale * LN2)
        jac = softmax_scale

    tl.debug_barrier()

    p = tl.exp2(qk_after - me_i[:, None])
    dp = tl.dot(do, tl.trans(v.to(tl.float32)))
    ds = (p * (dp - de_i[:, None]) * jac).to(q.dtype)
    dq += tl.dot(ds, k)
    return dq


@triton.jit
def _attn_bwd_block_dq(
    index_start_m,
    Q,
    K,
    V,
    B,
    Dropout,
    Do,
    Dq,
    M,
    D,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    PAD_ROWS: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_N: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    if IS_CAUSAL:
        index_end_n = min(
            actual_seqlen_k - actual_seqlen_q + index_start_m + BLOCK_M,
            actual_seqlen_k,
        )
        if index_end_n < 0:
            return
    else:
        index_end_n = actual_seqlen_k

    fully_masked_lines = actual_seqlen_q - actual_seqlen_k if IS_CAUSAL else 0
    mask_reached = fully_masked_lines >= index_start_m + BLOCK_M
    if (index_start_m >= actual_seqlen_q) or mask_reached:
        return

    offs_m = tl.arange(0, BLOCK_M) + index_start_m
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])

    dq_ptrs = Dq + (offs_m[:, None] * stride_dqm + offs_d[None, :])
    do_ptrs = Do + (offs_m[:, None] * stride_dom + offs_d[None, :])

    if BIAS_ON:
        bias_ptrs = B + (offs_m[:, None] * stride_bm + offs_n[None, :])
    else:
        bias_ptrs = None

    if USE_DROPOUT:
        dropout_offs = Dropout + offs_m[:, None] * actual_seqlen_k + offs_n[None, :]
    else:
        dropout_offs = None

    dq = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)
    q = padded_load(q_ptrs, offs_m, offs_d, PA0=PAD_ROWS, PA1=HEADS_PADDED, LA0=actual_seqlen_q, LA1=headdim)
    do = padded_load(do_ptrs, offs_m, offs_d, PA0=PAD_ROWS, PA1=HEADS_PADDED, LA0=actual_seqlen_q, LA1=headdim).to(
        tl.float32
    )
    me_i = tl.load(M + offs_m)
    de_i = tl.load(D + offs_m)

    uneven_n = actual_seqlen_k % BLOCK_N != 0
    attention_padding = VARLEN & uneven_n
    if IS_CAUSAL:
        first_masked_col = index_start_m + 1 + actual_seqlen_k - actual_seqlen_q
    elif attention_padding:
        first_masked_col = actual_seqlen_k
    else:
        first_masked_col = index_end_n
    nb_full_blocks = first_masked_col // BLOCK_N

    index_next_start_n = 0
    if nb_full_blocks > 0:
        for _ in range(0, nb_full_blocks):
            index_next_start_n = tl.multiple_of(index_next_start_n, BLOCK_N)
            dq = _attn_bwd_dq(
                index_next_start_n,
                q,
                dq,
                do,
                me_i,
                de_i,
                offs_m,
                offs_n,
                offs_d,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                dropout_offs,
                softmax_scale,
                dropout_prob,
                dropout_seed,
                stride_kn,
                stride_vn,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                USE_DROPOUT=USE_DROPOUT,
                MASKED=False,
                PAD_COLS=False,
                HEADS_PADDED=HEADS_PADDED,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
            )
            index_next_start_n += BLOCK_N

    if index_next_start_n < index_end_n:
        for index_start_n in range(index_next_start_n, index_end_n, BLOCK_N):
            pad_cols = (not EVEN_N) or (VARLEN and (index_start_n + BLOCK_N > actual_seqlen_k))
            dq = _attn_bwd_dq(
                index_start_n,
                q,
                dq,
                do,
                me_i,
                de_i,
                offs_m,
                offs_n,
                offs_d,
                k_ptrs,
                v_ptrs,
                bias_ptrs,
                dropout_offs,
                softmax_scale,
                dropout_prob,
                dropout_seed,
                stride_kn,
                stride_vn,
                actual_seqlen_q,
                actual_seqlen_k,
                headdim,
                window_left,
                window_right,
                logits_soft_cap,
                softmax_aux_ptrs,
                num_sinks,
                IS_CAUSAL=IS_CAUSAL,
                BIAS_ON=BIAS_ON,
                BOOL_BIAS=BOOL_BIAS,
                USE_DROPOUT=USE_DROPOUT,
                MASKED=True,
                PAD_COLS=pad_cols,
                HEADS_PADDED=HEADS_PADDED,
                SLIDING=SLIDING,
                SOFTCAP=SOFTCAP,
                USE_SINKS=USE_SINKS,
            )

    if fully_masked_lines > 0:
        dq = tl.where(offs_m[:, None] < fully_masked_lines, 0, dq)

    if HEADS_PADDED:
        if PAD_ROWS:
            tl.store(dq_ptrs, dq, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))
        else:
            tl.store(dq_ptrs, dq, mask=offs_d[None, :] < headdim)
    else:
        if PAD_ROWS:
            tl.store(dq_ptrs, dq, mask=offs_m[:, None] < actual_seqlen_q)
        else:
            tl.store(dq_ptrs, dq)


@triton.autotune(
    configs=[
        Config(
            {"BLOCK_M1": 16, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 16},
            num_warps=2,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 32, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 32},
            num_warps=2,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
            num_warps=2,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64},
            num_warps=2,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 16, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 16},
            num_warps=4,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 32, "BLOCK_N1": 16, "BLOCK_M2": 16, "BLOCK_N2": 32},
            num_warps=4,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 32, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 32},
            num_warps=4,
            num_stages=0,
        ),
        Config(
            {"BLOCK_M1": 64, "BLOCK_N1": 64, "BLOCK_M2": 64, "BLOCK_N2": 64},
            num_warps=4,
            num_stages=0,
        ),
    ],
    key=[
        "CQSeq",
        "CKSeq",
        "DRuntime",
        "VARLEN",
        "USE_DROPOUT",
        "IS_CAUSAL",
        "BIAS_ON",
        "BLOCK_HEADDIM",
        "SLIDING",
    ],
    prune_configs_by={"early_config_prune": config_prune_kernel},
)
@triton.heuristics(
    {
        "EVEN_M1": lambda args: args["QSeq"] % args["BLOCK_M1"] == 0,
        "EVEN_N1": lambda args: args["KSeq"] % args["BLOCK_N1"] == 0,
        "EVEN_M2": lambda args: args["QSeq"] % args["BLOCK_M2"] == 0,
        "EVEN_N2": lambda args: args["KSeq"] % args["BLOCK_N2"] == 0,
        "HEADS_PADDED": lambda args: args["headdim"] != args["BLOCK_HEADDIM"],
        "NUM_BLOCKS_KV": lambda args: math.ceil(args["KSeq"] / args["BLOCK_N1"]),
    }
)
@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    B,
    Do,
    M,
    D,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_bz,
    stride_bm,
    stride_bh,
    stride_doz,
    stride_dom,
    stride_doh,
    stride_dqz,
    stride_dqm,
    stride_dqh,
    stride_dkz,
    stride_dkn,
    stride_dkh,
    stride_dvz,
    stride_dvn,
    stride_dvh,
    nheads_q,
    num_repeats,
    window_left,
    window_right,
    QSeq,
    cum_seqlens_q,
    KSeq,
    cum_seqlens_k,
    seqlen_q_rounded,
    headdim,
    CQSeq,
    CKSeq,
    DRuntime,
    logits_soft_cap,
    softmax_aux,
    num_sinks,
    Dq,
    Dk,
    Dv,
    VARLEN: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M1: tl.constexpr,
    EVEN_N1: tl.constexpr,
    EVEN_M2: tl.constexpr,
    EVEN_N2: tl.constexpr,
    NUM_BLOCKS_KV: tl.constexpr,
    HEADS_PADDED: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    """Main backward pass kernel for flash attention gradient computation.

    Orchestrates the computation of gradients for Q, K, and V tensors using
    a two-phase approach: first computing Q gradients, then K/V gradients.

    This kernel is the entry point for the backward pass, managing work
    distribution across thread blocks for efficient gradient computation.
    """
    pid = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_z = off_zh // nheads_q
    off_head_q = off_zh % nheads_q
    off_head_kv = off_head_q // num_repeats

    if VARLEN:
        cu_seq_start_q = tl.load(cum_seqlens_q + off_z)
        cu_seq_start_k = tl.load(cum_seqlens_k + off_z)
        actual_seqlen_q = tl.load(cum_seqlens_q + off_z + 1) - cu_seq_start_q
        actual_seqlen_k = tl.load(cum_seqlens_k + off_z + 1) - cu_seq_start_k
        off_z = 0
    else:
        cu_seq_start_q = 0
        cu_seq_start_k = 0
        actual_seqlen_q = QSeq
        actual_seqlen_k = KSeq

    Q += off_z * stride_qz + off_head_q * stride_qh + cu_seq_start_q * stride_qm
    K += off_z * stride_kz + off_head_kv * stride_kh + cu_seq_start_k * stride_kn
    V += off_z * stride_vz + off_head_kv * stride_vh + cu_seq_start_k * stride_vn

    Do += off_z * stride_doz + off_head_q * stride_doh + cu_seq_start_q * stride_dom
    Dq += off_z * stride_dqz + off_head_q * stride_dqh + cu_seq_start_q * stride_dqm
    Dk += off_z * stride_dkz + off_head_q * stride_dkh + cu_seq_start_k * stride_dkn
    Dv += off_z * stride_dvz + off_head_q * stride_dvh + cu_seq_start_k * stride_dvn

    if BIAS_ON:
        B += off_z * stride_bz + off_head_q * stride_bh + cu_seq_start_q * stride_bm
    Dropout = (
        actual_seqlen_k * (cu_seq_start_q + actual_seqlen_q * (off_head_q + nheads_q * off_z)) if USE_DROPOUT else None
    )

    if USE_SINKS:
        softmax_aux_ptrs = softmax_aux + off_head_q * num_sinks
    else:
        softmax_aux_ptrs = softmax_aux

    D += off_zh * seqlen_q_rounded
    M += off_zh * seqlen_q_rounded

    if pid < NUM_BLOCKS_KV:
        i_start_n = pid
        pad_cols = (not EVEN_N1) or (VARLEN and ((i_start_n + 1) * BLOCK_N1 > actual_seqlen_k))
        _attn_bwd_block_dkdv(
            i_start_n * BLOCK_N1,
            Q,
            K,
            V,
            B,
            Dropout,
            Do,
            Dk,
            Dv,
            M,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dkn,
            stride_dvn,
            actual_seqlen_q,
            actual_seqlen_k,
            headdim,
            window_left,
            window_right,
            logits_soft_cap,
            softmax_aux_ptrs,
            num_sinks,
            IS_CAUSAL=IS_CAUSAL,
            BIAS_ON=BIAS_ON,
            BOOL_BIAS=BOOL_BIAS,
            USE_DROPOUT=USE_DROPOUT,
            PAD_COLS=pad_cols,
            HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M1,
            BLOCK_N=BLOCK_N1,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            SLIDING=SLIDING,
            SOFTCAP=SOFTCAP,
            USE_SINKS=USE_SINKS,
        )
    else:
        i_start_m = pid - NUM_BLOCKS_KV
        pad_rows = (not EVEN_M2) or (VARLEN and ((i_start_m + 1) * BLOCK_M2 > actual_seqlen_q))
        _attn_bwd_block_dq(
            i_start_m * BLOCK_M2,
            Q,
            K,
            V,
            B,
            Dropout,
            Do,
            Dq,
            M,
            D,
            softmax_scale,
            dropout_prob,
            dropout_seed,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            actual_seqlen_q,
            actual_seqlen_k,
            headdim,
            window_left,
            window_right,
            logits_soft_cap,
            softmax_aux_ptrs,
            num_sinks,
            VARLEN=VARLEN,
            IS_CAUSAL=IS_CAUSAL,
            BIAS_ON=BIAS_ON,
            BOOL_BIAS=BOOL_BIAS,
            USE_DROPOUT=USE_DROPOUT,
            PAD_ROWS=pad_rows,
            HEADS_PADDED=HEADS_PADDED,
            BLOCK_M=BLOCK_M2,
            BLOCK_N=BLOCK_N2,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_N=EVEN_N2,
            SLIDING=SLIDING,
            SOFTCAP=SOFTCAP,
            USE_SINKS=USE_SINKS,
        )


def _bwd_attention_kernel_call(
    dO: Float[Array, "batch seq_len_q num_heads head_dim"],
    q: Float[Array, "batch seq_len_q num_heads head_dim"],
    k: Float[Array, "batch seq_len_k num_heads head_dim"],
    v: Float[Array, "batch seq_len_k num_heads head_dim"],
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None,
    attention_mask: Bool[Array, "batch seq_len"] | None,
    o: Float[Array, "batch seq_len_q num_heads head_dim"],
    M: Float[Array, "batch num_heads max_seqlen_q_rounded"],
    dropout_prob: float,
    causal: bool,
    softmax_scale: float | None,
    dropout_seed: int | None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_sinks"] | Float[Array, "num_heads num_sinks"] | None = None,
) -> tuple[
    Float[Array, "batch seq_len_q num_heads head_dim"],
    Float[Array, "batch seq_len_k num_heads head_dim"],
    Float[Array, "batch seq_len_k num_heads head_dim"],
]:
    """Execute flash attention backward pass using Triton kernels.

    Prepares inputs and launches Triton kernels for gradient computation.
    Handles preprocessing and main backward kernel execution.

    Args:
        dO: Gradient of loss with respect to attention output
        q, k, v: Query, key, value tensors from forward pass
        bias: Optional attention bias from forward pass
        attention_mask: Legacy mask parameter
        o: Output from forward pass
        M: Log-sum-exp values from forward pass
        dropout_prob: Dropout probability used in forward pass
        causal: Whether causal masking was applied
        softmax_scale: Attention score scaling factor
        dropout_seed: Random seed for dropout
        cum_seqlens_q/k: Cumulative sequence lengths for variable-length mode
        sliding_window: Local attention window size

    Returns:
        tuple: Gradients (dq, dk, dv) for query, key, and value tensors
    """

    if sliding_window is None:
        window_left = 0
        window_right = 0
        sliding_flag = False
    else:
        if isinstance(sliding_window, int):
            window_left = int(sliding_window)
            window_right = 0 if causal else int(sliding_window)
        else:
            wl, wr = sliding_window
            window_left = int(wl)
            window_right = int(wr)
        assert window_left >= 0 and window_right >= 0
        sliding_flag = (window_left > 0) or (window_right > 0)

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
        softcap_flag = False
    else:
        logits_soft_cap_val = float(logits_soft_cap)
        softcap_flag = True

    if softmax_aux is None:
        use_sinks = False
        num_sinks_val = 0
        softmax_aux_tensor = jnp.zeros((1,), dtype=q.dtype)
    else:
        use_sinks = True

        if softmax_aux.ndim == 1:
            num_sinks_val = softmax_aux.shape[0]
            num_heads = q.shape[2]
            softmax_aux_tensor = jnp.broadcast_to(softmax_aux[None, :], (num_heads, num_sinks_val))
        elif softmax_aux.ndim == 2:
            num_sinks_val = softmax_aux.shape[1]
            softmax_aux_tensor = softmax_aux
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

        head_dim = q.shape[-1]
        softmax_scale = 1.0 / math.sqrt(float(head_dim)) if softmax_scale is None else softmax_scale
        softmax_aux_tensor = softmax_aux_tensor * softmax_scale
        if softcap_flag:
            softmax_aux_tensor = logits_soft_cap_val * jnp.tanh(softmax_aux_tensor / logits_soft_cap_val)

    varlen_from_cu = (cum_seqlens_q is not None) and (cum_seqlens_k is not None)
    if varlen_from_cu:
        assert cum_seqlens_q.dtype == jnp.int32 and cum_seqlens_k.dtype == jnp.int32
        batch_size, QSeq_max, nheads_q, head_dim = q.shape
        _, KSeq_max, nheads_kv, _ = k.shape
        assert nheads_q % nheads_kv == 0
        num_repeats = nheads_q // nheads_kv
        BOOL_BIAS = False
        softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

        max_seqlen_q = QSeq_max
        max_seqlen_k = KSeq_max
        max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128
        BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)

        from ._utilities import attention_pack_from_cu_static, attention_unpack_with_static_shape

        q_p = attention_pack_from_cu_static(q, cum_seqlens_q, max_tokens=batch_size * QSeq_max)
        k_p = attention_pack_from_cu_static(k, cum_seqlens_k, max_tokens=batch_size * KSeq_max)
        v_p = attention_pack_from_cu_static(v, cum_seqlens_k, max_tokens=batch_size * KSeq_max)
        o_p = attention_pack_from_cu_static(o, cum_seqlens_q, max_tokens=batch_size * QSeq_max)
        dO_p = attention_pack_from_cu_static(dO, cum_seqlens_q, max_tokens=batch_size * QSeq_max)

        oz, om, oh, _ = get_strides(o_p)
        doz, dom, doh, _ = get_strides(dO_p)
        qz, qm, qh, _ = get_strides(q_p)
        kz, kn, kh, _ = get_strides(k_p)
        vz, vn, vh, _ = get_strides(v_p)

        (delta,) = triton_call(
            o_p,
            dO_p,
            oz,
            om,
            oh,
            doz,
            dom,
            doh,
            nheads_q,
            max_seqlen_q,
            max_seqlen_q_rounded,
            cum_seqlens_q,
            head_dim,
            max_seqlen_q // 32,
            dtype_index(q_p),
            VARLEN=True,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            out_shape=[jax.ShapeDtypeStruct(shape=M.shape, dtype="f4", sharding=get_sharding(M))],
            grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch_size * nheads_q),
            kernel=_attn_bwd_preprocess,
            name="ejkernel::triton::flash_attn_bwd_preprocess",
        )

        bz = bm = bh = 0

        dq, dk, dv = triton_call(
            q_p,
            k_p,
            v_p,
            jnp.zeros((1,), q.dtype),
            dO_p,
            M,
            delta,
            softmax_scale,
            dropout_prob,
            dropout_seed if dropout_seed is not None else jnp.zeros((1,), q.dtype),
            qz,
            qm,
            qh,
            kz,
            kn,
            kh,
            vz,
            vn,
            vh,
            bz,
            bm,
            bh,
            doz,
            dom,
            doh,
            qz,
            qm,
            qh,
            kz,
            kn,
            kh,
            vz,
            vn,
            vh,
            nheads_q,
            num_repeats,
            window_left,
            window_right,
            max_seqlen_q,
            cum_seqlens_q,
            max_seqlen_k,
            cum_seqlens_k,
            max_seqlen_q_rounded,
            head_dim,
            max_seqlen_q // 32,
            max_seqlen_k // 32,
            dtype_index(q_p),
            logits_soft_cap_val,
            softmax_aux_tensor,
            num_sinks_val,
            BIAS_ON=False,
            VARLEN=True,
            IS_CAUSAL=causal,
            USE_DROPOUT=(dropout_prob > 0),
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            BOOL_BIAS=False,
            SLIDING=sliding_flag,
            SOFTCAP=softcap_flag,
            USE_SINKS=use_sinks,
            kernel=_attn_bwd,
            grid=lambda META: (
                triton.cdiv(max_seqlen_k, META["BLOCK_N1"]) + triton.cdiv(max_seqlen_q, META["BLOCK_M2"]),
                batch_size * nheads_q,
            ),
            out_shape=[
                jax.ShapeDtypeStruct(shape=q_p.shape, dtype="f4", sharding=get_sharding(q)),
                jax.ShapeDtypeStruct(shape=(*k_p.shape[:2], q_p.shape[2], k_p.shape[3]), dtype=k.dtype),
                jax.ShapeDtypeStruct(shape=(*v_p.shape[:2], q_p.shape[2], v_p.shape[3]), dtype=v.dtype),
            ],
            name="ejkernel::triton::flash_attn_bwd",
        )

        if num_repeats > 1:
            dk = dk.reshape(dk.shape[0], dk.shape[1], (nheads_q // num_repeats), num_repeats, -1).sum(axis=3)
            dv = dv.reshape(dv.shape[0], dv.shape[1], (nheads_q // num_repeats), num_repeats, -1).sum(axis=3)

        dq = attention_unpack_with_static_shape(dq, cum_seqlens_q, batch_size, QSeq_max)
        dk = attention_unpack_with_static_shape(dk, cum_seqlens_k, batch_size, KSeq_max)
        dv = attention_unpack_with_static_shape(dv, cum_seqlens_k, batch_size, KSeq_max)
        return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)

    if attention_mask is not None and varlen_from_cu:
        assert bias is None, "mask + bias not supported; use bias alone or pack bias."
        assert q.shape[1] == k.shape[1], "mask varlen path supports QSeq == KSeq only."
        varlen_mode = attention_mask.shape[0] > 1
        useless_padding = attention_mask.shape[1] - attention_mask.sum(-1).max().item()
        if useless_padding > 0:
            dO = dO[:, :-useless_padding]
            q = q[:, :-useless_padding]
            k = k[:, :-useless_padding]
            v = v[:, :-useless_padding]
            attention_mask = attention_mask[:, :-useless_padding]
            o = o[:, :-useless_padding]
    else:
        varlen_mode = False
        useless_padding = 0

    batch_size, QSeq, nheads_q, head_dim = q.shape
    _, KSeq, nheads_kv, _ = k.shape
    max_seqlen_q_rounded = math.ceil(QSeq / 128) * 128
    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    assert nheads_q % nheads_kv == 0
    assert M.shape == (batch_size, nheads_q, max_seqlen_q_rounded)
    BOOL_BIAS = False
    if not varlen_mode and attention_mask is not None:
        assert bias is None, "mask + bias not supported"
        BOOL_BIAS = True
        bias = attention_mask.astype(jnp.bool_)

    if varlen_mode:
        cum_seqlens_q = jnp.zeros((attention_mask.shape[0] + 1,), dtype=jnp.int32)
        cum_seqlens_k = jnp.zeros((attention_mask.shape[0] + 1,), dtype=jnp.int32)
        lengths = attention_mask.sum(axis=1, dtype="i4")
        cum_seqlens_q = cum_seqlens_q.at[1:].set(jnp.cumsum(lengths, axis=0, dtype="i4"))
        cum_seqlens_k = cum_seqlens_k.at[1:].set(jnp.cumsum(lengths, axis=0, dtype="i4"))
        max_seqlen_q = attention_mask.shape[1]
        max_seqlen_k = attention_mask.shape[1]

        dO = attention_pack_with_static_shape(dO, attention_mask)
        q = attention_pack_with_static_shape(q, attention_mask)
        k = attention_pack_with_static_shape(k, attention_mask)
        v = attention_pack_with_static_shape(v, attention_mask)
        o = attention_pack_with_static_shape(o, attention_mask)
        QSeq = q.shape[1]
        KSeq = k.shape[1]
    else:
        cum_seqlens_q = None
        cum_seqlens_k = None
        max_seqlen_q = QSeq
        max_seqlen_k = KSeq

    bz, bh, bm = calc_bias_strides(bias, batch_size, nheads_q, QSeq, KSeq)
    num_repeats = nheads_q // nheads_kv
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)

    oz, om, oh, _ = get_strides(o)
    doz, dom, doh, _ = get_strides(dO)
    qz, qm, qh, _ = get_strides(q)
    kz, kn, kh, _ = get_strides(k)
    vz, vn, vh, _ = get_strides(v)

    (delta,) = triton_call(
        o,
        dO,
        oz,
        om,
        oh,
        doz,
        dom,
        doh,
        nheads_q,
        QSeq,
        max_seqlen_q_rounded,
        cum_seqlens_q if cum_seqlens_q is not None else jnp.zeros((1,), jnp.int32),
        head_dim,
        max_seqlen_q // 32,
        dtype_index(q),
        VARLEN=varlen_mode,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        out_shape=[jax.ShapeDtypeStruct(shape=M.shape, dtype="f4", sharding=get_sharding(M))],
        grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch_size * nheads_q),
        kernel=_attn_bwd_preprocess,
        name="ejkernel::triton::flash_attn_bwd_preprocess",
    )

    dq, dk, dv = triton_call(
        q,
        k,
        v,
        bias if bias is not None else jnp.zeros((1,), jnp.float16),
        dO,
        M,
        delta,
        softmax_scale,
        dropout_prob,
        dropout_seed if dropout_seed is not None else jnp.zeros((1,), jnp.float16),
        qz,
        qm,
        qh,
        kz,
        kn,
        kh,
        vz,
        vn,
        vh,
        bz,
        bm,
        bh,
        doz,
        dom,
        doh,
        qz,
        qm,
        qh,
        kz,
        kn,
        kh,
        vz,
        vn,
        vh,
        nheads_q,
        num_repeats,
        window_left,
        window_right,
        QSeq,
        cum_seqlens_q if cum_seqlens_q is not None else jnp.zeros((1,), jnp.int32),
        KSeq,
        cum_seqlens_k if cum_seqlens_k is not None else jnp.zeros((1,), jnp.int32),
        max_seqlen_q_rounded,
        head_dim,
        max_seqlen_q // 32,
        max_seqlen_k // 32,
        dtype_index(q),
        logits_soft_cap_val,
        softmax_aux_tensor,
        num_sinks_val,
        BIAS_ON=(bias is not None),
        VARLEN=varlen_mode,
        IS_CAUSAL=causal,
        USE_DROPOUT=(dropout_prob > 0),
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BOOL_BIAS=BOOL_BIAS,
        SLIDING=sliding_flag,
        SOFTCAP=softcap_flag,
        USE_SINKS=use_sinks,
        kernel=_attn_bwd,
        grid=lambda META: (
            triton.cdiv(KSeq, META["BLOCK_N1"]) + triton.cdiv(QSeq, META["BLOCK_M2"]),
            batch_size * nheads_q,
        ),
        out_shape=[
            jax.ShapeDtypeStruct(shape=q.shape, dtype="f4", sharding=get_sharding(q)),
            jax.ShapeDtypeStruct(shape=(k.shape[0], k.shape[1], q.shape[2], k.shape[3]), dtype="f4"),
            jax.ShapeDtypeStruct(shape=(v.shape[0], v.shape[1], q.shape[2], v.shape[3]), dtype="f4"),
        ],
        name="ejkernel::triton::flash_attn_bwd",
    )

    if num_repeats > 1:
        dk = dk.reshape(dk.shape[0], dk.shape[1], (nheads_q // num_repeats), num_repeats, -1).sum(axis=3)
        dv = dv.reshape(dv.shape[0], dv.shape[1], (nheads_q // num_repeats), num_repeats, -1).sum(axis=3)

    if varlen_mode:
        dq = attention_unpack_with_static_shape(dq, cum_seqlens_q, batch_size, max_seqlen_q)
        dk = attention_unpack_with_static_shape(dk, cum_seqlens_k, batch_size, max_seqlen_k)
        dv = attention_unpack_with_static_shape(dv, cum_seqlens_k, batch_size, max_seqlen_k)

    if useless_padding > 0:
        dq = jnp.pad(dq, ((0, useless_padding), (0, 0), (0, 0)))
        dk = jnp.pad(dk, ((0, useless_padding), (0, 0), (0, 0)))
        dv = jnp.pad(dv, ((0, useless_padding), (0, 0), (0, 0)))

    return dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype)
