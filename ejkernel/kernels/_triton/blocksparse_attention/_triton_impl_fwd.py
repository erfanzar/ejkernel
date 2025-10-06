# Copyright 2025
# blocksparse (block-sparse + Flash) forward kernel for Triton/JAX

import math
from typing import Any

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Bool, Float, Int
from triton import Config

from ejkernel.callib import triton_call
from ejkernel.utils import dtype_index, get_strides

from ._utilities import (
    attention_pack_from_cu_static,
    attention_unpack_with_static_shape,
    calc_bias_strides,
    padded_load,
)


def config_prune_blocksparse_kernel(
    configs: list[Config],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Config]:
    """
    Prunes autotune configs w.r.t. sequence sizes and optional forced tile sizes
    (FORCE_BLOCK_M, FORCE_BLOCK_N). Also ensures BLOCK_N doesn't exceed KSeq.
    """
    kept = []
    QSeq = named_args["QSeq"]
    KSeq = named_args["KSeq"]
    force_m = named_args.get("FORCE_BLOCK_M", 0)
    force_n = named_args.get("FORCE_BLOCK_N", 0)
    for cfg in configs:
        bm = cfg.kwargs["BLOCK_M"]
        bn = cfg.kwargs["BLOCK_N"]
        if bm > QSeq or bn > KSeq:
            continue
        if force_m and bm != force_m:
            continue
        if force_n and bn != force_n:
            continue
        kept.append(cfg)
    if kept:
        return kept
    # Fallback set (small to ensure compile)
    return [
        Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
    ]


@triton.jit
def _attn_fwd_inner(
    q,
    m_i,  # running m in natural units (log)
    me_i,  # running lse in natural units: me_i = m_i + log(l_i)
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    acc_o,  # running unnormalized output (o_scratch)
    offs_m,
    offs_n,
    offs_d,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    dropout_offs,
    window_left,
    window_right,
    logits_soft_cap,
    softmax_aux_ptrs,
    num_sinks,
    stride_kn,
    stride_vn,
    index_start_n,
    actual_seqlen_q,
    actual_seqlen_k,
    headdim,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    MASKED: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    PADDED_COLS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    BIG_NEG: tl.constexpr = -float("inf")

    index_start_n = tl.multiple_of(index_start_n, BLOCK_N)

    k = padded_load(
        k_ptrs + (index_start_n + offs_n)[:, None] * stride_kn + offs_d[None, :],
        index_start_n + offs_n,
        offs_d,
        PA0=PADDED_COLS,
        PA1=PADDED_HEADS,
        LA0=actual_seqlen_k,
        LA1=headdim,
    ).to(tl.float32)

    # Scores (fp32)
    qk = tl.dot(q.to(tl.float32), tl.trans(k))

    # Masks
    if PADDED_COLS:
        valid_cols = (index_start_n + offs_n)[None, :] < actual_seqlen_k
        qk = tl.where(valid_cols, qk, BIG_NEG)
    if MASKED and IS_CAUSAL:
        causal_mask = offs_m[:, None] >= (index_start_n + offs_n - actual_seqlen_k + actual_seqlen_q)[None, :]
        qk = tl.where(causal_mask, qk, BIG_NEG)
    if SLIDING:
        shift = actual_seqlen_k - actual_seqlen_q
        j_aligned = (index_start_n + offs_n)[None, :] - shift
        i_idx = offs_m[:, None]
        in_window = (j_aligned >= (i_idx - window_left)) & (j_aligned <= (i_idx + window_right))
        qk = tl.where(in_window, qk, BIG_NEG)

    # Bias (natural units)
    if BIAS_ON:
        bias = tl.load(
            bias_ptrs + index_start_n,
            mask=((offs_m[:, None] < actual_seqlen_q) & ((index_start_n + offs_n) < actual_seqlen_k)[None, :])
            if PADDED_COLS
            else (offs_m[:, None] < actual_seqlen_q),
            other=0.0,
        )
        if BOOL_BIAS:
            qk = tl.where(bias, qk, BIG_NEG)
        else:
            qk = qk + bias.to(qk.dtype)

    # Scale + softcap (natural)
    qk = qk * softmax_scale
    if SOFTCAP:
        qk = tl.tanh(qk / logits_soft_cap) * logits_soft_cap

    # Tile max (natural log space like TPU reference)
    # Load and process sinks once if needed
    if USE_SINKS:
        sink_offs = tl.arange(0, 16)
        sink_mask = sink_offs < num_sinks
        aux_logits = tl.load(softmax_aux_ptrs + sink_offs, mask=sink_mask, other=BIG_NEG).to(tl.float32)
        # Apply softcap if enabled (aux is in natural units)
        if SOFTCAP:
            aux_logits = tl.tanh(aux_logits / logits_soft_cap) * logits_soft_cap

    # Compute tile max including sinks
    qk_max = tl.max(qk, 1)
    if USE_SINKS:
        aux_max = tl.max(tl.where(sink_mask, aux_logits, BIG_NEG))
        m_curr = tl.maximum(qk_max, aux_max)
    else:
        m_curr = qk_max

    # Stable recenter on m_next; guard rows that are fully-masked in this tile
    m_prev = m_i
    m_next = tl.maximum(m_prev, m_curr)
    row_valid = m_next > BIG_NEG

    # Only compute qk - m_next for valid rows to avoid (-inf) subtraction
    qk_delta = tl.where(row_valid[:, None], qk - m_next[:, None], 0.0)
    # Clip to avoid exp underflow/NaN in fp16/bf16
    qk_delta = tl.maximum(qk_delta, -80.0)

    s_curr = tl.where(row_valid[:, None], tl.exp(qk_delta), 0.0)
    l_curr = tl.sum(s_curr, 1)

    # Add sink contributions to denominator
    if USE_SINKS:
        # Compute contribution for each row, broadcasting aux across rows
        l_aux = tl.where(row_valid, tl.sum(tl.where(sink_mask, tl.exp(aux_logits - m_next[:, None]), 0.0), axis=1), 0.0)
        l_curr = l_curr + l_aux

    # Previous l in linear space
    l_prev = tl.where(me_i > BIG_NEG, tl.exp(me_i - m_prev), 0.0)

    # Combine prev/current with alpha factor
    alpha = tl.where(m_prev > BIG_NEG, tl.exp(m_prev - m_next), 0.0)
    l_next = l_prev * alpha + l_curr

    # Optional dropout: drop numerator only (denom stays pre-dropout)
    if USE_DROPOUT:
        dropout_offs = dropout_offs + index_start_n
        dropout_mask = tl.rand(dropout_seed, dropout_offs) > dropout_prob
        s_curr = tl.where(dropout_mask, s_curr, 0.0)

    # Accumulate o_scratch with same scale
    v = padded_load(
        v_ptrs + (index_start_n + offs_n)[:, None] * stride_vn + offs_d[None, :],
        index_start_n + offs_n,
        offs_d,
        PA0=PADDED_COLS,
        PA1=PADDED_HEADS,
        LA0=actual_seqlen_k,
        LA1=headdim,
    ).to(tl.float32)
    acc_o = tl.where(row_valid[:, None], acc_o * alpha[:, None], acc_o) + tl.dot(s_curr, v)

    # Update m and log-lse (keep m and me in sync, using natural log like TPU reference)
    valid_update = l_next > 0
    m_i = tl.where(valid_update, m_next, m_i)
    me_i = tl.where(valid_update, m_next + tl.log(l_next + 1e-30), me_i)

    return m_i, me_i, acc_o


@triton.autotune(
    configs=[
        # You can add/trim these to match your preferred tile-set
        Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 256, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 256, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ],
    key=[
        "CKSeq",
        "CQSeq",
        "DRuntime",
        "VARLEN",
        "USE_DROPOUT",
        "IS_CAUSAL",
        "BIAS_ON",
        "BLOCK_HEADDIM",
        "SLIDING",
        "USE_LAYOUT",  # NEW
        "MAX_DEGREE",  # NEW
        "FORCE_BLOCK_M",  # NEW
        "FORCE_BLOCK_N",  # NEW
    ],
    prune_configs_by={"early_config_prune": config_prune_blocksparse_kernel},
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["QSeq"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["KSeq"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _attn_blocksparse_fwd(
    q,
    k,
    v,
    B,
    softmax_scale,
    dropout_prob,
    dropout_seed,
    logits_soft_cap,
    softmax_aux,
    num_sinks,
    stride_qz,
    stride_qm,
    stride_qh,
    stride_kz,
    stride_kn,
    stride_kh,
    stride_vz,
    stride_vn,
    stride_vh,
    stride_oz,
    stride_om,
    stride_oh,
    stride_bz,
    stride_bm,
    stride_bh,
    nheads_q,
    num_repeats,
    window_left,
    window_right,
    QSeq,
    cum_seqlens_q,  # int32 [B+1] (or dummy)
    KSeq,
    cum_seqlens_k,  # int32 [B+1] (or dummy)
    max_seqlen_q_rounded,
    headdim,
    CQSeq,
    CKSeq,
    DRuntime,
    Po,
    M,
    # Layout
    layout_ptr,  # int32 [n_q_blocks, MAX_DEGREE] row-major
    degree_ptr,  # int32 [n_q_blocks]
    FORCE_BLOCK_M,  # runtime "hint" for autotune pruning; unused in kernel body
    FORCE_BLOCK_N,  # runtime "hint" for autotune pruning; unused in kernel body
    VARLEN: tl.constexpr,
    USE_DROPOUT: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_ON: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    BOOL_BIAS: tl.constexpr,
    USE_LAYOUT: tl.constexpr,  # NEW
    MAX_DEGREE: tl.constexpr,  # NEW
    BLOCK_HEADDIM: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids
    i_start_m = tl.program_id(0)
    off_zh = tl.program_id(1)
    off_head_q = off_zh % nheads_q
    off_head_kv = off_head_q // num_repeats
    off_z = off_zh // nheads_q

    # Variable-length setup
    if VARLEN:
        cu_q0 = tl.load(cum_seqlens_q + off_z)
        cu_q1 = tl.load(cum_seqlens_q + off_z + 1)
        cu_k0 = tl.load(cum_seqlens_k + off_z)
        cu_k1 = tl.load(cum_seqlens_k + off_z + 1)
        actual_seqlen_q = cu_q1 - cu_q0
        actual_seqlen_k = cu_k1 - cu_k0
        if i_start_m * BLOCK_M >= actual_seqlen_q:
            return
        cu_seq_start_q = cu_q0
        cu_seq_start_k = cu_k0
        off_z = 0
    else:
        actual_seqlen_q = QSeq
        actual_seqlen_k = KSeq
        cu_seq_start_q = 0
        cu_seq_start_k = 0

    # LN2: tl.constexpr = 1.44269504089
    # softmax_scale = softmax_scale * LN2

    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    fully_masked_lines = (actual_seqlen_q - actual_seqlen_k) if IS_CAUSAL else 0
    if IS_CAUSAL and fully_masked_lines >= (i_start_m + 1) * BLOCK_M:
        return

    # Base pointers
    q_ptrs = (
        q
        + off_z * stride_qz
        + off_head_q * stride_qh
        + cu_seq_start_q * stride_qm
        + (offs_m[:, None] * stride_qm + offs_d[None, :])
    )
    k_base = k + off_z * stride_kz + off_head_kv * stride_kh + cu_seq_start_k * stride_kn
    v_base = v + off_z * stride_vz + off_head_kv * stride_vh + cu_seq_start_k * stride_vn

    if BIAS_ON:
        bias_ptrs = (
            B
            + off_z * stride_bz
            + off_head_kv * stride_bh
            + cu_seq_start_q * stride_bm
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
    else:
        bias_ptrs = None

    if USE_DROPOUT:
        dropout_off = actual_seqlen_k * (cu_seq_start_q + actual_seqlen_q * (off_head_q + nheads_q * off_z))
        dropout_offs = dropout_off + offs_m[:, None] * actual_seqlen_k + offs_n[None, :]
    else:
        dropout_offs = None

    # Sinks pointer for this head
    softmax_aux_ptrs = softmax_aux + off_head_q * num_sinks if USE_SINKS else softmax_aux

    me_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Load Q tile
    pad_rows = (not EVEN_M) or (VARLEN and (i_start_m * BLOCK_M > actual_seqlen_q))
    q = padded_load(q_ptrs, offs_m, offs_d, PA0=pad_rows, PA1=PADDED_HEADS, LA0=actual_seqlen_q, LA1=headdim)

    # Decide which K tiles to visit
    if USE_LAYOUT:
        # n_k_blocks for validity checks
        n_k_blocks = (actual_seqlen_k + BLOCK_N - 1) // BLOCK_N
        # Row id in layout (one layout shared across batch/heads)
        q_block = i_start_m

        # Load degree once
        deg = tl.load(degree_ptr + q_block)  # int32

        # Iterate adjacency list [q_block, :]
        for li in range(0, MAX_DEGREE):
            valid_li = li < deg

            col_block = tl.load(layout_ptr + q_block * MAX_DEGREE + li, mask=valid_li, other=0)
            # Additional checks: col_block bounds
            in_bounds = (col_block >= 0) & (col_block < n_k_blocks)
            process = valid_li & in_bounds

            index_start_n = col_block * BLOCK_N
            pad_cols = (index_start_n + BLOCK_N > actual_seqlen_k) or VARLEN

            if process:
                m_i, me_i, acc_o = _attn_fwd_inner(
                    q,
                    m_i,
                    me_i,
                    k_base,
                    v_base,
                    bias_ptrs,
                    acc_o,
                    offs_m,
                    offs_n,
                    offs_d,
                    softmax_scale,
                    dropout_prob,
                    dropout_seed,
                    dropout_offs,
                    window_left,
                    window_right,
                    logits_soft_cap,
                    softmax_aux_ptrs,
                    num_sinks,
                    stride_kn,
                    stride_vn,
                    index_start_n,
                    actual_seqlen_q,
                    actual_seqlen_k,
                    headdim,
                    USE_DROPOUT=USE_DROPOUT,
                    IS_CAUSAL=IS_CAUSAL,
                    BIAS_ON=BIAS_ON,
                    BOOL_BIAS=BOOL_BIAS,
                    MASKED=True,
                    SLIDING=SLIDING,  # keep element-level sliding if desired
                    SOFTCAP=SOFTCAP,
                    USE_SINKS=USE_SINKS,
                    PADDED_COLS=pad_cols,
                    PADDED_HEADS=PADDED_HEADS,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
    else:
        # Dense or sliding-window "blocksparse" scanning over a contiguous range
        if SLIDING:
            q_start = i_start_m * BLOCK_M
            q_end = tl.minimum(q_start + BLOCK_M, actual_seqlen_q) - 1  # inclusive
            shift = actual_seqlen_k - actual_seqlen_q
            wr = 0 if IS_CAUSAL else window_right

            jmin_aligned = q_start - window_left
            jmax_aligned = q_end + wr
            if IS_CAUSAL:
                jmax_aligned = tl.minimum(jmax_aligned, q_end)

            jmin = jmin_aligned + shift
            jmax = jmax_aligned + shift

            jmin = tl.maximum(0, jmin)
            jmax = tl.minimum(actual_seqlen_k - 1, jmax)

            if jmin <= jmax:
                start_n = (jmin // BLOCK_N) * BLOCK_N
                end_n = ((jmax // BLOCK_N) + 1) * BLOCK_N
                for index_start_n in range(start_n, end_n, BLOCK_N):
                    pad_cols = (index_start_n + BLOCK_N > actual_seqlen_k) or VARLEN
                    m_i, me_i, acc_o = _attn_fwd_inner(
                        q,
                        m_i,
                        me_i,
                        k_base,
                        v_base,
                        bias_ptrs,
                        acc_o,
                        offs_m,
                        offs_n,
                        offs_d,
                        softmax_scale,
                        dropout_prob,
                        dropout_seed,
                        dropout_offs,
                        window_left,
                        window_right,
                        logits_soft_cap,
                        softmax_aux_ptrs,
                        num_sinks,
                        stride_kn,
                        stride_vn,
                        index_start_n,
                        actual_seqlen_q,
                        actual_seqlen_k,
                        headdim,
                        USE_DROPOUT=USE_DROPOUT,
                        IS_CAUSAL=IS_CAUSAL,
                        BIAS_ON=BIAS_ON,
                        BOOL_BIAS=BOOL_BIAS,
                        MASKED=True,
                        SLIDING=SLIDING,
                        SOFTCAP=SOFTCAP,
                        USE_SINKS=USE_SINKS,
                        PADDED_COLS=pad_cols,
                        PADDED_HEADS=PADDED_HEADS,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )
            else:
                # Early-out: no valid keys for this Q tile
                offs_m2 = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
                lse_ptrs = M + off_zh * max_seqlen_q_rounded + offs_m2
                tl.store(lse_ptrs, me_i)  # keep -inf
                offs_d2 = tl.arange(0, BLOCK_HEADDIM)
                out_ptrs = (
                    Po
                    + off_z * stride_oz
                    + off_head_q * stride_oh
                    + cu_seq_start_q * stride_om
                    + (offs_m2[:, None] * stride_om + offs_d2[None, :])
                )
                tl.store(
                    out_ptrs,
                    tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32),
                    mask=(offs_m2[:, None] < actual_seqlen_q) & (offs_d2[None, :] < headdim),
                )
                return
        else:
            # Dense fallback (same as original dense end_n path)
            if IS_CAUSAL:
                end_n = tl.minimum(actual_seqlen_k - actual_seqlen_q + (i_start_m + 1) * BLOCK_M, actual_seqlen_k)
                if end_n < 0:
                    return
            else:
                end_n = actual_seqlen_k
            next_start_n = 0
            nb_full_blocks = end_n // BLOCK_N
            for _ in range(0, nb_full_blocks):
                m_i, me_i, acc_o = _attn_fwd_inner(
                    q,
                    m_i,
                    me_i,
                    k_base,
                    v_base,
                    bias_ptrs,
                    acc_o,
                    offs_m,
                    offs_n,
                    offs_d,
                    softmax_scale,
                    dropout_prob,
                    dropout_seed,
                    dropout_offs,
                    window_left,
                    window_right,
                    logits_soft_cap,
                    softmax_aux_ptrs,
                    num_sinks,
                    stride_kn,
                    stride_vn,
                    next_start_n,
                    actual_seqlen_q,
                    actual_seqlen_k,
                    headdim,
                    USE_DROPOUT=USE_DROPOUT,
                    IS_CAUSAL=IS_CAUSAL,
                    BIAS_ON=BIAS_ON,
                    BOOL_BIAS=BOOL_BIAS,
                    MASKED=False,
                    SLIDING=False,
                    SOFTCAP=SOFTCAP,
                    USE_SINKS=USE_SINKS,
                    PADDED_COLS=False,
                    PADDED_HEADS=PADDED_HEADS,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
                next_start_n += BLOCK_N
            if next_start_n < end_n:
                for index_start_n in range(next_start_n, end_n, BLOCK_N):
                    pad_cols = (index_start_n + BLOCK_N > actual_seqlen_k) or VARLEN
                    m_i, me_i, acc_o = _attn_fwd_inner(
                        q,
                        m_i,
                        me_i,
                        k_base,
                        v_base,
                        bias_ptrs,
                        acc_o,
                        offs_m,
                        offs_n,
                        offs_d,
                        softmax_scale,
                        dropout_prob,
                        dropout_seed,
                        dropout_offs,
                        window_left,
                        window_right,
                        logits_soft_cap,
                        softmax_aux_ptrs,
                        num_sinks,
                        stride_kn,
                        stride_vn,
                        index_start_n,
                        actual_seqlen_q,
                        actual_seqlen_k,
                        headdim,
                        USE_DROPOUT=USE_DROPOUT,
                        IS_CAUSAL=IS_CAUSAL,
                        BIAS_ON=BIAS_ON,
                        BOOL_BIAS=BOOL_BIAS,
                        MASKED=True,
                        SLIDING=False,
                        SOFTCAP=SOFTCAP,
                        USE_SINKS=USE_SINKS,
                        PADDED_COLS=pad_cols,
                        PADDED_HEADS=PADDED_HEADS,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )

    valid_rows = me_i > (-float("inf"))

    # Optional causal zeroing before normalization
    if IS_CAUSAL and fully_masked_lines > i_start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)

    # Normalize with exp(m - me) in natural log space (like TPU reference); scrub invalid rows
    if USE_DROPOUT:
        o_scale_raw = tl.exp(m_i - me_i) / (1 - dropout_prob)
    else:
        o_scale_raw = tl.exp(m_i - me_i)
    o_scale = tl.where(valid_rows, o_scale_raw, 0.0)
    acc_o = tl.where(valid_rows[:, None], acc_o * o_scale[:, None], 0.0)

    if IS_CAUSAL and fully_masked_lines > i_start_m * BLOCK_M:
        acc_o = tl.where(offs_m[:, None] < fully_masked_lines, 0, acc_o)

    # Store LSE (me_i) and O
    offs_m = i_start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    lse_ptrs = M + off_zh * max_seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, me_i)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Po
        + off_z * stride_oz
        + off_head_q * stride_oh
        + cu_seq_start_q * stride_om
        + (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < actual_seqlen_q) & (offs_d[None, :] < headdim))


def elem_strides_from_shape(shape):
    # Row-major (C-order) element strides
    strides = [0] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return tuple(strides)


def _blocksparse_fwd_attention_kernel_call(
    q: Float[Array, "batch seq_len_q num_heads head_dim"],
    k: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    *,
    # Optional masks/bias
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    # Flash extras
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    # Sparsity controls
    sliding_window: int | tuple[int, int] | None = None,
    layout: Int[Array, "n_q_blocks max_degree"] | None = None,  # block indices (>=0)
    degree: Int[Array, "n_q_blocks"] | None = None,
    max_degree: int = 0,  # needed when layout is provided
    # Tiling controls (optional, to match your layout block size)
    force_block_m: int = 0,
    force_block_n: int = 0,
):
    """
    Forward launcher for blocksparse (block-sparse + Flash) attention.

    If layout is provided, the kernel visits only those K blocks per Q block.
    If sliding_window is provided, it additionally masks within tiles.
    If neither is provided, falls back to dense FlashAttention behavior.
    """
    # Sliding-window parsing
    if sliding_window is None:
        window_left, window_right, sliding_flag = 0, 0, False
    else:
        if isinstance(sliding_window, int):
            window_left = int(sliding_window)
            window_right = 0 if causal else int(sliding_window)
        else:
            wl, wr = sliding_window
            window_left, window_right = int(wl), int(wr)
        assert window_left >= 0 and window_right >= 0
        sliding_flag = (window_left > 0) or (window_right > 0)

    # Softcap parsing
    if logits_soft_cap is None:
        logits_soft_cap_val, softcap_flag = 0.0, False
    else:
        logits_soft_cap_val, softcap_flag = float(logits_soft_cap), True

    # Sinks parsing
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

    # Layout parsing
    use_layout = layout is not None
    if use_layout:
        assert degree is not None, "When passing layout, you must also pass degree"
        assert max_degree > 0, "max_degree must be > 0 when using layout"
        assert layout.dtype == jnp.int32 and degree.dtype == jnp.int32

    varlen_from_cu = (cum_seqlens_q is not None) and (cum_seqlens_k is not None)

    if varlen_from_cu:
        assert cum_seqlens_q.dtype == jnp.int32 and cum_seqlens_k.dtype == jnp.int32
        batch = q.shape[0]
        QSeq_max = int(q.shape[1])
        KSeq_max = int(k.shape[1])
        nheads_q = q.shape[2]
        nheads_kv = k.shape[2]
        head_dim = q.shape[3]
        assert nheads_q % nheads_kv == 0
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype in [jnp.float16, jnp.bfloat16]

        # Pack varlen
        q_packed = attention_pack_from_cu_static(q, cum_seqlens_q, max_tokens=batch * QSeq_max)
        k_packed = attention_pack_from_cu_static(k, cum_seqlens_k, max_tokens=batch * KSeq_max)
        v_packed = attention_pack_from_cu_static(v, cum_seqlens_k, max_tokens=batch * KSeq_max)

        qz, qm, qh, _ = get_strides(q_packed.shape)
        kz, kn, kh, _ = get_strides(k_packed.shape)
        vz, vn, vh, _ = get_strides(v_packed.shape)
        oz, om, oh, _ = get_strides(q_packed.shape)

        if bias is not None:
            raise ValueError("Bias + VARLEN requires packed bias; not implemented here.")

        softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
        BOOL_BIAS = False

        max_seqlen_q = QSeq_max
        max_seqlen_k = KSeq_max
        max_seqlen_q_rounded = math.ceil(max_seqlen_q / 128) * 128
        BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
        PADDED_HEADS = BLOCK_HEADDIM > head_dim
        num_repeats = nheads_q // nheads_kv

        # Validate layout shape against block_m if forced
        if use_layout and force_block_m > 0:
            n_q_blocks = (max_seqlen_q + force_block_m - 1) // force_block_m
            assert layout.shape[0] == n_q_blocks, f"layout rows {layout.shape[0]} != n_q_blocks {n_q_blocks}"
            assert layout.shape[1] == max_degree, f"layout cols {layout.shape[1]} != max_degree {max_degree}"
            assert degree.shape[0] == n_q_blocks

        metaparams = dict(
            VARLEN=True,
            USE_DROPOUT=(dropout_prob > 0),
            IS_CAUSAL=causal,
            BIAS_ON=False,
            SLIDING=sliding_flag,
            SOFTCAP=softcap_flag,
            USE_SINKS=use_sinks,
            BOOL_BIAS=BOOL_BIAS,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            PADDED_HEADS=PADDED_HEADS,
            USE_LAYOUT=use_layout,
            MAX_DEGREE=max_degree if use_layout else 1,
        )

        out_shape = [
            jax.ShapeDtypeStruct(q_packed.shape, q_packed.dtype),
            jax.ShapeDtypeStruct((batch, nheads_q, max_seqlen_q_rounded), jnp.float32),
        ]

        out, lse = triton_call(
            q_packed,
            k_packed,
            v_packed,
            jnp.zeros((1,), q.dtype),
            softmax_scale,
            dropout_prob,
            dropout_seed if dropout_seed is not None else jnp.zeros((1,), q.dtype),
            logits_soft_cap_val,
            softmax_aux_tensor,
            num_sinks_val,
            qz,
            qm,
            qh,
            kz,
            kn,
            kh,
            vz,
            vn,
            vh,
            oz,
            om,
            oh,
            0,
            0,
            0,
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
            max_seqlen_q // 128,
            max_seqlen_k // 128,
            dtype_index(q_packed),
            # extra args
            jnp.zeros((1,), jnp.int32) if not use_layout else layout,
            jnp.zeros((1,), jnp.int32) if not use_layout else degree,
            int(force_block_m),
            int(force_block_n),
            kernel=_attn_blocksparse_fwd,
            out_shape=out_shape,
            grid=lambda META: (triton.cdiv(max_seqlen_q, META["BLOCK_M"]), batch * nheads_q),
            name="triton::ops::_attn_blocksparse_fwd_varlen",
            **metaparams,
            FORCE_BLOCK_M=int(force_block_m),
            FORCE_BLOCK_N=int(force_block_n),
        )

        out_unpacked = attention_unpack_with_static_shape(out, cum_seqlens_q, batch, QSeq_max)
        return out_unpacked, lse

    # Dense (non-packed) path
    batch, QSeq, nheads_q, head_dim = q.shape
    _, KSeq, nheads_kv, _ = k.shape
    expected_kv_shape = (batch, KSeq, nheads_kv, head_dim)
    assert k.shape == expected_kv_shape
    assert v.shape == expected_kv_shape
    assert nheads_q % nheads_kv == 0
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [jnp.float16, jnp.bfloat16]

    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    BOOL_BIAS = False
    if attention_mask is not None:
        assert bias is None, "Use either attention_mask (bool) or bias, not both."
        BOOL_BIAS = True
        bias = attention_mask.astype(jnp.bool_)

    bz, bh, bm = calc_bias_strides(bias, batch, nheads_q, QSeq, KSeq)

    max_seqlen_q_rounded = math.ceil(QSeq / 128) * 128
    BLOCK_HEADDIM = max(triton.next_power_of_2(head_dim), 16)
    PADDED_HEADS = BLOCK_HEADDIM > head_dim
    num_repeats = nheads_q // nheads_kv

    # Validate layout shape against forced block_m
    if use_layout and force_block_m > 0:
        n_q_blocks = (QSeq + force_block_m - 1) // force_block_m
        assert layout.shape[0] == n_q_blocks, f"layout rows {layout.shape[0]} != n_q_blocks {n_q_blocks}"
        assert layout.shape[1] == max_degree, f"layout cols {layout.shape[1]} != max_degree {max_degree}"
        assert degree.shape[0] == n_q_blocks

    qz, qm, qh, _ = elem_strides_from_shape(q.shape)
    oz, om, oh, _ = elem_strides_from_shape(q.shape)
    kz, kn, kh, _ = elem_strides_from_shape(k.shape)
    vz, vn, vh, _ = elem_strides_from_shape(v.shape)

    metaparams = dict(
        VARLEN=False,
        USE_DROPOUT=(dropout_prob > 0),
        IS_CAUSAL=causal,
        BIAS_ON=(bias is not None),
        SLIDING=sliding_flag,
        SOFTCAP=softcap_flag,
        USE_SINKS=use_sinks,
        BOOL_BIAS=BOOL_BIAS,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        PADDED_HEADS=PADDED_HEADS,
        USE_LAYOUT=use_layout,
        MAX_DEGREE=max_degree if use_layout else 1,
    )

    out_shape = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, nheads_q, max_seqlen_q_rounded), jnp.float32),
    ]

    out, lse = triton_call(
        q,
        k,
        v,
        bias if bias is not None else jnp.zeros((1,), q.dtype),
        softmax_scale,
        dropout_prob,
        dropout_seed if dropout_seed is not None else jnp.zeros((1,), q.dtype),
        logits_soft_cap_val,
        softmax_aux_tensor,
        num_sinks_val,
        qz,
        qm,
        qh,
        kz,
        kn,
        kh,
        vz,
        vn,
        vh,
        oz,
        om,
        oh,
        bz,
        bm,
        bh,
        nheads_q,
        num_repeats,
        window_left,
        window_right,
        QSeq,
        jnp.zeros((1,), jnp.int32),
        KSeq,
        jnp.zeros((1,), jnp.int32),
        max_seqlen_q_rounded,
        head_dim,
        QSeq // 128,
        KSeq // 128,
        dtype_index(q),
        # extra args
        jnp.zeros((1,), jnp.int32) if not use_layout else layout,
        jnp.zeros((1,), jnp.int32) if not use_layout else degree,
        int(force_block_m),
        int(force_block_n),
        kernel=_attn_blocksparse_fwd,
        out_shape=out_shape,
        grid=lambda META: (triton.cdiv(QSeq, META["BLOCK_M"]), batch * nheads_q),
        name="triton::ops::_attn_blocksparse_fwd",
        **metaparams,
        FORCE_BLOCK_M=int(force_block_m),
        FORCE_BLOCK_N=int(force_block_n),
    )

    return out, lse
