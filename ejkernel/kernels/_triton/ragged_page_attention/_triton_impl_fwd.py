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

import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import triton_call


@triton.jit
def _ragged_paged_attn_fwd(
    q_ptr,
    kv_pages_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_q_lens_ptr,
    softmax_scale,
    logits_soft_cap,
    total_tokens,
    num_seqs,
    num_q_heads,
    num_kv_heads,
    pages_per_seq,
    head_dim,
    total_tokens_rounded,
    window_left,
    window_right,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    kv_stride_p,
    kv_stride_s,
    kv_stride_c,
    kv_stride_d,
    bt_stride_s,
    bt_stride_p,
    o_stride_m,
    o_stride_h,
    o_stride_d,
    lse_stride_h,
    lse_stride_m,
    o_ptr,
    lse_ptr,
    NUM_REPEATS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NPAGES: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_NUM_SEQS: tl.constexpr,
    PAGES_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < total_tokens

    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < head_dim

    q_ptrs = q_ptr + offs_m[:, None] * q_stride_m + pid_h * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    kv_head = pid_h // NUM_REPEATS
    k_combined_idx = 2 * kv_head
    v_combined_idx = 2 * kv_head + 1

    for seq_idx in tl.static_range(0, MAX_NUM_SEQS):
        seq_active = seq_idx < num_seqs

        q_start = tl.load(cu_q_lens_ptr + seq_idx, mask=seq_active, other=0)
        q_end = tl.load(cu_q_lens_ptr + seq_idx + 1, mask=seq_active, other=0)
        kv_len = tl.load(context_lens_ptr + seq_idx, mask=seq_active, other=0)
        q_len = q_end - q_start

        row_mask = mask_m & seq_active & (offs_m >= q_start) & (offs_m < q_end)
        row_pos = offs_m - q_start
        row_idx = (kv_len - q_len) + row_pos

        end_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE

        for p_blk in tl.static_range(0, PAGES_PER_SEQ, BLOCK_NPAGES):
            for j in tl.static_range(0, BLOCK_NPAGES):
                p_ind = p_blk + j
                page_active = (p_ind < end_pages) & seq_active

                page_id_ptr = block_tables_ptr + seq_idx * bt_stride_s + p_ind * bt_stride_p
                page_id = tl.load(page_id_ptr, mask=page_active, other=0)

                k_in_page = tl.arange(0, PAGE_SIZE)
                k_abs = p_ind * PAGE_SIZE + k_in_page
                k_valid = k_abs < kv_len

                if SLIDING:
                    left_bound = tl.maximum(row_idx - window_left, 0)
                    if IS_CAUSAL:
                        right_bound = row_idx
                    else:
                        rb = tl.minimum(row_idx + window_right, kv_len - 1)
                        right_bound = rb
                else:
                    left_bound = tl.zeros_like(row_idx)
                    if IS_CAUSAL:
                        right_bound = row_idx
                    else:
                        right_bound = tl.full_like(row_idx, kv_len - 1)

                allowed = (k_abs[None, :] >= left_bound[:, None]) & (k_abs[None, :] <= right_bound[:, None])
                s_mask = row_mask[:, None] & page_active & k_valid[None, :] & allowed

                base_page = kv_pages_ptr + page_id * kv_stride_p
                k_ptrs = (
                    base_page
                    + k_in_page[:, None] * kv_stride_s
                    + (k_combined_idx) * kv_stride_c
                    + offs_d[None, :] * kv_stride_d
                )
                v_ptrs = (
                    base_page
                    + k_in_page[:, None] * kv_stride_s
                    + (v_combined_idx) * kv_stride_c
                    + offs_d[None, :] * kv_stride_d
                )

                k_tile = tl.load(k_ptrs, mask=(k_valid[:, None] & d_mask[None, :] & page_active), other=0.0).to(
                    tl.float32
                )
                v_tile = tl.load(v_ptrs, mask=(k_valid[:, None] & d_mask[None, :] & page_active), other=0.0).to(
                    tl.float32
                )

                qk = tl.dot(q, tl.trans(k_tile)) * softmax_scale

                if SOFTCAP:
                    inv_cap = 1.0 / logits_soft_cap
                    qk = logits_soft_cap * tl.tanh(qk * inv_cap)

                neg_large = -1.0e30
                qk_masked = tl.where(s_mask, qk, neg_large)

                has_any_i32 = tl.max(s_mask.to(tl.int32), axis=1)
                has_any = has_any_i32 != 0

                m_curr = tl.max(qk_masked, axis=1)
                qk_minus_m = tl.where(has_any[:, None], qk_masked - m_curr[:, None], neg_large)
                s_curr = tl.exp(qk_minus_m)
                l_curr = tl.sum(s_curr, axis=1)
                qkv = tl.dot(s_curr, v_tile)

                m_next = tl.maximum(m_i, m_curr)
                m_next = tl.where(has_any, m_next, m_i)

                alpha = tl.where(has_any, tl.exp(m_i - m_next), 1.0)
                beta = tl.where(has_any, tl.exp(m_curr - m_next), 0.0)
                l_next = alpha * l_i + beta * l_curr

                o_num = (alpha[:, None] * l_i[:, None]) * acc + (beta[:, None] * qkv)
                den = tl.where(l_next[:, None] > 0, l_next[:, None], 1.0)
                o_new = o_num / den

                update_mask = row_mask & has_any
                m_i = tl.where(update_mask, m_next, m_i)
                l_i = tl.where(update_mask, l_next, l_i)
                acc = tl.where(update_mask[:, None], o_new, acc)

    o_ptrs = o_ptr + offs_m[:, None] * o_stride_m + pid_h * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & d_mask[None, :])

    lse_vals = m_i + tl.log(l_i)
    lse_ptrs = lse_ptr + pid_h * lse_stride_h + offs_m * lse_stride_m
    lse_mask = offs_m < total_tokens_rounded
    tl.store(lse_ptrs, lse_vals, mask=lse_mask)


def _contig_strides_3(shape):
    _M, H, D = shape
    return (H * D, D, 1)


def _contig_strides_4(shape):
    _P, S, C, D = shape
    return (S * C * D, C * D, D, 1)


def ragged_paged_attention_triton_call(
    queries: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    cu_q_lens: jax.Array,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    causal: bool = True,
    block_m: int = 128,
    block_npages: int = 2,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    assert queries.ndim == 3 and kv_pages.ndim == 4
    assert queries.dtype in (jnp.float16, jnp.bfloat16) and kv_pages.dtype == queries.dtype
    assert context_lens.dtype == jnp.int32 and block_tables.dtype == jnp.int32 and cu_q_lens.dtype == jnp.int32

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, combined_kv_heads, head_dim_kv = map(int, kv_pages.shape)
    assert head_dim == head_dim_kv and combined_kv_heads % 2 == 0
    num_kv_heads = combined_kv_heads // 2

    num_seqs, pages_per_seq = map(int, block_tables.shape)
    assert context_lens.shape[0] == num_seqs and cu_q_lens.shape[0] == num_seqs + 1
    assert num_q_heads % num_kv_heads == 0
    num_repeats = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

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

    BLOCK_M = int(block_m)
    BLOCK_NPAGES = int(block_npages)
    BLOCK_DMODEL = max(triton.next_power_of_2(head_dim), 16)
    total_tokens_rounded = int(math.ceil(total_tokens / 128) * 128)

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)
    lse_sh, lse_sm = total_tokens_rounded, 1

    out_shape = [
        jax.ShapeDtypeStruct(queries.shape, queries.dtype),
        jax.ShapeDtypeStruct((num_q_heads, total_tokens_rounded), jnp.float32),
    ]

    metaparams = dict(
        NUM_REPEATS=num_repeats,
        IS_CAUSAL=bool(causal),
        SLIDING=bool(sliding_flag),
        SOFTCAP=bool(softcap_flag),
        BLOCK_M=BLOCK_M,
        BLOCK_NPAGES=BLOCK_NPAGES,
        BLOCK_DMODEL=BLOCK_DMODEL,
        MAX_NUM_SEQS=num_seqs,
        PAGES_PER_SEQ=pages_per_seq,
        PAGE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out, _lse = triton_call(
        queries,
        kv_pages,
        block_tables,
        context_lens,
        cu_q_lens,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(total_tokens),
        int(num_seqs),
        int(num_q_heads),
        int(num_kv_heads),
        int(pages_per_seq),
        int(head_dim),
        int(total_tokens_rounded),
        int(window_left),
        int(window_right),
        int(q_sm),
        int(q_sh),
        int(q_sd),
        int(kv_sp),
        int(kv_ss),
        int(kv_sc),
        int(kv_sd),
        int(bt_ss),
        int(bt_sp),
        int(o_sm),
        int(o_sh),
        int(o_sd),
        int(lse_sh),
        int(lse_sm),
        kernel=_ragged_paged_attn_fwd,
        out_shape=out_shape,
        grid=lambda META: (triton.cdiv(total_tokens, META["BLOCK_M"]), num_q_heads),
        name="ejkernel::triton::ragged_page_attn",
        **metaparams,
    )
    return out
