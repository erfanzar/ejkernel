# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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
import triton
import triton.language as tl


@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    q,
    k_cache,
    v_cache,
    sm_scale,
    slot_mapping,
    b_seqlen,
    stride_slot_mapping_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    logits_out,
    KVGroups: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LOGIT_CAP: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head = cur_head // KVGroups
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < BLOCK_DIMK
    mask_dv = offs_dv < BLOCK_DIMV
    cur_batch_seq_len = tl.load(b_seqlen + cur_batch)
    cur_batch_req_idx = cur_batch
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(q + off_q, mask=mask_d, other=0.0)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                slot_mapping + stride_slot_mapping_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[None, :]
            k = tl.load(k_cache + offs_buf_k, mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]), other=0.0)
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale
            if LOGIT_CAP > 0:
                qk = LOGIT_CAP * tanh(qk / LOGIT_CAP)
            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(v_cache + offs_buf_v, mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]), other=0.0)
            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)
            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max
        offs_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv
        tl.store(logits_out + offs_mid_o, acc / e_sum, mask=(mask_dv))
        offs_mid_o_1 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + BLOCK_DIMV
        tl.store(logits_out + offs_mid_o_1, e_max + tl.log(e_sum))


@triton.jit
def _fwd_grouped_kernel_stage1(
    q,
    k_cache,
    v_cache,
    sm_scale,
    slot_mapping,
    b_seqlen,
    stride_slot_mapping_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    logits_out,
    KVGroups: tl.constexpr,
    Qheads: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    LOGIT_CAP: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(KVGroups, BLOCK_H)
    split_kv_id = tl.program_id(2)
    if KVGroups > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = KVGroups
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < Qheads)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < BLOCK_DIMK
    mask_dv = offs_dv < BLOCK_DIMV
    cur_batch_seq_len = tl.load(b_seqlen + cur_batch)
    cur_batch_req_idx = cur_batch
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :]
    q = tl.load(q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < BLOCK_DIMK
        off_qpe = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        qpe = tl.load(q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0)
    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)
    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                slot_mapping + stride_slot_mapping_b * cur_batch_req_idx + offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[:, None]
            k = tl.load(k_cache + offs_buf_k, mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]), other=0.0)
            qk = tl.dot(q, k.to(q.dtype))
            if BLOCK_DPE > 0:
                offs_buf_kpe = kv_loc[None, :] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_dpe[:, None]
                kpe = tl.load(
                    k_cache + offs_buf_kpe, mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]), other=0.0
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale
            if LOGIT_CAP > 0:
                qk = LOGIT_CAP * tanh(qk / LOGIT_CAP)
            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))
            offs_buf_v = kv_loc[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = tl.load(v_cache + offs_buf_v, mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]), other=0.0)
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)
            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max
        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv[None, :]
        )
        tl.store(logits_out + offs_mid_o, acc / e_sum[:, None], mask=(mask_h[:, None]) & (mask_dv[None, :]))
        offs_mid_o_1 = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + BLOCK_DIMV
        tl.store(logits_out + offs_mid_o_1, e_max + tl.log(e_sum), mask=mask_h)


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    b_seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    o,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_batch_seq_len = tl.load(b_seqlen + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < BLOCK_DIMV
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + BLOCK_DIMV
    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(o + cur_batch * stride_obs + cur_head * stride_oh + offs_d, acc / e_sum, mask=mask_d)
