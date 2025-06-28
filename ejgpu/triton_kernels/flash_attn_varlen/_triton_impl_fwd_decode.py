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
def _fwd_kernel_stage1(
    q_ptr,
    k_ptr,
    v_ptr,
    sm_scale,
    k_ptr_new,
    v_ptr_new,
    cache_seqlens,
    cache_batch_idx,
    alibi_slopes,
    stride_qz,
    stride_qm,
    stride_qg,
    stride_qh,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kg,
    stride_kh,
    stride_kd,
    stride_vz,
    stride_vn,
    stride_vg,
    stride_vh,
    stride_vd,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_kn_z,
    stride_kn_n,
    stride_kn_g,
    stride_kn_h,
    stride_kn_d,
    stride_vn_z,
    stride_vn_n,
    stride_vn_g,
    stride_vn_h,
    stride_vn_d,
    stride_az,
    stride_ah,
    MAX_SEQLENS_Q,
    MAX_SEQLENS_K,
    MAX_SEQLENS_N,
    BLOCK_N_PER_SPLIT,
    o_ptr_k,
    metadata_ptr,
    q_heads: tl.constexpr,
    kv_heads: tl.constexpr,
    GroupQuery: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    BOUNDS_CHECKS_N: tl.constexpr,
    USE_CACHE_SEQLENs: tl.constexpr,
    USE_CACHE_BATCH_IDX: tl.constexpr,
    NEW_KV: tl.constexpr,
    IS_GQA: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DIM != BLOCK_DIM
    if PADDED_HEAD:
        d_mask = tl.arange(0, BLOCK_DIM) < ACTUAL_BLOCK_DIM

    start_m = tl.program_id(0)
    off_zhg = tl.program_id(1)
    off_z = off_zhg // (q_heads * GroupQuery)
    off_h_q = (off_zhg // GroupQuery) % q_heads
    off_g_q = off_zhg % GroupQuery
    splitk_idx = tl.program_id(2)

    if USE_CACHE_BATCH_IDX:
        cache_batch_idx = tl.load(cache_batch_idx + off_z)
    else:
        cache_batch_idx = off_z

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    lo = splitk_idx * BLOCK_N_PER_SPLIT
    if USE_CACHE_SEQLENs:
        cache_seqlen_last_idx = tl.load(cache_seqlens + off_z)
        if NEW_KV:
            kv_len = cache_seqlen_last_idx + MAX_SEQLENS_N
        else:
            kv_len = cache_seqlen_last_idx
    else:
        kv_len = MAX_SEQLENS_K
    hi = tl.minimum((splitk_idx + 1) * BLOCK_N_PER_SPLIT, kv_len)

    HEAD_RATIO: tl.constexpr = q_heads // kv_heads
    if IS_GQA:
        k_head_idx = off_h_q // HEAD_RATIO
        v_head_idx = k_head_idx
    else:
        k_head_idx = off_h_q
        v_head_idx = off_h_q
    k_base = k_ptr + k_head_idx * stride_kh + cache_batch_idx * stride_kz + off_g_q * stride_kg
    v_base = v_ptr + v_head_idx * stride_vh + cache_batch_idx * stride_vz + off_g_q * stride_vg

    if NEW_KV:
        knew_base = k_ptr_new + k_head_idx * stride_kn_h + off_z * stride_kn_z + off_g_q * stride_kn_g

        if USE_CACHE_SEQLENs:
            start_idx = tl.load(cache_seqlens + off_z)
        else:
            start_idx = MAX_SEQLENS_K - MAX_SEQLENS_N

        for i in range(0, MAX_SEQLENS_N, BLOCK_N):
            k_new_block = tl.load(
                knew_base
                + tl.arange(0, BLOCK_DIM)[:, None] * stride_kn_d
                + (tl.arange(0, BLOCK_N) + i)[None, :] * stride_kn_n,
                mask=(tl.arange(0, BLOCK_N)[None, :] + i < MAX_SEQLENS_N)
                & (tl.arange(0, BLOCK_DIM)[:, None] < ACTUAL_BLOCK_DIM),
                other=0,
            )

            tl.store(
                k_base
                + tl.arange(0, BLOCK_DIM)[:, None] * stride_kd
                + (tl.arange(0, BLOCK_N) + i + start_idx)[None, :] * stride_kn,
                k_new_block,
                mask=(tl.arange(0, BLOCK_N)[None, :] + i < MAX_SEQLENS_N)
                & (tl.arange(0, BLOCK_DIM)[:, None] < ACTUAL_BLOCK_DIM),
            )

        vnew_base = v_ptr_new + v_head_idx * stride_vn_h + off_z * stride_vn_z + off_g_q * stride_vn_g
        for i in range(0, MAX_SEQLENS_N, BLOCK_N):
            v_new_block = tl.load(
                vnew_base
                + (tl.arange(0, BLOCK_N) + i)[:, None] * stride_vn_n
                + tl.arange(0, BLOCK_DIM)[None, :] * stride_vn_d,
                mask=(tl.arange(0, BLOCK_N)[:, None] + i < MAX_SEQLENS_N)
                & (tl.arange(0, BLOCK_DIM)[None, :] < ACTUAL_BLOCK_DIM),
                other=0,
            )

            tl.store(
                v_base
                + (tl.arange(0, BLOCK_N) + i + start_idx)[:, None] * stride_vn
                + tl.arange(0, BLOCK_DIM)[None, :] * stride_vd,
                v_new_block,
                mask=(tl.arange(0, BLOCK_N)[:, None] + i < MAX_SEQLENS_N)
                & (tl.arange(0, BLOCK_DIM)[None, :] < ACTUAL_BLOCK_DIM),
            )

    Q_block_ptr = tl.make_block_ptr(
        base=q_ptr + off_h_q * stride_qh + off_z * stride_qz + off_g_q * stride_qg,
        shape=(MAX_SEQLENS_Q, ACTUAL_BLOCK_DIM),
        strides=(stride_qm, stride_qd),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DIM),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        base=k_base,
        shape=(ACTUAL_BLOCK_DIM, hi),
        strides=(stride_kd, stride_kn),
        offsets=(0, lo),
        block_shape=(BLOCK_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_base,
        shape=(hi, ACTUAL_BLOCK_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(lo, 0),
        block_shape=(BLOCK_N, BLOCK_DIM),
        order=(1, 0),
    )

    K_scale_shift_block_ptr = None
    V_scale_shift_block_ptr = None

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504
    q = tl.load(tl.advance(Q_block_ptr, (0, 0)), boundary_check=(0,))
    q = (q * qk_scale).to(q.dtype)
    if PADDED_HEAD:
        q = tl.where(d_mask[None, :], q, 0.0)
    for start_n in range(lo, hi, BLOCK_N):
        k, v = load_k_v_group(
            K_block_ptr,
            V_block_ptr,
            K_scale_shift_block_ptr,
            V_scale_shift_block_ptr,
            BOUNDS_CHECKS_N,
            1,
            BLOCK_DIM,
            ACTUAL_BLOCK_DIM,
            q_ptr.dtype.element_ty,
            0,
        )
        if PADDED_HEAD:
            k = tl.where(d_mask[:, None], k, 0.0)
            v = tl.where(d_mask[None, :], v, 0.0)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if USE_ALIBI:
            row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)
            relative_pos = row_idx[:, None] + kv_len - (MAX_SEQLENS_Q + col_idx[None, :])
            relative_pos = tl.abs(relative_pos)
            alibi_bias = -1 * alibi_slope * relative_pos
            qk += alibi_bias * 1.44269504

        if IS_CAUSAL:
            row_idx = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            col_idx = start_n + tl.arange(0, BLOCK_N)

            col_offset = MAX_SEQLENS_Q - kv_len
            causal_mask = row_idx[:, None] >= (col_offset + col_idx[None, :])

            qk = tl.where(causal_mask, qk, float("-inf"))

        if BOUNDS_CHECKS_N:
            qk = tl.where(tl.arange(0, BLOCK_N) < hi - start_n, qk, float("-inf"))

        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        if IS_CAUSAL:
            alpha = tl.math.exp2(tl.where(m_i > float("-inf"), m_i - m_i_new, float("-inf")))
        else:
            alpha = tl.math.exp2(m_i - m_i_new)
        if IS_CAUSAL:
            qk = tl.where(qk > float("-inf"), qk - m_i_new[:, None], float("-inf"))
        else:
            qk = qk - m_i_new[:, None]

        p = tl.math.exp2(qk)

        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        p = p.to(q_ptr.dtype.element_ty)

        acc *= alpha[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    O_block_ptr = tl.make_block_ptr(
        base=o_ptr_k + off_zhg * stride_osk_zhg + splitk_idx * stride_osk_s,
        shape=(MAX_SEQLENS_Q, BLOCK_DIM),
        strides=(stride_osk_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DIM),
        order=(1, 0),
    )
    tl.store(
        tl.advance(O_block_ptr, (0, 0)),
        acc,
        boundary_check=(0,),
    )
    metadata_ptrs = (
        metadata_ptr + off_zhg * stride_mzhg + splitk_idx * stride_ms + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    )
    tl.store(metadata_ptrs, m_i)
    tl.store(metadata_ptrs + stride_m2, l_i)


@triton.jit
def load_k_v_group(
    K_block_ptr,
    V_block_ptr,
    K_scale_shift_block_ptr,
    V_scale_shift_block_ptr,
    BOUNDS_CHECKS_N: tl.constexpr,
    PACKED_PER_VAL: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    dtype: tl.constexpr,
    group_id: tl.constexpr,
):
    K_block_ptr = tl.advance(K_block_ptr, (ACTUAL_BLOCK_DIM * group_id, 0))
    V_block_ptr = tl.advance(V_block_ptr, (0, ACTUAL_BLOCK_DIM * group_id))
    k = tl.load(K_block_ptr, boundary_check=(1,) if BOUNDS_CHECKS_N else ())
    v = tl.load(V_block_ptr, boundary_check=(0,) if BOUNDS_CHECKS_N else ())

    return k, v


@triton.jit
def cast_uint32_to_half2(scale_shift):
    scale = scale_shift & 0xFFFF
    shift = scale_shift >> 16
    scale = scale.to(tl.uint16).to(tl.float16, bitcast=True)
    shift = shift.to(tl.uint16).to(tl.float16, bitcast=True)
    return scale, shift


@triton.jit
def dequantize(
    x_,
    scale,
    shift,
    PACKED_PER_VAL: tl.constexpr = 8,
):
    BLOCK_N: tl.constexpr = x_.shape[0]
    BLOCK_DMODEL_PACKED: tl.constexpr = x_.shape[1]
    offsets = tl.arange(0, PACKED_PER_VAL) * 4
    quant_offset = x_[:, None, :] >> offsets[None, :, None]
    quant_offset = tl.view(quant_offset, (BLOCK_N, BLOCK_DMODEL_PACKED * PACKED_PER_VAL))
    quant_offset = (quant_offset & 0xF).to(tl.uint16).to(tl.float16, bitcast=True)
    quant_offset = (quant_offset * 32768.0).to(tl.float16)
    scale_512 = scale * 512

    dequant = quant_offset * scale_512 + shift
    return dequant


@triton.jit
def _fwd_kernel_stage2(
    o_ptr_k,
    metadata_ptr,
    stride_osk_zhg,
    stride_osk_s,
    stride_osk_m,
    stride_osk_k,
    stride_mzhg,
    stride_m2,
    stride_ms,
    stride_mm,
    stride_oz,
    stride_oh,
    stride_og,
    stride_om,
    stride_lse_zhg,
    out_ptr,
    lse_ptr,
    H: tl.constexpr,
    G: tl.constexpr,
    split_k: tl.constexpr,
    splitK_pow2: tl.constexpr,
    use_mask: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    off_zhg = tl.program_id(0)
    off_z = off_zhg // (H * G)
    off_h = (off_zhg // G) % H
    off_g = off_zhg % G
    off_m = tl.program_id(1)
    off_k = tl.program_id(2)

    spk_idx = tl.arange(0, splitK_pow2)
    kidx = tl.arange(0, BLOCK_SIZE)

    metadata_ptrs = metadata_ptr + stride_mzhg * off_zhg + spk_idx * stride_ms + off_m * stride_mm

    o_ptr = (
        o_ptr_k
        + off_zhg * stride_osk_zhg
        + stride_osk_m * off_m
        + off_k * BLOCK_SIZE
        + stride_osk_s * spk_idx[:, None]
        + kidx[None, :] * stride_osk_k
    )

    if use_mask:
        spk_mask = spk_idx < split_k
        l_m = tl.load(metadata_ptrs, mask=spk_mask, other=float("-inf"))
        l_sum = tl.load(metadata_ptrs + stride_m2, mask=spk_mask, other=0.0)
        acc = tl.load(o_ptr, mask=spk_mask[:, None], other=0.0)
    else:
        l_m = tl.load(metadata_ptrs)
        l_sum = tl.load(metadata_ptrs + stride_m2)
        acc = tl.load(o_ptr)

    g_m = tl.max(l_m, axis=0)

    if IS_CAUSAL:
        l_m_offset = l_m - g_m
        alpha = tl.where(l_m_offset > float("-inf"), tl.math.exp2(l_m_offset), 0.0)
    else:
        alpha = tl.math.exp2(l_m - g_m)

    l_sum *= alpha
    g_sum = tl.sum(l_sum, axis=0)
    acc = acc * alpha[:, None]

    if IS_CAUSAL:
        g_sum_safe = tl.where(g_sum > 0, g_sum, 1.0)
        acc_out = tl.sum(acc, axis=0) / g_sum_safe
    else:
        acc_out = tl.sum(acc, axis=0) / g_sum

    out_ptrs = (
        out_ptr
        + stride_oz * off_z
        + stride_oh * off_h
        + stride_og * off_g
        + stride_om * off_m
        + off_k * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)
    )
    tl.store(out_ptrs, acc_out)

    l_ptrs = lse_ptr + off_zhg * stride_lse_zhg + off_m
    if IS_CAUSAL:
        lse = tl.where(g_sum > 0, (g_m + tl.math.log2(g_sum)) / 1.44269504, g_m)
        tl.store(l_ptrs, lse)
    else:
        tl.store(l_ptrs, (g_m + tl.math.log2(g_sum)) / 1.44269504)


def get_split_k(B: int, G: int, H: int, Mk: int) -> int:
    """Heuristic for the number of splits"""
    bh = max(B * H, 1)
    split_k = max(Mk, 1024) // bh
    max_chunk_size = 64
    while split_k > 0 and Mk / split_k < max_chunk_size:
        split_k = split_k // 2
    while B * H * G * split_k >= 1024:
        split_k = split_k // 2
    split_k = min(split_k, 512)
    split_k = max(split_k, 1)
    return split_k
