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

from ...utils import is_cdna, is_rdna


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


AUTOTUNE_KEYS = [
    "IS_CAUSAL",
    "dropout_p",
    "MAX_SEQLENS_Q",
    "MAX_SEQLENS_K",
    "ACTUAL_BLOCK_DMODEL",
    "VARLEN",
    "q_heads",
    "kv_heads",
]


def get_cdna_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
    ], AUTOTUNE_KEYS


def get_rdna_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "PRE_LOAD_V": False}, num_stages=1, num_warps=2),
    ], AUTOTUNE_KEYS


def get_default_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "PRE_LOAD_V": False}, num_stages=1, num_warps=4),
    ], AUTOTUNE_KEYS


def get_autotune_configs():
    if is_rdna():
        return get_rdna_autotune_configs()
    elif is_cdna():
        return get_cdna_autotune_configs()
    else:
        return get_default_autotune_configs()


autotune_configs, autotune_keys = get_autotune_configs()


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    k_ptrs,
    v_ptrs,
    bias_ptrs,
    stride_kn,
    stride_vk,
    stride_bn,
    start_m,
    actual_seqlen_k,
    actual_seqlen_q,
    dropout_p,
    philox_seed,
    batch_philox_offset,
    block_min,
    block_max,
    offs_n_causal,
    n_extra_tokens,
    alibi_slope,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OFFS_M: tl.constexpr,
    OFFS_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
    MASK_STEPS: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    sm_scale: tl.constexpr,
):
    for start_n in range(block_min, block_max, BLOCK_N):
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DIM)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DIM, actual_seqlen_k)
        if PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DIM)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if MASK_STEPS:
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        qk_scaled = qk * sm_scale
        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk_scaled = tl.where(causal_mask, qk_scaled, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            qk_scaled += load_fn(
                bias_ptrs,
                OFFS_M,
                bias_offs_n,
                actual_seqlen_q,
                actual_seqlen_k,
            )
        if alibi_slope is not None:
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(
                alibi_slope,
                actual_seqlen_q,
                actual_seqlen_k,
                global_m_positions,
                global_n_positions,
            )
            qk_scaled += alibi_block
        m_ij = tl.maximum(m_i, tl.max(qk_scaled, 1))
        q_shifted = qk_scaled - m_ij[:, None]
        p = tl.math.exp(q_shifted)
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            p = tl.where(keep, p, 0.0)
        m_diff = m_i - m_ij
        alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DIM)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
    return acc, l_i, m_i


@triton.autotune(configs=autotune_configs, key=autotune_keys, use_cuda_graph=True)
@triton.jit
def attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    sm_scale,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_az,
    stride_ah,
    stride_lz,
    stride_lh,
    stride_lm,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset,
    alibi_slopes,
    o_ptr,
    l_ptr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    MAX_SEQLENS_Q: tl.constexpr,
    MAX_SEQLENS_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    USE_ALIBI: tl.constexpr,
    USE_BIAS: tl.constexpr,
    kv_heads: tl.constexpr,
    VARLEN: tl.constexpr,
    q_heads: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DIM)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        n_blocks = min(n_blocks, n_blocks_seqlen)
        if n_blocks <= 0:
            o_offset = o_ptr + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=o_ptr.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            l_offset = l_ptr + off_z * stride_lz + off_h_q * stride_lh + cu_seqlens_q_start * stride_lm
            l_ptrs = l_offset + offs_m * stride_lm
            le = tl.full([BLOCK_M], value=0.0, dtype=tl.float32)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, le, mask=l_ptrs_mask)
            return

    GROUP_SIZE: tl.constexpr = q_heads // kv_heads
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N

    PADDED_HEAD: tl.constexpr = ACTUAL_BLOCK_DIM != BLOCK_DIM
    q_offset = q_ptr + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = k_ptr + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = v_ptr + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        bias_offset = off_h_q * stride_bh
        bias_ptrs = b_ptr + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if ENABLE_DROPOUT:
        off_hz = off_z * q_heads + off_h_q
        batch_philox_offset = philox_offset + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DIM], dtype=tl.float32)
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DIM)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        masked_blocks = padded_block_k
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            block_min,
            block_max,
            0,
            0,
            alibi_slope,
            False,
            BLOCK_M,
            BLOCK_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            False,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            ACTUAL_BLOCK_DIM,
            sm_scale,
        )
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    if masked_blocks > 0:
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_ptrs,
            v_ptrs,
            bias_ptrs,
            stride_kn,
            stride_vk,
            stride_bn,
            start_m,
            seqlen_k,
            seqlen_q,
            dropout_p,
            philox_seed,
            batch_philox_offset,
            block_min,
            block_max,
            offs_n_causal,
            n_extra_tokens,
            alibi_slope,
            IS_CAUSAL,
            BLOCK_M,
            BLOCK_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            PRE_LOAD_V,
            True,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            ACTUAL_BLOCK_DIM,
            sm_scale,
        )
    l_recip = 1 / l_i[:, None]
    acc = acc * l_recip
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(o_ptr.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DIM,), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))

    l_offset = l_ptr + off_z * stride_lz + off_h_q * stride_lh + cu_seqlens_q_start * stride_lm
    l_ptrs = l_offset + offs_m * stride_lm
    softmax_lse = m_i + tl.math.log(l_i)

    if IS_CAUSAL:
        lse_mask = (start_m_idx + tl.arange(0, BLOCK_M)) < causal_start_idx
        softmax_lse = tl.where(lse_mask, 0.0, softmax_lse)

    overflow_size = end_m_idx - seqlen_q
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M,), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask)
    else:
        tl.store(l_ptrs, softmax_lse)
    o_offset = o_ptr + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DIM], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DIM)
    tl.store(o_ptrs, acc.to(o_ptr.dtype.element_ty), mask=o_ptrs_mask)
