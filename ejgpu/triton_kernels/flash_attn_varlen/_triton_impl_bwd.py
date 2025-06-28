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

import jax
import triton
import triton.language as tl
from eformer.callib import cdiv, triton_call
from jax import numpy as jnp

from ...utils import get_shape_from_layout, get_strides, get_strides_from_layout


@triton.jit
def _bwd_preprocess_use_o(
    o_ptr,
    do_ptr,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    cu_seqlens_q,
    max_seqlen_q,
    delta_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    SEQLENS_Q: tl.constexpr,
    q_heads: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_z = pid_bh // q_heads
    off_h = pid_bh % q_heads

    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        SEQLENS_Q = q_end - q_start
    else:
        q_start = 0
        SEQLENS_Q = max_seqlen_q
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DIM)
    mask_m = off_m < SEQLENS_Q
    mask_d = off_d < ACTUAL_BLOCK_DIM
    o_offset = o_ptr + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    do_offset = do_ptr + off_z * stride_oz + off_h * stride_oh + q_start * stride_om
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    o = tl.load(out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    delta_offset = delta_ptr + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale,
    q_offset,
    k_offset,
    v_offset,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    d_offset,
    l_offset,
    stride_qm,
    stride_qk,
    stride_kn,
    stride_kk,
    stride_vn,
    stride_vk,
    stride_deltam,
    SEQLENS_Q,
    SEQLENS_K,
    start_n,
    num_block_m,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DIM)
    mask_n = offs_n < SEQLENS_K
    mask_d = offs_d < ACTUAL_BLOCK_DIM
    kv_mask = mask_n[:, None] & mask_d[None, :]
    dv = tl.zeros([BLOCK_N, BLOCK_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DIM], dtype=tl.float32)
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    for start_m in range(0, num_block_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        mask_m = offs_m < SEQLENS_Q
        q_mask = mask_m[:, None] & mask_d[None, :]
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        if CAUSAL:
            col_offset = SEQLENS_Q - SEQLENS_K
            causal_mask = offs_m[:, None] >= (col_offset + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))
        l_ptrs = l_offset + offs_m * stride_deltam
        l_i = tl.load(l_ptrs, mask=mask_m)
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])
        p_mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(p_mask, p, 0.0)
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        dp = tl.dot(do, tl.trans(v))
        d_ptrs = d_offset + offs_m * stride_deltam
        Di = tl.load(d_ptrs, mask=mask_m)
        ds = (p * (dp - Di[:, None])) * sm_scale
        ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)
        dk += tl.dot(tl.trans(ds), q)
        if SEQUENCE_PARALLEL:
            dq = tl.dot(ds, k)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            dq += tl.dot(ds, k)
        tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=kv_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=kv_mask)


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    do_ptr,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_dq_all,
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
    stride_vn,
    stride_vk,
    stride_deltaz,
    stride_deltah,
    stride_deltam,
    q_heads,
    num_block_m,
    num_block_n,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    ACTUAL_BLOCK_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    off_hz = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_hz // q_heads
    off_h = off_hz % q_heads
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + off_z)
        q_end = tl.load(cu_seqlens_q + off_z + 1)
        k_start = tl.load(cu_seqlens_k + off_z)
        k_end = tl.load(cu_seqlens_k + off_z + 1)
        SEQLENS_Q = q_end - q_start
        SEQLENS_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        SEQLENS_Q = max_seqlen_q
        SEQLENS_K = max_seqlen_k
    q_offset = Q + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    k_offset = K + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    v_offset = V + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    do_offset = do_ptr + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    l_offset = L + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    d_offset = D + off_z * stride_deltaz + off_h * stride_deltah + q_start * stride_deltam
    dk_offset = DK + off_z * stride_kz + off_h * stride_kh + k_start * stride_kn
    dv_offset = DV + off_z * stride_vz + off_h * stride_vh + k_start * stride_vn
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + start_n * stride_dq_all + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    else:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh + q_start * stride_qm
    if SEQUENCE_PARALLEL:
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            d_offset,
            l_offset,
            stride_qm,
            stride_qk,
            stride_kn,
            stride_kk,
            stride_vn,
            stride_vk,
            stride_deltam,
            SEQLENS_Q,
            SEQLENS_K,
            start_n,
            num_block_m,
            BLOCK_M=BLOCK_M,
            BLOCK_DIM=BLOCK_DIM,
            ACTUAL_BLOCK_DIM=ACTUAL_BLOCK_DIM,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2,
        )
    else:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                d_offset,
                l_offset,
                stride_qm,
                stride_qk,
                stride_kn,
                stride_kk,
                stride_vn,
                stride_vk,
                stride_deltam,
                SEQLENS_Q,
                SEQLENS_K,
                start_n,
                num_block_m,
                BLOCK_M=BLOCK_M,
                BLOCK_DIM=BLOCK_DIM,
                ACTUAL_BLOCK_DIM=ACTUAL_BLOCK_DIM,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2,
            )


def attention_prefill_backward_triton_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    sm_scale: float,
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    use_exp2: bool,
    sequence_parallel=True,
):
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = get_shape_from_layout(
        q,
        k,
        layout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
    )
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    batch_headsize = batch * nheads_q
    is_varlen = layout == "thd"
    if max_seqlen_q <= 32 or max_seqlen_k <= 32:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 64
        BLOCK_N = 64
    num_warps = 4
    num_stages = 1
    num_blocks_m = cdiv(max_seqlen_q, BLOCK_M)
    num_blocks_n = cdiv(max_seqlen_k, BLOCK_N)
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DIM = padded_d_model
    ACTUAL_BLOCK_DIM = head_size
    if sequence_parallel:
        dq = jnp.zeros((num_blocks_n, *q.shape), dtype=q.dtype)
    else:
        dq = jnp.zeros(q.shape, dtype=q.dtype)
    stride_dq_all = get_strides(dq)[0]

    dk = jnp.empty_like(k)
    dv = jnp.empty_like(v)
    if is_varlen:
        stride_deltam, stride_deltah = get_strides(softmax_lse)
        stride_deltaz = 0
    else:
        stride_deltaz, stride_deltah, stride_deltam = get_strides(softmax_lse)
    (delta,) = triton_call(
        o,
        do,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_deltaz,
        stride_deltah,
        stride_deltam,
        cu_seqlens_q,
        max_seqlen_q,
        BLOCK_DIM=BLOCK_DIM,
        ACTUAL_BLOCK_DIM=ACTUAL_BLOCK_DIM,
        SEQLENS_Q=max_seqlen_q,
        IS_VARLEN=is_varlen,
        q_heads=nheads_q,
        BLOCK_M=BLOCK_M,
        grid=lambda META: (num_blocks_m, batch_headsize),
        out_shape=[jax.ShapeDtypeStruct(shape=softmax_lse.shape, dtype=softmax_lse.dtype)],
        kernel=_bwd_preprocess_use_o,
        name="ejgpu:attn_backward:prefill-delta",
    )
    dq, dk, dv = triton_call(
        q,
        k,
        v,
        sm_scale,
        o,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_dq_all,
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
        stride_vn,
        stride_vk,
        stride_deltaz,
        stride_deltah,
        stride_deltam,
        nheads_q,
        num_blocks_m,
        num_blocks_n,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        grid=lambda META: (batch_headsize, num_blocks_n if sequence_parallel else 1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DIM=BLOCK_DIM,
        ACTUAL_BLOCK_DIM=ACTUAL_BLOCK_DIM,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        num_warps=num_warps,
        num_stages=num_stages,
        IS_VARLEN=is_varlen,
        kernel=_bwd_kernel,
        name="ejgpu:attn_backward:prefill-delta",
        out_shape=[
            jax.ShapeDtypeStruct(dq.shape, dq.dtype),
            jax.ShapeDtypeStruct(dk.shape, dk.dtype),
            jax.ShapeDtypeStruct(dv.shape, dv.dtype),
        ],
        input_output_aliases={6: 0, 7: 1, 8: 2},
    )

    if sequence_parallel:
        dq = dq.sum(axis=0)

    return dq, dk, dv
