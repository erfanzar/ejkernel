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

"""Dedicated Triton GEMV kernels for quantized matmul M==1 families.

These kernels mirror GemLite's family split:
- GEMV Split-K for 8-bit effective modes
- GEMV RevSplit-K for 4-bit effective modes

They are intentionally separate from generic GEMM kernels to avoid
M==1 autotune-pruning corner cases and improve small-batch behavior.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import cdiv, strides_from_shape, triton_call
from ejkernel.quantization._utils.fp_tables import _get_e2m1_table, _get_e4m3_table, _get_nf4_table

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


def _get_decode_tables() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build decode lookup tables as local arrays (no global state)."""

    nf4_table = _get_nf4_table()
    e2m1_table, _ = _get_e2m1_table()
    e4m3_table, _ = _get_e4m3_table()
    e8m0_exp2_table = jnp.exp2(jnp.arange(256, dtype=jnp.uint8).astype(jnp.int8).astype(jnp.float32))
    return nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table


def _mode_to_id(mode: str) -> int:
    mode = mode.lower()
    if mode == "affine":
        return 0
    if mode == "nf4":
        return 1
    if mode == "mxfp4":
        return 2
    if mode == "mxfp8":
        return 3
    if mode == "nvfp4":
        return 4
    if mode == "nvfp8":
        return 5
    raise ValueError(f"Unsupported mode for Triton GEMV path: {mode}")


@triton.jit
def _decode_q_values(
    q,
    scale_vals,
    bias_vals,
    NF4_TABLE,
    E2M1_TABLE,
    E4M3_TABLE,
    E8M0_TABLE,
    MODE_ID: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    if MODE_ID == 0:
        out = q.to(tl.float32) * scale_vals.to(tl.float32)
        if HAS_BIAS:
            out = out + bias_vals.to(tl.float32)
        return out

    if MODE_ID == 1:
        return tl.load(NF4_TABLE + q.to(tl.int32)) * scale_vals.to(tl.float32)

    if MODE_ID == 2:
        scale = tl.load(E8M0_TABLE + scale_vals.to(tl.int32))
        return tl.load(E2M1_TABLE + q.to(tl.int32)) * scale

    if MODE_ID == 3:
        scale = tl.load(E8M0_TABLE + scale_vals.to(tl.int32))
        return tl.load(E4M3_TABLE + q.to(tl.int32)) * scale

    if MODE_ID == 4:
        scale = tl.load(E4M3_TABLE + scale_vals.to(tl.int32))
        return tl.load(E2M1_TABLE + q.to(tl.int32)) * scale

    scale = tl.load(E4M3_TABLE + scale_vals.to(tl.int32))
    return tl.load(E4M3_TABLE + q.to(tl.int32)) * scale


@triton.jit
def _qmm_gemv_splitk_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    NF4_TABLE,
    E2M1_TABLE,
    E4M3_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MODE_ID: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N

    if VALUES_PER_WORD == 8:
        q_bits = 4
        q_mask = 0xF
    else:
        q_bits = 8
        q_mask = 0xFF

    acc = tl.zeros((BN,), dtype=tl.float32)

    if not TRANSPOSE:
        word_offsets_n = offs_n // VALUES_PER_WORD
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        shifts_n = (offs_n % VALUES_PER_WORD) * q_bits
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x_vals = tl.load(X + offs_k * stride_xk, mask=k_mask, other=0.0).to(tl.float32)

        if TRANSPOSE:
            word_offsets_k = offs_k // VALUES_PER_WORD
            shifts_k = (offs_k % VALUES_PER_WORD) * q_bits

            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets_k[None, :] * stride_wq1,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            )
            q = (w_word >> shifts_k[None, :]) & q_mask
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[:, None] * stride_ws0 + group_idx_k[None, :] * stride_ws1,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[:, None] * stride_wb0 + group_idx_k[None, :] * stride_wb1,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0,
                )
            else:
                wb = ws

            w_vals = _decode_q_values(
                q,
                ws,
                wb,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & n_mask[None, :] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word >> shifts_n[None, :]) & q_mask
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0,
                )
            else:
                wb = ws

            w_vals = _decode_q_values(
                q,
                ws,
                wb,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals * x_vals[:, None], axis=0)

    out_ptrs = O + offs_n * stride_on
    if SPLIT_K == 1:
        tl.store(out_ptrs, acc, mask=n_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=n_mask)


@triton.jit
def _qmm_gemv_revsplitk_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    NF4_TABLE,
    E2M1_TABLE,
    E4M3_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    REV_PARTS: tl.constexpr,
    MODE_ID: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k2 = tl.program_id(1) * 2

    offs_n = pid_n * BN + tl.arange(0, BN)
    n_mask = offs_n < N

    if VALUES_PER_WORD == 8:
        q_bits = 4
        q_mask = 0xF
    else:
        q_bits = 8
        q_mask = 0xFF

    acc = tl.zeros((BN,), dtype=tl.float32)

    if not TRANSPOSE:
        word_offsets_n = offs_n // VALUES_PER_WORD
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        shifts_n = (offs_n % VALUES_PER_WORD) * q_bits
        group_idx_n = offs_n // GROUP_SIZE

    # GemLite-style reverse split-K: each program processes two split parts.
    for k0 in tl.range(0, K, BK * REV_PARTS, loop_unroll_factor=1):
        offs_k0 = k0 + pid_k2 * BK + tl.arange(0, BK)
        offs_k1 = k0 + (pid_k2 + 1) * BK + tl.arange(0, BK)
        # pass 0
        k_mask0 = offs_k0 < K
        x_vals0 = tl.load(X + offs_k0 * stride_xk, mask=k_mask0, other=0.0).to(tl.float32)

        if TRANSPOSE:
            word_offsets_k0 = offs_k0 // VALUES_PER_WORD
            shifts_k0 = (offs_k0 % VALUES_PER_WORD) * q_bits
            w_word0 = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets_k0[None, :] * stride_wq1,
                mask=n_mask[:, None] & k_mask0[None, :],
                other=0,
            )
            q0 = (w_word0 >> shifts_k0[None, :]) & q_mask
            group_idx_k0 = offs_k0 // GROUP_SIZE
            ws0 = tl.load(
                Wscale + offs_n[:, None] * stride_ws0 + group_idx_k0[None, :] * stride_ws1,
                mask=n_mask[:, None] & k_mask0[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb0 = tl.load(
                    Wbias + offs_n[:, None] * stride_wb0 + group_idx_k0[None, :] * stride_wb1,
                    mask=n_mask[:, None] & k_mask0[None, :],
                    other=0,
                )
            else:
                wb0 = ws0

            w_vals0 = _decode_q_values(
                q0,
                ws0,
                wb0,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals0 * x_vals0[None, :], axis=1)
        else:
            w_word0 = tl.load(
                Wq + offs_k0[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask0[:, None] & n_mask[None, :] & word_mask_n[None, :],
                other=0,
            )
            q0 = (w_word0 >> shifts_n[None, :]) & q_mask
            ws0 = tl.load(
                Wscale + offs_k0[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask0[:, None] & n_mask[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb0 = tl.load(
                    Wbias + offs_k0[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask0[:, None] & n_mask[None, :],
                    other=0,
                )
            else:
                wb0 = ws0

            w_vals0 = _decode_q_values(
                q0,
                ws0,
                wb0,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals0 * x_vals0[:, None], axis=0)

        # pass 1
        k_mask1 = offs_k1 < K
        x_vals1 = tl.load(X + offs_k1 * stride_xk, mask=k_mask1, other=0.0).to(tl.float32)

        if TRANSPOSE:
            word_offsets_k1 = offs_k1 // VALUES_PER_WORD
            shifts_k1 = (offs_k1 % VALUES_PER_WORD) * q_bits
            w_word1 = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets_k1[None, :] * stride_wq1,
                mask=n_mask[:, None] & k_mask1[None, :],
                other=0,
            )
            q1 = (w_word1 >> shifts_k1[None, :]) & q_mask
            group_idx_k1 = offs_k1 // GROUP_SIZE
            ws1 = tl.load(
                Wscale + offs_n[:, None] * stride_ws0 + group_idx_k1[None, :] * stride_ws1,
                mask=n_mask[:, None] & k_mask1[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb1 = tl.load(
                    Wbias + offs_n[:, None] * stride_wb0 + group_idx_k1[None, :] * stride_wb1,
                    mask=n_mask[:, None] & k_mask1[None, :],
                    other=0,
                )
            else:
                wb1 = ws1

            w_vals1 = _decode_q_values(
                q1,
                ws1,
                wb1,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals1 * x_vals1[None, :], axis=1)
        else:
            w_word1 = tl.load(
                Wq + offs_k1[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask1[:, None] & n_mask[None, :] & word_mask_n[None, :],
                other=0,
            )
            q1 = (w_word1 >> shifts_n[None, :]) & q_mask
            ws1 = tl.load(
                Wscale + offs_k1[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask1[:, None] & n_mask[None, :],
                other=0,
            )
            if HAS_BIAS:
                wb1 = tl.load(
                    Wbias + offs_k1[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask1[:, None] & n_mask[None, :],
                    other=0,
                )
            else:
                wb1 = ws1

            w_vals1 = _decode_q_values(
                q1,
                ws1,
                wb1,
                NF4_TABLE,
                E2M1_TABLE,
                E4M3_TABLE,
                E8M0_TABLE,
                MODE_ID=MODE_ID,
                HAS_BIAS=HAS_BIAS,
            )
            acc += tl.sum(w_vals1 * x_vals1[:, None], axis=0)

    out_ptrs = O + offs_n * stride_on
    if REV_PARTS <= 2:
        tl.store(out_ptrs, acc, mask=n_mask)
    else:
        tl.atomic_add(out_ptrs, acc, mask=n_mask)


def quantized_matmul_triton_gemv(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: QuantizationMode,
    kernel_family: Literal["gemv_splitk", "gemv_revsplitk"],
    split_k: int,
    revsplit_parts: int | None,
    block_n: int,
) -> jax.Array:
    """Run dedicated Triton GEMV family kernels for M==1 quantized matmul."""
    nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table = _get_decode_tables()
    if int(x.shape[0]) != 1:
        raise ValueError("Dedicated GEMV kernels require M == 1.")

    mode_id = _mode_to_id(mode)
    values_per_word = 32 // bits

    M = int(x.shape[0])
    K = int(x.shape[1])
    if transpose:
        N = int(w.shape[0])
    else:
        N = int(scales.shape[1]) * int(group_size)

    stride_xm, stride_xk = strides_from_shape(x.shape)
    del stride_xm
    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)
    if biases is not None:
        stride_wb0, stride_wb1 = strides_from_shape(biases.shape)
    else:
        stride_wb0, stride_wb1 = 0, 0

    out_shape = (M, N)
    _, stride_on = strides_from_shape(out_shape)
    bias_arg = biases if biases is not None else scales

    bn = 64 if N <= 128 else (128 if N <= 2048 else min(int(block_n), 256))
    if bn <= 0:
        bn = 128

    if kernel_family == "gemv_splitk":
        split_k = max(1, int(split_k))
        (out,) = triton_call(
            x,
            w,
            scales,
            bias_arg,
            nf4_table,
            e2m1_table,
            e4m3_table,
            e8m0_exp2_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=out_shape, dtype=jnp.float32)],
            grid=lambda META: (cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=_qmm_gemv_splitk_kernel,
            zeroed_outputs=(0,),
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_wb0=stride_wb0,
            stride_wb1=stride_wb1,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=values_per_word,
            BN=bn,
            BK=32,
            SPLIT_K=split_k,
            MODE_ID=mode_id,
            TRANSPOSE=transpose,
            HAS_BIAS=biases is not None,
        )
        return out.astype(jnp.bfloat16)

    rev_parts = 2 if revsplit_parts is None else int(revsplit_parts)
    rev_parts = max(2, rev_parts)
    if rev_parts not in {2, 4, 8, 16}:
        raise ValueError("revsplit_k_parts must be one of {2,4,8,16} for GEMV RevSplit-K.")

    (out,) = triton_call(
        x,
        w,
        scales,
        bias_arg,
        nf4_table,
        e2m1_table,
        e4m3_table,
        e8m0_exp2_table,
        N,
        K,
        out_shape=[jax.ShapeDtypeStruct(shape=out_shape, dtype=jnp.float32)],
        grid=lambda META: (cdiv(N, META["BN"]), cdiv(META["REV_PARTS"], 2)),
        kernel=_qmm_gemv_revsplitk_kernel,
        zeroed_outputs=(0,),
        stride_xk=stride_xk,
        stride_wq0=stride_wq0,
        stride_wq1=stride_wq1,
        stride_ws0=stride_ws0,
        stride_ws1=stride_ws1,
        stride_wb0=stride_wb0,
        stride_wb1=stride_wb1,
        stride_on=stride_on,
        GROUP_SIZE=group_size,
        VALUES_PER_WORD=values_per_word,
        BN=bn,
        BK=32,
        REV_PARTS=rev_parts,
        MODE_ID=mode_id,
        TRANSPOSE=transpose,
        HAS_BIAS=biases is not None,
    )
    return out.astype(jnp.bfloat16)


__all__ = ("quantized_matmul_triton_gemv",)
