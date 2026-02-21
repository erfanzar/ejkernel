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

"""Triton kernels for quantized matrix multiplication.

This module contains the low-level Triton GPU kernels for quantized matmul
operations. It provides optimized fused dequantization and matmul kernels
for "affine", "nf4", "mxfp4", "mxfp8", "nvfp4", and "nvfp8" quantization modes.

The kernels use split-K parallelism for improved performance on small M
dimensions and support both transposed (NxK) and non-transposed (KxN)
weight layouts.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import cdiv, strides_from_shape, triton_call
from ejkernel.quantization._utils.fp_tables import _get_e2m1_table, _get_e4m3_table, _get_nf4_table
from ejkernel.quantization._utils.qparams import (
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    select_qmm_kernel_family,
)

from ._triton_impl_gemv import quantized_matmul_triton_gemv

#: Supported quantization modes for Triton kernels.
QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


def _get_decode_tables() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build decode lookup tables as local arrays (no global state)."""

    nf4_table = _get_nf4_table()
    e2m1_table, _ = _get_e2m1_table()
    e4m3_table, _ = _get_e4m3_table()
    e8m0_exp2_table = jnp.exp2(jnp.arange(256, dtype=jnp.uint8).astype(jnp.int8).astype(jnp.float32))
    return nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table


@triton.jit
def _nf4_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 4-bit NF4 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def _e2m1_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 4-bit E2M1 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def _e4m3_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 8-bit E4M3 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def qmm_dequant_nf4_kernel(
    Wq,
    Wscale,
    NF4_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = _nf4_to_f32(q.to(tl.int32), NF4_TABLE) * ws
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = _nf4_to_f32(q.to(tl.int32), NF4_TABLE) * ws
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_affine4_kernel(
    Wq,
    Wscale,
    Wbias,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_affine8_kernel(
    Wq,
    Wscale,
    Wbias,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_mxfp4_kernel(
    Wq,
    Wscale,
    E2M1_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_mxfp8_kernel(
    Wq,
    Wscale,
    E4M3_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_nvfp4_kernel(
    Wq,
    Wscale,
    E2M1_TABLE,
    E4M3_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_nvfp8_kernel(
    Wq,
    Wscale,
    E4M3_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


def _zeroed_outputs_for_splitk(meta: dict) -> tuple[int, ...]:
    """Return output indices that should be zeroed for split-K kernels.

    When using split-K parallelism, the output buffer must be zeroed before
    the kernel runs because partial results are accumulated via atomic_add.

    Args:
        meta: Kernel metadata containing SPLIT_K configuration.

    Returns:
        Tuple of output indices to zero, or empty tuple if SPLIT_K == 1.
    """
    return (0,) if meta["SPLIT_K"] > 1 else ()


def _env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    return value.lower() in {"1", "true", "yes", "y"}


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer env var with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(1, value)


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    """Parse a non-negative integer env var with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(0, value)


def _parse_matmul_precision(value: str):
    value = value.lower()
    if value == "highest":
        return jax.lax.Precision.HIGHEST
    if value == "high":
        return jax.lax.Precision.HIGH
    if value == "fastest":
        return jax.lax.Precision.FASTEST
    return jax.lax.Precision.DEFAULT


def _parse_output_dtype(value: str):
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"fp16", "float16"}:
        return jnp.float16
    if value in {"fp32", "float32"}:
        return jnp.float32
    return None


@lru_cache(maxsize=1)
def _cuda_max_shared_mem_per_block_bytes() -> int | None:
    """Best-effort query of CUDA's max shared memory per block (opt-in limit).

    JAX's Triton runtime will attempt to launch autotune candidates as-is; when
    a config requires more shared memory than the device allows by default,
    JAX logs messages like "Unable to launch autotune config on device".

    Prefer querying `cudaDevAttrMaxSharedMemoryPerBlockOptin` (attribute 97) so
    we do not over-prune on GPUs where the Triton launcher opts-in to the larger
    shared-memory limit. Fall back to `cudaDevAttrMaxSharedMemoryPerBlock`
    (attribute 8) when the opt-in attribute isn't available.
    """
    try:
        import ctypes
    except Exception:
        return None

    lib = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None

    cudaGetDevice = getattr(lib, "cudaGetDevice", None)
    cudaDeviceGetAttribute = getattr(lib, "cudaDeviceGetAttribute", None)
    if cudaGetDevice is None or cudaDeviceGetAttribute is None:
        return None

    cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudaGetDevice.restype = ctypes.c_int
    cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cudaDeviceGetAttribute.restype = ctypes.c_int

    dev = ctypes.c_int()
    if int(cudaGetDevice(ctypes.byref(dev))) != 0:
        return None

    def _get_attr(attr_id: int) -> int | None:
        val = ctypes.c_int()
        if int(cudaDeviceGetAttribute(ctypes.byref(val), attr_id, int(dev.value))) != 0:
            return None
        out = int(val.value)
        return out if out > 0 else None

    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    cudaDevAttrMaxSharedMemoryPerBlock = 8
    return _get_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin) or _get_attr(cudaDevAttrMaxSharedMemoryPerBlock)


@lru_cache(maxsize=1)
def _qmm_smem_limit_bytes() -> int:
    """Return an estimate of usable shared memory per CTA for QMM kernels.

    This helper is used while constructing module-level autotune config lists,
    so it must not query JAX runtime backend state (which would eagerly
    initialize backends at import time).
    """
    # Conservative default when CUDA runtime querying is unavailable.
    default = 96 * 1024

    # Prefer the device-reported per-block limit (opt-in when available).
    limit = _cuda_max_shared_mem_per_block_bytes()
    if limit is None:
        return default

    # Keep the historical 96KB cap for compile time bounds, but honor smaller
    # device limits to avoid invalid configs (and warnings) during autotune.
    return min(default, int(limit))


def _qmm_estimated_smem_bytes(*, bm: int, bn: int, bk: int, num_stages: int) -> int:
    # Two tiles (A: BMxBK, B: BKxBN) with 16-bit elements, pipelined by num_stages.
    return int((bm * bk + bk * bn) * 2 * num_stages)


def _qmm_autotune_configs(mode: str, *, bits: int | None = None) -> list[triton.Config]:
    """Generate autotune configurations for quantized matmul kernels.

    Produces a curated set of Triton configurations tuned per mode, while
    keeping compile time bounded. Configurations that exceed the shared
    memory limit are filtered out.
    """
    mode = mode.lower()
    configs: list[triton.Config] = []
    smem_limit = _qmm_smem_limit_bytes()
    # Minifloat decode modes tend to be more register/smem heavy; keep the
    # search space limited to configs that reliably launch on common GPUs.
    if mode in {"mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        smem_limit = min(smem_limit, 64 * 1024)

    if mode == "nf4":
        bm_choices = (32, 64, 128)
        bn_choices = (32, 64, 128, 256)
        bk_choices = (32, 64)
        split_ks = (1, 2, 4, 8, 16)
        extra = ()
    elif mode in {"affine", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        bm_choices = (32, 64, 128, 256)
        bn_choices = (64, 128, 256, 512)
        if bits == 8:
            bk_choices = (64, 128)
        else:
            bk_choices = (32, 64, 128)
        split_ks = (1, 2, 4, 8, 16)
        extra = (
            (64, 512, 64, 8, 1, 1),
            (64, 512, 128, 8, 1, 1),
            (128, 512, 64, 8, 1, 1),
            (128, 512, 128, 8, 1, 1),
            (128, 256, 64, 4, 1, 1),
            (128, 256, 128, 4, 1, 1),
            (64, 256, 64, 4, 1, 1),
            (128, 128, 128, 8, 2, 1),
        )
    else:
        raise ValueError(f"Unsupported mode for autotune configs: {mode}")

    for bm in bm_choices:
        for bn in bn_choices:
            for bk in bk_choices:
                if bk >= 128:
                    stages_choices = (1, 2)
                elif bk >= 64:
                    stages_choices = (1, 2, 3)
                else:
                    stages_choices = (1, 2, 3, 4)

                for num_stages in stages_choices:
                    # Empirically, some nf4 configs (notably BN=256, BK=64 with
                    # pipelining) can fail to launch on common NVIDIA GPUs,
                    # spamming "Unable to launch autotune config" warnings.
                    if mode == "nf4" and bn >= 256 and bk == 64 and num_stages > 1:
                        continue
                    smem = _qmm_estimated_smem_bytes(bm=bm, bn=bn, bk=bk, num_stages=num_stages)
                    if smem > smem_limit:
                        continue

                    if bm >= 256 or bn >= 256:
                        warps = (4, 8)
                    elif bm >= 128 or bn >= 128:
                        warps = (4, 8)
                    else:
                        warps = (2, 4)

                    for num_warps in warps:
                        for split_k in split_ks:
                            if bk >= 128 and (bm >= 256 or bn >= 256) and num_stages > 1:
                                continue
                            configs.append(
                                triton.Config(
                                    {"BM": bm, "BN": bn, "BK": bk, "SPLIT_K": split_k},
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )

    for bm, bn, bk, num_warps, num_stages, split_k in extra:
        smem = _qmm_estimated_smem_bytes(bm=bm, bn=bn, bk=bk, num_stages=num_stages)
        if smem > smem_limit:
            continue
        configs.append(
            triton.Config(
                {"BM": bm, "BN": bn, "BK": bk, "SPLIT_K": split_k},
                num_warps=num_warps,
                num_stages=num_stages,
            )
        )

    return configs


def _qmm_autotune_configs_large(mode: str, *, bits: int | None = None) -> list[triton.Config]:
    """Generate a compact config set specialized for large square-ish shapes."""
    mode = mode.lower()
    configs: list[triton.Config] = []
    smem_limit = _qmm_smem_limit_bytes()
    if mode in {"mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        smem_limit = min(smem_limit, 64 * 1024)

    if mode == "nf4":
        bm_choices = (128, 256)
        bn_choices = (128, 256)
        bk_choices = (64, 128)
    elif mode in {"affine", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        bm_choices = (128, 256)
        bn_choices = (128, 256)
        bk_choices = (64, 128)
    else:
        raise ValueError(f"Unsupported mode for autotune configs: {mode}")

    split_ks = (1, 2)
    num_warps_choices = (4, 8)
    # Include stage=1 for nf4 so BN=256 cells have a valid candidate without
    # tripping launch constraints on some GPUs.
    num_stages_choices = (1, 2, 3) if mode == "nf4" else (2, 3, 4)

    for bm in bm_choices:
        for bn in bn_choices:
            for bk in bk_choices:
                for num_warps in num_warps_choices:
                    for num_stages in num_stages_choices:
                        if mode == "nf4" and bn >= 256 and bk == 64 and num_stages > 1:
                            continue
                        smem = _qmm_estimated_smem_bytes(bm=bm, bn=bn, bk=bk, num_stages=num_stages)
                        if smem > smem_limit:
                            continue
                        for split_k in split_ks:
                            configs.append(
                                triton.Config(
                                    {"BM": bm, "BN": bn, "BK": bk, "SPLIT_K": split_k},
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )

    return configs


_QMM_AUTOTUNE_CONFIGS_NF4 = _qmm_autotune_configs("nf4")
_QMM_AUTOTUNE_CONFIGS_AFFINE4 = _qmm_autotune_configs("affine", bits=4)
_QMM_AUTOTUNE_CONFIGS_AFFINE8 = _qmm_autotune_configs("affine", bits=8)
_QMM_AUTOTUNE_CONFIGS_MXFP4 = _qmm_autotune_configs("mxfp4", bits=4)
_QMM_AUTOTUNE_CONFIGS_MXFP8 = _qmm_autotune_configs("mxfp8", bits=8)
_QMM_AUTOTUNE_CONFIGS_NVFP4 = _qmm_autotune_configs("nvfp4", bits=4)
_QMM_AUTOTUNE_CONFIGS_NVFP8 = _qmm_autotune_configs("nvfp8", bits=8)

_QMM_AUTOTUNE_CONFIGS_NF4_LARGE = _qmm_autotune_configs_large("nf4")
_QMM_AUTOTUNE_CONFIGS_AFFINE4_LARGE = _qmm_autotune_configs_large("affine", bits=4)
_QMM_AUTOTUNE_CONFIGS_AFFINE8_LARGE = _qmm_autotune_configs_large("affine", bits=8)
_QMM_AUTOTUNE_CONFIGS_MXFP4_LARGE = _qmm_autotune_configs_large("mxfp4", bits=4)
_QMM_AUTOTUNE_CONFIGS_MXFP8_LARGE = _qmm_autotune_configs_large("mxfp8", bits=8)
_QMM_AUTOTUNE_CONFIGS_NVFP4_LARGE = _qmm_autotune_configs_large("nvfp4", bits=4)
_QMM_AUTOTUNE_CONFIGS_NVFP8_LARGE = _qmm_autotune_configs_large("nvfp8", bits=8)


def _qmm_config_pruner(configs, nargs, **_kwargs):
    """Lightweight config pruning based on problem shape.

    Reduces autotune compile time by skipping configs that are
    clearly mismatched with the input dimensions and by capping
    the candidate set size after scoring.
    """
    m = int(nargs["M"])
    n = int(nargs["N"])
    k = int(nargs["K"])
    group_size = int(nargs.get("GROUP_SIZE", 0))
    requested_split_k = int(nargs.get("SPLIT_K", 1))
    smem_limit = _qmm_smem_limit_bytes()
    max_candidates_default = _parse_positive_int_env("EJKERNEL_TRITON_QMM_AUTOTUNE_MAX_CANDIDATES", 12)
    max_candidates_large_default = _parse_positive_int_env("EJKERNEL_TRITON_QMM_AUTOTUNE_MAX_CANDIDATES_LARGE", 8)
    min_smem_large_default = _parse_nonnegative_int_env("EJKERNEL_TRITON_QMM_AUTOTUNE_MIN_SMEM_BYTES_LARGE", 24 * 1024)

    def _nearest_choices(value: int, choices: tuple[int, ...], count: int) -> set[int]:
        ranked = sorted(choices, key=lambda x: abs(x - value))
        return set(ranked[:count])

    bm_keep = _nearest_choices(m, (32, 64, 128, 256), count=2)
    bn_keep = _nearest_choices(n, (32, 64, 128, 256, 512), count=2)
    bk_keep = _nearest_choices(k, (32, 64, 128), count=2)

    pruned = []
    large_mn = m >= 2048 and n >= 2048
    large_k = k >= 4096
    min_smem_bytes = min_smem_large_default if (large_mn and large_k) else 0
    if large_mn and large_k:
        bm_keep = {128}
        bn_keep = {128, 256}
        bk_keep = {64, 128}
    for cfg in configs:
        bm = cfg.kwargs["BM"]
        bn = cfg.kwargs["BN"]
        bk = cfg.kwargs["BK"]
        split_k = cfg.kwargs["SPLIT_K"]
        num_stages = int(getattr(cfg, "num_stages", 1))
        smem = _qmm_estimated_smem_bytes(bm=bm, bn=bn, bk=bk, num_stages=num_stages)

        # Avoid configs that will never launch due to shared memory limits.
        if smem > smem_limit:
            continue
        # For large GEMM shapes, drop tiny-SMEM tiles that are almost always
        # throughput-poor but still expensive to autotune.
        if min_smem_bytes > 0 and smem < min_smem_bytes:
            continue

        if bm not in bm_keep or bn not in bn_keep or bk not in bk_keep:
            continue
        if m <= 64 and bm > 128:
            continue
        if n <= 64 and bn > 128:
            continue
        if k <= 64 and bk > 64:
            continue
        if large_mn:
            if bm < 128 or bn < 128:
                continue
            if split_k > 1:
                continue
        if large_k and bk < 64:
            continue
        if group_size and group_size >= 64 and bk < group_size:
            continue
        if split_k > 1 and k < bk * split_k:
            continue

        pruned.append(cfg)
    if pruned:

        def _score(cfg) -> tuple[int, int, int, int, int]:
            bm = int(cfg.kwargs["BM"])
            bn = int(cfg.kwargs["BN"])
            bk = int(cfg.kwargs["BK"])
            split_k = int(cfg.kwargs.get("SPLIT_K", 1))
            split_penalty = 0 if split_k == requested_split_k else 1
            distance = abs(bm - m) + abs(bn - n) + abs(bk - k)
            tile_volume = bm * bn * bk
            smem = _qmm_estimated_smem_bytes(
                bm=bm,
                bn=bn,
                bk=bk,
                num_stages=int(getattr(cfg, "num_stages", 1)),
            )
            return (
                split_penalty,
                distance,
                -tile_volume,
                -smem,
                -int(getattr(cfg, "num_warps", 1)),
            )

        pruned.sort(key=_score)
        max_candidates = max_candidates_large_default if (large_mn and large_k) else max_candidates_default
        return pruned[:max_candidates]

    # Keep autotune alive for legal-but-narrow shapes (notably M==1 overrides).
    fallback = [cfg for cfg in configs if int(cfg.kwargs.get("SPLIT_K", 1)) == requested_split_k]
    if fallback:
        return fallback[:1]
    return configs[:1]


def _unwrap_triton_fn(kernel):
    """Return the original Python function for a Triton kernel wrapper."""
    fn = getattr(kernel, "fn", kernel)
    return getattr(fn, "fn", fn)


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NF4,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_nf4_kernel(
    X,
    Wq,
    Wscale,
    NF4_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NF4 dequantization and matrix multiplication Triton kernel.

    Performs x @ dequant(w) where w is packed in NF4 (4-bit NormalFloat) format.
    Each 32-bit word contains 8 NF4 codes that are decoded via table lookup,
    scaled by per-group scale factors, and multiplied with the activation tile.

    Supports both transposed (NxK) and non-transposed (KxN) weight layouts.
    Uses split-K parallelism with atomic accumulation when SPLIT_K > 1.

    Args:
        X: Input activation matrix pointer, shape (M, K).
        Wq: Packed NF4 weights pointer (uint32, 8 values per word).
        Wscale: Per-group scale factors pointer.
        NF4_TABLE: NF4 codebook lookup table pointer (16 float32 entries).
        M, N, K: Matrix dimensions.
        O: Output matrix pointer, shape (M, N).
        stride_*: Tensor stride parameters.
        GROUP_SIZE: Number of elements per quantization group.
        VALUES_PER_WORD: Number of quantized values per uint32 word (8 for NF4).
        BM, BK, BN: Block tile sizes for M, K, N dimensions.
        SPLIT_K: Split-K parallelism factor.
        USE_BF16: If True, use BF16 for dot product tiles; otherwise FP16.
        TRANSPOSE: If True, weights are in NxK layout; otherwise KxN.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

        w = _nf4_to_f32(q.to(tl.int32), NF4_TABLE).to(dot_ty) * ws.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nf4_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NF4_LARGE,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_nf4_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_AFFINE8,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_affine8_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = True,
):
    """Fused affine dequantization and matrix multiplication Triton kernel (8-bit)."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[None, :] * stride_wb0 + group_idx_k[:, None] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

        w = q.to(dot_ty) * ws.to(dot_ty)
        if HAS_BIAS:
            w = w + wb.to(dot_ty)

        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_affine8_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_AFFINE8_LARGE,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_affine8_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_AFFINE4,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_affine4_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = True,
):
    """Fused affine dequantization and matrix multiplication Triton kernel (4-bit).

    Performs x @ dequant(w) where w is packed in affine quantization format.
    Dequantization applies: w_float = w_int * scale + bias (when HAS_BIAS)
    or w_float = w_int * scale (when not HAS_BIAS).

    Supports 4-bit quantization with per-group scale and bias factors.
    Multiple quantized values are packed into uint32 words.

    Uses split-K parallelism with atomic accumulation when SPLIT_K > 1.

    Args:
        X: Input activation matrix pointer, shape (M, K).
        Wq: Packed quantized weights pointer (uint32).
        Wscale: Per-group scale factors pointer.
        Wbias: Per-group bias factors pointer (ignored if HAS_BIAS=False).
        M, N, K: Matrix dimensions.
        O: Output matrix pointer, shape (M, N).
        stride_*: Tensor stride parameters.
        GROUP_SIZE: Number of elements per quantization group.
        Fixed 4-bit quantization (8 values per uint32).
        BM, BK, BN: Block tile sizes for M, K, N dimensions.
        SPLIT_K: Split-K parallelism factor.
        USE_BF16: If True, use BF16 for dot product tiles; otherwise FP16.
        TRANSPOSE: If True, weights are in NxK layout; otherwise KxN.
        HAS_BIAS: If True, apply per-group bias during dequantization.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[None, :] * stride_wb0 + group_idx_k[:, None] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

        w = q.to(dot_ty) * ws.to(dot_ty)
        if HAS_BIAS:
            w = w + wb.to(dot_ty)

        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_affine4_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_AFFINE4_LARGE,
    key=["M", "N", "K", "GROUP_SIZE", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_affine4_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_MXFP4,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_mxfp4_kernel(
    X,
    Wq,
    Wscale,
    E2M1_TABLE,
    E8M0_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused MXFP4 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        w = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_mxfp4_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_MXFP4_LARGE,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_mxfp4_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_MXFP8,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_mxfp8_kernel(
    X,
    Wq,
    Wscale,
    E4M3_TABLE,
    E8M0_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused MXFP8 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        w = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_mxfp8_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_MXFP8_LARGE,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_mxfp8_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NVFP4,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_nvfp4_kernel(
    X,
    Wq,
    Wscale,
    E2M1_TABLE,
    E4M3_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NVFP4 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        w = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nvfp4_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NVFP4_LARGE,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_nvfp4_kernel)))


@triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NVFP8,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)
@triton.jit
def qmm_nvfp8_kernel(
    X,
    Wq,
    Wscale,
    E4M3_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NVFP8 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        w = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nvfp8_kernel_large = triton.autotune(
    configs=_QMM_AUTOTUNE_CONFIGS_NVFP8_LARGE,
    key=["M", "N", "K", "TRANSPOSE", "USE_BF16"],
    prune_configs_by={"early_config_prune": _qmm_config_pruner},
)(triton.jit(_unwrap_triton_fn(qmm_nvfp8_kernel)))


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve and validate quantization parameters for Triton kernels.

    Applies mode-specific defaults and validates that the parameters are
    compatible with the Triton kernel implementations.

    Args:
        mode: Quantization mode ("affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8").
        group_size: Number of elements per quantization group, or None for default.
        bits: Bit-width per quantized element, or None for default.

    Returns:
        Tuple of (resolved_group_size, resolved_bits).

    Raises:
        ValueError: If mode is not supported by Triton kernels.
        ValueError: If group_size is not in {32, 64, 128} for affine mode.
        ValueError: If bits is not in {4, 8} for affine mode.
        ValueError: If bits != 4 for nf4 mode.
        ValueError: If group_size/bits mismatch for explicit MXFP/NVFP modes.
    """
    _, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return int(group_size), int(bits)


def _validate_shapes(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
) -> tuple[int, int, int]:
    """Validate input array shapes and extract matrix dimensions.

    Performs shape validation to ensure all inputs are compatible and
    extracts the M, K, N dimensions for the matmul operation.

    Args:
        x: Input activation matrix of shape (M, K).
        w: Packed uint32 weights. Shape depends on transpose setting.
        scales: Per-group scales array.
        biases: Per-group affine additive offsets (optional).
        transpose: If True, weights are in NxK layout; if False, KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element.

    Returns:
        Tuple of (M, K, N) dimensions for the matmul operation.

    Raises:
        ValueError: If any input is not 2D.
        ValueError: If packed weight shape doesn't match expected dimensions.
        ValueError: If scales/affine-offset shapes are inconsistent.
    """
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 2:
        raise ValueError("x, w, and scales must be 2D arrays.")
    if biases is not None and biases.ndim != 2:
        raise ValueError("biases must be 2D when provided.")

    M, K = x.shape
    values_per_word = 32 // bits

    if transpose:
        N = w.shape[0]
        words_expected = math.ceil(K / values_per_word)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match K dimension.")
        if scales.shape[0] != N:
            raise ValueError("scales first dimension must match N when transpose=True.")
        groups_expected = K // group_size
        if scales.shape[1] != groups_expected:
            raise ValueError("scales second dimension must match K/group_size.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")
    else:
        if w.shape[0] != K:
            raise ValueError("Packed weight first dimension must match K when transpose=False.")
        groups_expected = scales.shape[1]
        N = groups_expected * group_size
        words_expected = math.ceil(N / values_per_word)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match N dimension.")
        if scales.shape[0] != K:
            raise ValueError("scales first dimension must match K when transpose=False.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")

    return M, K, N


def _validate_weight_shapes(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
) -> tuple[int, int]:
    """Validate weight/scales shapes and return (K, N)."""
    if w.ndim != 2 or scales.ndim != 2:
        raise ValueError("w and scales must be 2D arrays.")
    if biases is not None and biases.ndim != 2:
        raise ValueError("biases must be 2D when provided.")

    values_per_word = 32 // bits
    if transpose:
        N = w.shape[0]
        K = scales.shape[1] * group_size
        words_expected = math.ceil(K / values_per_word)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match K dimension.")
        if scales.shape[0] != N:
            raise ValueError("scales first dimension must match N when transpose=True.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")
    else:
        K = w.shape[0]
        N = scales.shape[1] * group_size
        words_expected = math.ceil(N / values_per_word)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match N dimension.")
        if scales.shape[0] != K:
            raise ValueError("scales first dimension must match K when transpose=False.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")

    return K, N


def _select_split_k(k: int, block_k: int, max_split: int = 8) -> int:
    """Select the split-K factor based on the K dimension size.

    Heuristically determines how many splits to use along the K dimension
    for parallel reduction. Larger K dimensions benefit from more splits
    to increase parallelism on the GPU.

    Args:
        k: Total K dimension size.
        block_k: Block size for the K dimension.
        max_split: Maximum allowed split-K value.

    Returns:
        Split-K factor (1, 2, 4, or 8).
    """
    if block_k <= 0:
        return 1
    tiles = math.ceil(k / block_k)
    if tiles >= 256:
        return min(8, max_split)
    if tiles >= 128:
        return min(4, max_split)
    if tiles >= 64:
        return min(2, max_split)
    return 1


def quantized_matmul_dequant_triton(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    use_bf16: bool = True,
) -> jax.Array:
    """Dequantize packed weights into BF16/FP16 for two-stage matmul."""
    nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table = _get_decode_tables()
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires affine metadata.")
    if mode != "affine" and biases is not None:
        raise ValueError("affine metadata must be None for non-affine modes.")
    if mode == "affine" and bits not in (4, 8):
        raise ValueError("Triton affine kernel supports bits in {4, 8}.")

    K, N = _validate_weight_shapes(
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )

    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)

    out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
    if transpose:
        deq_shape = (N, K)
        stride_or, stride_oc = strides_from_shape(deq_shape)
        r_dim, c_dim = N, K
    else:
        deq_shape = (K, N)
        stride_or, stride_oc = strides_from_shape(deq_shape)
        r_dim, c_dim = K, N

    br = 128
    bc = 128

    def grid(META):
        return (cdiv(r_dim, META["BR"]), cdiv(c_dim, META["BC"]))

    if mode == "nf4":
        (w_deq,) = triton_call(
            w,
            scales,
            nf4_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nf4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "affine":
        stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
        if bits == 4:
            kernel = qmm_dequant_affine4_kernel
            values_per_word = 8
        else:
            kernel = qmm_dequant_affine8_kernel
            values_per_word = 4
        (w_deq,) = triton_call(
            w,
            scales,
            biases,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_wb0=stride_wb0,
            stride_wb1=stride_wb1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=values_per_word,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "mxfp4":
        (w_deq,) = triton_call(
            w,
            scales,
            e2m1_table,
            e8m0_exp2_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_mxfp4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "mxfp8":
        (w_deq,) = triton_call(
            w,
            scales,
            e4m3_table,
            e8m0_exp2_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_mxfp8_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "nvfp4":
        (w_deq,) = triton_call(
            w,
            scales,
            e2m1_table,
            e4m3_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nvfp4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "nvfp8":
        (w_deq,) = triton_call(
            w,
            scales,
            e4m3_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nvfp8_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    else:
        raise ValueError(f"Unsupported mode for two-stage path: {mode}")

    return w_deq


def quantized_matmul_triton(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    use_bf16: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
) -> jax.Array:
    """Execute quantized matmul using Triton GPU kernels.

    This is the core Triton implementation that dispatches to the
    appropriate quantization kernel based on the mode parameter. The
    kernels perform fused dequantization and matmul for optimal performance.

    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights. For transpose=True, shape is
            (N, ceil(K/values_per_word)). For transpose=False, shape is
            (K, ceil(N/values_per_word)), where values_per_word = 32 // bits.
        scales: Per-group scales. Shape is (N, K//group_size) for
            transpose=True or (K, N//group_size) for transpose=False.
        biases: Per-group affine additive offsets (required for affine mode only). Must have
            the same shape as scales.
        transpose: If True, weights are stored in NxK layout and the kernel
            computes x @ w.T. If False, weights are in KxN layout and the
            kernel computes x @ w. Default is False.
        group_size: Number of elements per quantization group. If None,
            uses mode defaults (affine/nf4: 64, mxfp4/mxfp8: 32, nvfp4/nvfp8: 16).
        bits: Bit-width per quantized element. If None, uses mode defaults
            (affine/nf4/mxfp4/nvfp4: 4, mxfp8/nvfp8: 8).
        mode: Quantization mode. One of "affine", "nf4", "mxfp4", "mxfp8",
            "nvfp4", "nvfp8".
        use_bf16: If True, use BF16 for dot product input tiles.
            If False, use FP16. Default is True.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If mode is "affine" but affine metadata is missing.
        ValueError: If mode is not "affine" but affine metadata is provided.
        ValueError: If bits/group_size are invalid for the selected mode.
        ValueError: If input shapes are invalid or inconsistent.
    """
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

    if use_bf16 and getattr(x, "dtype", None) == jnp.float16:
        use_bf16 = False

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires affine metadata.")
    if mode != "affine" and biases is not None:
        raise ValueError("affine metadata must be None for non-affine modes.")

    if mode == "affine" and bits not in (4, 8):
        raise ValueError("Triton affine kernel supports bits in {4, 8}.")

    M, K, N = _validate_shapes(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )

    kernel_family, family_revsplit_parts = select_qmm_kernel_family(
        m=int(M),
        mode=mode,  # type: ignore[arg-type]
        bits=bits,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if kernel_family == "gemm":
        split_k_selected = 1
    elif kernel_family == "gemm_splitk":
        if split_k is None:
            split_k_selected = max(1, _select_split_k(K, block_k, max_split=8))
        else:
            split_k_selected = max(1, int(split_k))
    elif kernel_family == "gemv_splitk":
        if split_k is None:
            split_k_selected = max(1, _select_split_k(K, block_k, max_split=16))
        else:
            split_k_selected = max(1, int(split_k))
    else:
        split_k_selected = 2 if family_revsplit_parts is None else int(family_revsplit_parts)

    if split_k_selected not in {1, 2, 4, 8, 16}:
        raise ValueError("split_k must be one of {1,2,4,8,16}.")

    if kernel_family in {"gemv_splitk", "gemv_revsplitk"}:
        return quantized_matmul_triton_gemv(
            x,
            w,
            scales,
            biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,  # type: ignore[arg-type]
            kernel_family=kernel_family,  # type: ignore[arg-type]
            split_k=split_k_selected,
            revsplit_parts=family_revsplit_parts,
            block_n=block_n,
        )

    nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table = _get_decode_tables()

    stride_xm, stride_xk = strides_from_shape(x.shape)
    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)
    stride_om, stride_on = strides_from_shape((M, N))

    num_warps = int(num_warps) if num_warps is not None else 4
    num_stages = int(num_stages) if num_stages is not None else 3

    use_large_kernel = M >= 4096 and N >= 4096 and K >= 4096 and kernel_family in {"gemm", "gemm_splitk"}
    use_two_stage = _env_flag("EJKERNEL_QMM_TWO_STAGE", "1") and use_large_kernel

    if use_two_stage:
        out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
        output_dtype = jnp.bfloat16
        precision_env = os.getenv("EJKERNEL_QMM_MATMUL_PRECISION", "")
        if precision_env:
            matmul_precision = _parse_matmul_precision(precision_env)
        else:
            max_dim = max(M, N, K)
            if max_dim <= 2048:
                matmul_precision = jax.lax.Precision.FASTEST
            elif max_dim <= 4096:
                matmul_precision = jax.lax.Precision.HIGH
            else:
                matmul_precision = jax.lax.Precision.DEFAULT

        if transpose:
            deq_shape = (N, K)
            stride_or, stride_oc = strides_from_shape(deq_shape)
            r_dim, c_dim = N, K
        else:
            deq_shape = (K, N)
            stride_or, stride_oc = strides_from_shape(deq_shape)
            r_dim, c_dim = K, N

        br = 128
        bc = 128

        def grid(META):
            return (cdiv(r_dim, META["BR"]), cdiv(c_dim, META["BC"]))

        if mode == "nf4":
            (w_deq,) = triton_call(
                w,
                scales,
                nf4_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nf4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "affine":
            stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
            if bits == 4:
                kernel = qmm_dequant_affine4_kernel
                values_per_word = 8
            else:
                kernel = qmm_dequant_affine8_kernel
                values_per_word = 4
            (w_deq,) = triton_call(
                w,
                scales,
                biases,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_wb0=stride_wb0,
                stride_wb1=stride_wb1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=values_per_word,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "mxfp4":
            (w_deq,) = triton_call(
                w,
                scales,
                e2m1_table,
                e8m0_exp2_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_mxfp4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "mxfp8":
            (w_deq,) = triton_call(
                w,
                scales,
                e4m3_table,
                e8m0_exp2_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_mxfp8_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=4,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "nvfp4":
            (w_deq,) = triton_call(
                w,
                scales,
                e2m1_table,
                e4m3_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nvfp4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "nvfp8":
            (w_deq,) = triton_call(
                w,
                scales,
                e4m3_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nvfp8_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=4,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        else:
            raise ValueError(f"Unsupported mode for two-stage path: {mode}")

        x_cast = x.astype(out_dtype)
        if transpose:
            dimension_numbers = (((1,), (1,)), ((), ()))
        else:
            dimension_numbers = (((1,), (0,)), ((), ()))
        out = jax.lax.dot_general(
            x_cast,
            w_deq,
            dimension_numbers=dimension_numbers,
            precision=matmul_precision,
            preferred_element_type=output_dtype,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nf4":
        kernel = qmm_nf4_kernel_large if use_large_kernel else qmm_nf4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            nf4_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (
                cdiv(M, META["BM"]),
                cdiv(N, META["BN"]),
                META["SPLIT_K"],
            ),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)
    if mode == "affine":
        stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
        bias_arg = biases if biases is not None else scales
        if bits == 4:
            kernel = qmm_affine4_kernel_large if use_large_kernel else qmm_affine4_kernel
        else:
            kernel = qmm_affine8_kernel_large if use_large_kernel else qmm_affine8_kernel

        (out,) = triton_call(
            x,
            w,
            scales,
            bias_arg,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_wb0=stride_wb0,
            stride_wb1=stride_wb1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8 if bits == 4 else 4,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            HAS_BIAS=biases is not None,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "mxfp4":
        kernel = qmm_mxfp4_kernel_large if use_large_kernel else qmm_mxfp4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e2m1_table,
            e8m0_exp2_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "mxfp8":
        kernel = qmm_mxfp8_kernel_large if use_large_kernel else qmm_mxfp8_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e4m3_table,
            e8m0_exp2_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nvfp4":
        kernel = qmm_nvfp4_kernel_large if use_large_kernel else qmm_nvfp4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e2m1_table,
            e4m3_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nvfp8":
        kernel = qmm_nvfp8_kernel_large if use_large_kernel else qmm_nvfp8_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e4m3_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    raise ValueError(f"Unsupported quantization mode for Triton: {mode}")
