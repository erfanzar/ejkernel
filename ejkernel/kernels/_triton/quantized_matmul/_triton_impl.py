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
for "affine" and "nf4" quantization modes.

The kernels use split-K parallelism for improved performance on small M
dimensions and support both transposed (NxK) and non-transposed (KxN)
weight layouts.
"""

from __future__ import annotations

import math
from typing import Literal

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import cdiv, strides_from_shape, triton_call
from ejkernel.quantization._utils.fp_tables import _get_nf4_table
from ejkernel.quantization._utils.grouping import _require_bits

#: Supported quantization modes for Triton kernels.
QuantizationMode = Literal["affine", "nf4"]

_NF4_TABLE = _get_nf4_table()


@triton.jit
def _nf4_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 4-bit NF4 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


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


def _qmm_autotune_configs() -> list[triton.Config]:
    """Generate autotune configurations for quantized matmul kernels.

    Produces a set of Triton configurations covering various block sizes,
    warp counts, pipeline stages, and split-K values. Configurations that
    exceed the shared memory limit (96 KB) are filtered out.

    Returns:
        List of triton.Config instances for autotuning.
    """
    configs: list[triton.Config] = []
    bm_choices = (64, 128, 256)
    bn_choices = (64, 128, 256)
    bk_choices = (32, 64, 128)
    split_ks = (1, 2, 4)
    smem_limit = 96 * 1024

    for bm in bm_choices:
        for bn in bn_choices:
            for bk in bk_choices:
                if bk >= 128:
                    stages_choices = (2,)
                elif bk >= 64:
                    stages_choices = (2, 3)
                else:
                    stages_choices = (2,)

                for num_stages in stages_choices:
                    smem = (bm * bk + bk * bn) * 2 * num_stages
                    if smem > smem_limit:
                        continue

                    if bm >= 256 or bn >= 256:
                        warps = (4, 8)
                    elif bm >= 128 or bn >= 128:
                        warps = (4,)
                    else:
                        warps = (2, 4)

                    for num_warps in warps:
                        for split_k in split_ks:
                            if bk >= 128 and (bm >= 256 or bn >= 256):
                                continue
                            configs.append(
                                triton.Config(
                                    {"BM": bm, "BN": bn, "BK": bk, "SPLIT_K": split_k},
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )
    return configs


_QMM_AUTOTUNE_CONFIGS = _qmm_autotune_configs()


@triton.autotune(configs=_QMM_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
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
            shifts = tl.arange(0, VALUES_PER_WORD) * 4
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            ws = tl.trans(ws)
        else:
            word_offsets = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask[None, :],
                other=0,
            )
            shifts = tl.arange(0, VALUES_PER_WORD) * 4
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            group_idx = offs_n // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
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


@triton.autotune(configs=_QMM_AUTOTUNE_CONFIGS, key=["M", "N", "K", "BITS"])
@triton.jit
def qmm_affine_kernel(
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
    BITS: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = True,
):
    """Fused affine dequantization and matrix multiplication Triton kernel.

    Performs x @ dequant(w) where w is packed in affine quantization format.
    Dequantization applies: w_float = w_int * scale + bias (when HAS_BIAS)
    or w_float = w_int * scale (when not HAS_BIAS).

    Supports 4-bit and 8-bit quantization with per-group scale and bias
    factors. Multiple quantized values are packed into uint32 words.

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
        BITS: Bit-width per quantized element (4 or 8).
        VALUES_PER_WORD: Number of quantized values per uint32 (32 // BITS).
        BM, BK, BN: Block tile sizes for M, K, N dimensions.
        SPLIT_K: Split-K parallelism factor.
        USE_BF16: If True, use BF16 for dot product tiles; otherwise FP16.
        TRANSPOSE: If True, weights are in NxK layout; otherwise KxN.
        HAS_BIAS: If True, apply per-group bias during dequantization.
    """
    tl.static_assert((BITS == 4) | (BITS == 8))

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = (1 << BITS) - 1

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
            shifts = tl.arange(0, VALUES_PER_WORD) * BITS
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            ws = tl.trans(ws)
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                wb = tl.trans(wb)
        else:
            word_offsets = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask[None, :],
                other=0,
            )
            shifts = tl.arange(0, VALUES_PER_WORD) * BITS
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            group_idx = offs_n // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
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


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve and validate quantization parameters for Triton kernels.

    Applies mode-specific defaults and validates that the parameters are
    compatible with the Triton kernel implementations.

    Args:
        mode: Quantization mode ("affine" or "nf4").
        group_size: Number of elements per quantization group, or None for default.
        bits: Bit-width per quantized element, or None for default.

    Returns:
        Tuple of (resolved_group_size, resolved_bits).

    Raises:
        ValueError: If mode is not supported by Triton kernels.
        ValueError: If group_size is not in {32, 64, 128} for affine mode.
        ValueError: If bits != 4 for nf4 mode.
    """
    mode = mode.lower()
    if mode in ("w4a16", "w8a16"):
        raise ValueError("w4a16/w8a16 are not supported by the Triton backend.")
    if mode == "affine":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else _require_bits(bits, {2, 3, 4, 5, 6, 7, 8})
        if group_size not in {32, 64, 128}:
            raise ValueError("affine mode supports group_size in {32, 64, 128}.")
        return group_size, bits
    if mode == "nf4":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        return group_size, bits
    raise ValueError(f"Unsupported quantization mode for Triton: {mode}")


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
        biases: Per-group biases array (optional).
        transpose: If True, weights are in NxK layout; if False, KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element.

    Returns:
        Tuple of (M, K, N) dimensions for the matmul operation.

    Raises:
        ValueError: If any input is not 2D.
        ValueError: If packed weight shape doesn't match expected dimensions.
        ValueError: If scales/biases shapes are inconsistent.
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
) -> jax.Array:
    """Execute quantized matmul using Triton GPU kernels.

    This is the core Triton implementation that dispatches to either the
    NF4 or affine quantization kernel based on the mode parameter. The
    kernels perform fused dequantization and matmul for optimal performance.

    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights. For transpose=True, shape is
            (N, ceil(K/values_per_word)). For transpose=False, shape is
            (K, ceil(N/values_per_word)), where values_per_word = 32 // bits.
        scales: Per-group scales. Shape is (N, K//group_size) for
            transpose=True or (K, N//group_size) for transpose=False.
        biases: Per-group biases (required for affine mode only). Must have
            the same shape as scales.
        transpose: If True, weights are stored in NxK layout and the kernel
            computes x @ w.T. If False, weights are in KxN layout and the
            kernel computes x @ w. Default is False.
        group_size: Number of elements per quantization group. If None,
            defaults to 64 for both affine and nf4 modes.
        bits: Bit-width per quantized element. If None, defaults to 4.
            Affine mode supports {4, 8}; nf4 mode requires 4.
        mode: Quantization mode. Either "affine" (linear scale+bias) or
            "nf4" (4-bit NormalFloat codebook).
        use_bf16: If True, use BF16 for dot product input tiles.
            If False, use FP16. Default is True.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If mode is "affine" but biases is None.
        ValueError: If mode is not "affine" but biases is provided.
        ValueError: If bits is not in {4, 8} for affine mode.
        ValueError: If input shapes are invalid or inconsistent.
    """
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)

    if use_bf16 and getattr(x, "dtype", None) == jnp.float16:
        use_bf16 = False

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires biases.")
    if mode != "affine" and biases is not None:
        raise ValueError("biases must be None for non-affine modes.")

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

    stride_xm, stride_xk = strides_from_shape(x.shape)
    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)
    stride_om, stride_on = strides_from_shape((M, N))

    num_warps = int(num_warps) if num_warps is not None else 4
    num_stages = int(num_stages) if num_stages is not None else 3

    if mode == "nf4":
        (out,) = triton_call(
            x,
            w,
            scales,
            _NF4_TABLE,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=qmm_nf4_kernel,
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
        )
        return out

    stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
    bias_arg = biases if biases is not None else scales

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
        kernel=qmm_affine_kernel,
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
        BITS=bits,
        VALUES_PER_WORD=32 // bits,
        USE_BF16=use_bf16,
        TRANSPOSE=transpose,
        HAS_BIAS=biases is not None,
    )
    return out
