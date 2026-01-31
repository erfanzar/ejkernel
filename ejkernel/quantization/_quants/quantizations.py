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

"""Quantization functions for weight compression.

This module provides functions to quantize, dequantize, and perform matrix
multiplication with quantized weights. It supports multiple quantization modes:

- **affine**: Linear scale+bias quantization with configurable bit-width
- **nf4**: 4-bit NormalFloat codebook optimized for normal distributions
- **mxfp4**: Microscaling FP4 (E2M1) with E8M0 shared exponent
- **mxfp8**: Microscaling FP8 (E4M3) with E8M0 shared exponent
- **nvfp4**: NVIDIA FP4 (E2M1) with E4M3 per-group scale
- **nvfp8**: NVIDIA FP8 (E4M3) with E4M3 per-group scale

All modes pack quantized values into uint32 arrays using LSB-first ordering,
compatible with the MLX quantization format.
"""

from __future__ import annotations

from typing import Literal

import jax
from jax import numpy as jnp

from ejkernel.callib._ejit import ejit

from .._utils.bitpack import _pack_bits, _unpack_bits
from .._utils.fp_tables import (
    _get_e2m1_max,
    _get_e2m1_table,
    _get_e4m3_max,
    _get_e4m3_table,
    _get_e4m3_table_q,
    _get_nf4_table,
)
from .._utils.grouping import _quantize_to_codebook, _require_bits, _reshape_groups

#: Supported quantization modes.
QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


def quantize(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize a weight tensor into packed uint32 codes with per-group scales and biases.

    Splits the last dimension of `w` into groups, computes per-group quantization
    parameters, maps values to integer codes, and packs the codes into uint32 words
    using LSB-first ordering (MLX-compatible).

    Args:
        w: Weight tensor to quantize, with at least 2 dimensions. The last
            dimension must be divisible by `group_size`.
        group_size: Number of elements per quantization group. If None, a
            mode-specific default is used (affine: 64, mxfp4/mxfp8: 32,
            nvfp4/nvfp8: 16, nf4: 64). Allowed values depend on mode:

            - affine: {8, 16, 32, 64, 128, 256, 512}
            - mxfp4/mxfp8: 32 (fixed)
            - nvfp4/nvfp8: 16 (fixed)
            - nf4: any divisor of the last dimension
        bits: Bit-width per quantized element. If None, a mode-specific default
            is used. Allowed values depend on mode:

            - affine: {2, 3, 4, 5, 6, 7, 8} (default 4)
            - mxfp4/nvfp4/nf4: 4 (fixed)
            - mxfp8/nvfp8: 8 (fixed)
        mode: Quantization mode to use. One of:

            - ``"affine"``: Linear scale+bias quantization. Computes per-group
              min/max and maps values to [0, 2^bits - 1].
            - ``"mxfp4"``: Microscaling FP4 (E2M1) with shared E8M0 exponent.
            - ``"mxfp8"``: Microscaling FP8 (E4M3) with shared E8M0 exponent.
            - ``"nvfp4"``: NVIDIA FP4 (E2M1) with E4M3 per-group scale.
            - ``"nvfp8"``: NVIDIA FP8 (E4M3) with E4M3 per-group scale.
            - ``"nf4"``: 4-bit NormalFloat codebook (QLoRA-style).

    Returns:
        A tuple whose contents depend on the mode:

        - **affine**: ``(w_q, scales, biases)`` where scales and biases have
          the same dtype as `w`.
        - **all other modes**: ``(w_q, scales)`` where scales is uint8.

        In all cases, ``w_q`` is a packed uint32 array with elements stored
        LSB-first.

    Raises:
        ValueError: If `group_size` is not valid for the selected mode.
        ValueError: If `bits` is not valid for the selected mode.
        ValueError: If the last dimension of `w` is not divisible by `group_size`.
        ValueError: If `mode` is not a recognized quantization mode.

    Note:
        Estimated reconstruction error (quantize then dequantize) on a float32
        linear ramp input, shape (512, 1024) unless noted:

        - mxfp4:  MAE ~0.0417, RMSE ~0.0546, max ~0.125
        - mxfp8:  MAE ~0.0104, RMSE ~0.0136, max ~0.03125
        - nvfp4:  MAE ~0.0121, RMSE ~0.0166, max ~0.0469 (shape (512, 512))
        - nvfp8:  MAE ~0.0105, RMSE ~0.0137, max ~0.03125 (shape (512, 512))
        - affine: MAE ~3.9e-06, RMSE ~4.6e-06, max ~7.7e-06
        - nf4:    MAE ~1.2e-04, RMSE ~1.4e-04, max ~2.4e-04

        Errors depend on input distribution, dtype, and group_size; treat these
        as rough sanity checks, not guarantees.
    """
    mode = mode.lower()
    if mode == "affine":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else _require_bits(bits, {2, 3, 4, 5, 6, 7, 8})
        if group_size not in {8, 16, 32, 64, 128, 256, 512}:
            raise ValueError("affine mode supports group_size in {8,16,32,64,128,256,512}.")

        w_groups, _ = _reshape_groups(w, group_size)
        alpha = jnp.max(w_groups, axis=-1)
        beta = jnp.min(w_groups, axis=-1)
        scale = (alpha - beta) / (2**bits - 1)
        scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)

        q = jnp.round((w_groups - beta[..., None]) / scale[..., None])
        q = jnp.clip(q, 0, 2**bits - 1).astype(jnp.uint32)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, scale.astype(w.dtype), beta.astype(w.dtype)

    if mode == "mxfp4":
        group_size = 32 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 32 or bits != 4:
            raise ValueError("mxfp4 requires group_size=32 and bits=4.")
        w_groups, _ = _reshape_groups(w, group_size)
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        e2m1_max = _get_e2m1_max()
        exp = jnp.where(
            max_abs > 0,
            jnp.ceil(jnp.log2(max_abs / e2m1_max)),
            0.0,
        )
        exp = jnp.clip(exp, -128, 127).astype(jnp.int8)
        scale = jnp.exp2(exp.astype(jnp.float32))
        scale = jnp.where(scale == 0, 1.0, scale)

        normalized = w_groups / scale[..., None]
        e2m1_table, _ = _get_e2m1_table()
        q = _quantize_to_codebook(normalized, e2m1_table)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, exp.astype(jnp.uint8)

    if mode == "mxfp8":
        group_size = 32 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 32 or bits != 8:
            raise ValueError("mxfp8 requires group_size=32 and bits=8.")
        w_groups, _ = _reshape_groups(w, group_size)
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        e4m3_max = _get_e4m3_max()
        exp = jnp.where(
            max_abs > 0,
            jnp.ceil(jnp.log2(max_abs / e4m3_max)),
            0.0,
        )
        exp = jnp.clip(exp, -128, 127).astype(jnp.int8)
        scale = jnp.exp2(exp.astype(jnp.float32))
        scale = jnp.where(scale == 0, 1.0, scale)

        normalized = w_groups / scale[..., None]
        e4m3_table_q = _get_e4m3_table_q()
        q = _quantize_to_codebook(normalized, e4m3_table_q)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, exp.astype(jnp.uint8)

    if mode == "nvfp4":
        group_size = 16 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 16 or bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4.")
        w_groups, _ = _reshape_groups(w, group_size)
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        e2m1_max = _get_e2m1_max()
        scale_raw = jnp.where(max_abs > 0, max_abs / e2m1_max, 0.0)
        e4m3_table_q = _get_e4m3_table_q()
        scale_q = _quantize_to_codebook(scale_raw, e4m3_table_q).astype(jnp.uint32)
        e4m3_table, _ = _get_e4m3_table()
        scale = e4m3_table[scale_q.astype(jnp.int32)]
        scale = jnp.where(scale == 0, 1.0, scale)

        normalized = w_groups / scale[..., None]
        e2m1_table, _ = _get_e2m1_table()
        q = _quantize_to_codebook(normalized, e2m1_table)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, scale_q.astype(jnp.uint8)

    if mode == "nvfp8":
        group_size = 16 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 16 or bits != 8:
            raise ValueError("nvfp8 requires group_size=16 and bits=8.")
        w_groups, _ = _reshape_groups(w, group_size)
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        e4m3_max = _get_e4m3_max()
        scale_raw = jnp.where(max_abs > 0, max_abs / e4m3_max, 0.0)
        e4m3_table_q = _get_e4m3_table_q()
        scale_q = _quantize_to_codebook(scale_raw, e4m3_table_q).astype(jnp.uint32)
        e4m3_table, _ = _get_e4m3_table()
        scale = e4m3_table[scale_q.astype(jnp.int32)]
        scale = jnp.where(scale == 0, 1.0, scale)

        normalized = w_groups / scale[..., None]
        q = _quantize_to_codebook(normalized, e4m3_table_q)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, scale_q.astype(jnp.uint8)

    if mode == "nf4":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        w_groups, _ = _reshape_groups(w, group_size)
        absmax = jnp.max(jnp.abs(w_groups), axis=-1)
        normalized = w_groups / (absmax[..., None] + jnp.finfo(w.dtype).tiny)
        nf4_table = _get_nf4_table()
        q = _quantize_to_codebook(normalized, nf4_table)
        packed = _pack_bits(q.reshape(*w.shape[:-1], -1), bits)
        return packed, absmax.astype(w.dtype)

    raise ValueError(f"Unsupported quantization mode: {mode}")


def dequantize(
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
) -> jax.Array:
    """Dequantize packed codes produced by `quantize`.

    Reconstructs the original floating-point weights from quantized codes
    and per-group parameters. The dequantization formula depends on the mode:

    - **affine**: `w = q * scale + bias`
    - **nf4**: `w = nf4_table[q] * scale`
    - **mxfp4**: `w = e2m1_table[q] * 2^exp`
    - **mxfp8**: `w = e4m3_table[q] * 2^exp`
    - **nvfp4**: `w = e2m1_table[q] * e4m3_table[scale]`
    - **nvfp8**: `w = e4m3_table[q] * e4m3_table[scale]`

    Args:
        w_q: Packed uint32 codes produced by `quantize()`.
        scales: Per-group scales. Dtype depends on mode:
            - affine/nf4: float (same as original weights)
            - mxfp4/mxfp8: uint8 (E8M0 exponent)
            - nvfp4/nvfp8: uint8 (E4M3 scale code)
        biases: Per-group biases (required and only valid for affine mode).
        group_size: Number of elements per quantization group.
            Must match the value used in `quantize()`.
        bits: Bit-width per quantized element. Must match `quantize()`.
        mode: Quantization mode. Must match `quantize()`.

    Returns:
        Reconstructed float32 array with shape (*scales.shape[:-1], n) where
        n = scales.shape[-1] * group_size.

    Raises:
        ValueError: If mode is "affine" but biases is None.
        ValueError: If parameters are invalid for the selected mode.
    """
    mode = mode.lower()
    if mode == "affine":
        if biases is None:
            raise ValueError("affine dequantize requires biases.")
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else _require_bits(bits, {2, 3, 4, 5, 6, 7, 8})
        if group_size not in {32, 64, 128}:
            raise ValueError("affine mode supports group_size in {32, 64, 128}.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.float32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        out = q * scales[..., None] + biases[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "mxfp4":
        group_size = 32 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 32 or bits != 4:
            raise ValueError("mxfp4 requires group_size=32 and bits=4.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        e2m1_table, _ = _get_e2m1_table()
        vals = e2m1_table[q]
        exp = scales.astype(jnp.int8).astype(jnp.float32)
        scale = jnp.exp2(exp)
        out = vals * scale[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "mxfp8":
        group_size = 32 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 32 or bits != 8:
            raise ValueError("mxfp8 requires group_size=32 and bits=8.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        e4m3_table, _ = _get_e4m3_table()
        vals = e4m3_table[q]
        exp = scales.astype(jnp.int8).astype(jnp.float32)
        scale = jnp.exp2(exp)
        out = vals * scale[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "nvfp4":
        group_size = 16 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 16 or bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        e2m1_table, _ = _get_e2m1_table()
        vals = e2m1_table[q]
        e4m3_table, _ = _get_e4m3_table()
        scale = e4m3_table[scales.astype(jnp.int32)]
        out = vals * scale[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "nvfp8":
        group_size = 16 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 16 or bits != 8:
            raise ValueError("nvfp8 requires group_size=16 and bits=8.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        e4m3_table, _ = _get_e4m3_table()
        vals = e4m3_table[q]
        scale = e4m3_table[scales.astype(jnp.int32)]
        out = vals * scale[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "nf4":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        nf4_table = _get_nf4_table()
        vals = nf4_table[q]
        out = vals * scales[..., None]
        return out.reshape(*scales.shape[:-1], n)

    raise ValueError(f"Unsupported quantization mode: {mode}")


@ejit(static_argnames=["transpose", "group_size", "bits", "mode"])
def quantized_matmul(
    x: jax.Array,
    w: jax.Array,
    /,
    scales: jax.Array,
    biases: jax.Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
) -> jax.Array:
    """Perform matrix multiplication with quantized weights (dense implementation).

    This is a simple dequantize-then-matmul implementation. For better performance
    on GPU, use `ejkernel.modules.operations.quantized_matmul` which provides
    fused dequantization kernels.

    The operation computes:
        - If transpose=True: `x @ dequantize(w, scales, biases).T`
        - If transpose=False: `x @ dequantize(w, scales, biases)`

    Args:
        x: Input activation matrix of shape (M, K).
        w: Packed uint32 weights produced by `quantize()`.
        scales: Per-group scales for dequantization.
        biases: Per-group biases (required for affine mode only).
        transpose: If True, weights are in NxK layout (transposed).
            If False, weights are in KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element.
        mode: Quantization mode.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.
    """
    w_f = dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode)
    return x @ w_f.T if transpose else x @ w_f


def prepack_quantized_weights(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    transpose: bool = True,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Prepack weights for fast quantized matmul kernels.

    This function quantizes weights in the optimal layout for fused quantized
    matmul kernels. By default, it transposes the input weights so that the
    packed layout is KxN, which allows the Triton and XLA kernels to read
    weights contiguously along the K dimension.

    Args:
        w: Weight matrix to quantize. Typically shape (N, K) where N is the
            output dimension and K is the input dimension.
        group_size: Number of elements per quantization group. If None, uses
            mode-specific default.
        bits: Bit-width per quantized element. If None, uses mode default.
        mode: Quantization mode (affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8).
        transpose: If True (default), transpose `w` before quantization to
            produce KxN packed layout. If False, quantize `w` directly
            (use when w is already in KxN layout).

    Returns:
        For affine mode: (w_q, scales, biases) tuple
        For other modes: (w_q, scales) tuple

    Example:
        >>> # Typical usage: weights are (N, K), we want KxN packed layout
        >>> w_q, scales, biases = prepack_quantized_weights(weights, mode="affine")
        >>> # Then call quantized_matmul with transpose=False
        >>> output = quantized_matmul(x, w_q, scales, biases, transpose=False)
    """
    w_in = w.T if transpose else w
    return quantize(w_in, group_size=group_size, bits=bits, mode=mode)
