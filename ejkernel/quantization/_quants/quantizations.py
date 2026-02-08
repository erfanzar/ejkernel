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

Supported quantization modes:
- ``affine`` with bits in ``{4, 8}``
- ``nf4`` (4-bit normal-float codebook)
- ``mxfp4`` / ``mxfp8``
- ``nvfp4`` / ``nvfp8``

Quantization axis is explicit via ``axis``:
- ``axis='row'``: group over output channels (logical weight rows)
- ``axis='col'``: group over input channels (logical weight cols)
"""

from __future__ import annotations

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
from .._utils.grouping import _quantize_to_codebook, _reshape_groups
from .._utils.qparams import (
    QuantizationAxis,
    QuantizationMode,
    normalize_axis,
    resolve_prepack_axis,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
)


def _to_quant_layout(w: jax.Array, axis: QuantizationAxis) -> jax.Array:
    """Map logical weight layout to quantization/runtime layout.

    ``axis='row'`` maps logical ``(..., out_features, in_features)`` to
    ``(..., in_features, out_features)`` so grouping along the last dimension
    groups over output channels.
    """
    if w.ndim < 2:
        raise ValueError("quantize expects inputs with two or more dimensions.")
    return jnp.swapaxes(w, -2, -1) if axis == "row" else w


def quantize(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis = "row",
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Quantize weights into packed uint32 codes.

    Returns:
      - affine: ``(w_q, scales, zeros)`` with dequant formula ``(q - zero) * scale``
      - nf4/mxfp4/mxfp8/nvfp4/nvfp8: ``(w_q, scales)``
    """
    axis = normalize_axis(axis)
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)

    w_layout = _to_quant_layout(w, axis)
    w_groups, _ = _reshape_groups(w_layout, group_size)

    if mode == "affine":
        qmax = 2**bits - 1
        alpha = jnp.max(w_groups, axis=-1)
        beta = jnp.min(w_groups, axis=-1)

        scale = (alpha - beta) / qmax
        scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
        zero = -beta / scale

        q = jnp.round(w_groups / scale[..., None] + zero[..., None])
        q = jnp.clip(q, 0, qmax).astype(jnp.uint32)
        packed = _pack_bits(q.reshape(*w_layout.shape[:-1], -1), bits)
        return packed, scale.astype(w.dtype), zero.astype(w.dtype)

    if mode == "nf4":
        codebook = _get_nf4_table()
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        scale = jnp.where(max_abs == 0, jnp.ones_like(max_abs), max_abs)
        normalized = w_groups / scale[..., None]
        q = _quantize_to_codebook(normalized, codebook)
        packed = _pack_bits(q.reshape(*w_layout.shape[:-1], -1), bits)
        return packed, scale.astype(w.dtype)

    if mode in {"mxfp4", "mxfp8"}:
        max_abs = jnp.max(jnp.abs(w_groups), axis=-1)
        if bits == 4:
            vmax = _get_e2m1_max()
            codebook, _ = _get_e2m1_table()
        else:
            vmax = _get_e4m3_max()
            codebook = _get_e4m3_table_q()

        exp = jnp.where(max_abs > 0, jnp.ceil(jnp.log2(max_abs / vmax)), 0.0)
        exp = jnp.clip(exp, -128, 127).astype(jnp.int8)

        scale = jnp.exp2(exp.astype(jnp.float32))
        scale = jnp.where(scale == 0, 1.0, scale)
        normalized = w_groups / scale[..., None]

        q = _quantize_to_codebook(normalized, codebook)
        packed = _pack_bits(q.reshape(*w_layout.shape[:-1], -1), bits)
        return packed, exp.astype(jnp.uint8)

    # mode in {"nvfp4", "nvfp8"}
    if bits == 4:
        vmax = _get_e2m1_max()
        q_codebook, _ = _get_e2m1_table()
    else:
        vmax = _get_e4m3_max()
        q_codebook = _get_e4m3_table_q()

    scale_raw = jnp.where(jnp.max(jnp.abs(w_groups), axis=-1) > 0, jnp.max(jnp.abs(w_groups), axis=-1) / vmax, 0.0)
    scale_codebook = _get_e4m3_table_q()
    scale_q = _quantize_to_codebook(scale_raw, scale_codebook).astype(jnp.uint32)

    e4m3_table, _ = _get_e4m3_table()
    scale = e4m3_table[scale_q.astype(jnp.int32)]
    scale = jnp.where(scale == 0, 1.0, scale)

    normalized = w_groups / scale[..., None]
    q = _quantize_to_codebook(normalized, q_codebook)
    packed = _pack_bits(q.reshape(*w_layout.shape[:-1], -1), bits)
    return packed, scale_q.astype(jnp.uint8)


def dequantize(
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None = None,
    *,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis = "row",
) -> jax.Array:
    """Dequantize packed weights from ``quantize``.

    For affine mode, metadata is ``zeros`` and the dequantization formula is
    ``(q - zero) * scale``.
    """
    axis = normalize_axis(axis)
    del axis  # kept for API symmetry and future layout-aware validation.
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine dequantize requires `zeros`.")

        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.float32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        out = (q - zeros[..., None]) * scales[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode == "nf4":
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)
        table = _get_nf4_table()
        vals = table[q]
        out = vals * scales[..., None]
        return out.reshape(*scales.shape[:-1], n)

    if mode in {"mxfp4", "mxfp8"}:
        n_groups = scales.shape[-1]
        n = n_groups * group_size
        q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
        q = q.reshape(*scales.shape[:-1], n_groups, group_size)

        if bits == 4:
            table, _ = _get_e2m1_table()
        else:
            table, _ = _get_e4m3_table()

        vals = table[q]
        exp = scales.astype(jnp.int8).astype(jnp.float32)
        scale = jnp.exp2(exp)
        out = vals * scale[..., None]
        return out.reshape(*scales.shape[:-1], n)

    # mode in {"nvfp4", "nvfp8"}
    n_groups = scales.shape[-1]
    n = n_groups * group_size
    q = _unpack_bits(w_q, n, bits).astype(jnp.int32)
    q = q.reshape(*scales.shape[:-1], n_groups, group_size)

    e4m3_table, _ = _get_e4m3_table()
    if bits == 4:
        q_table, _ = _get_e2m1_table()
    else:
        q_table = e4m3_table

    vals = q_table[q]
    scale = e4m3_table[scales.astype(jnp.int32)]
    out = vals * scale[..., None]
    return out.reshape(*scales.shape[:-1], n)


@ejit(static_argnames=["transpose", "group_size", "bits", "mode", "axis"])
def quantized_matmul(
    x: jax.Array,
    w: jax.Array,
    /,
    scales: jax.Array,
    zeros: jax.Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
) -> jax.Array:
    """Dense reference quantized matmul: dequantize then matmul."""
    if axis is not None:
        _, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)

    # Runtime layout determines dequant axis convention.
    dequant_axis: QuantizationAxis = "col" if transpose else "row"
    w_f = dequantize(
        w,
        scales,
        zeros,
        group_size=group_size,
        bits=bits,
        mode=mode,
        axis=dequant_axis,
    )
    return x @ w_f.T if transpose else x @ w_f


def prepack_quantized_weights(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    transpose: bool = True,
    axis: QuantizationAxis | None = None,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
    """Prepack logical ``(out_features, in_features)`` weights.

    Backward compatibility:
      - if ``axis`` is omitted, ``transpose=True`` maps to ``axis='row'``
      - if ``axis`` is omitted, ``transpose=False`` maps to ``axis='col'``
    """
    axis = resolve_prepack_axis(axis=axis, transpose=transpose)
    return quantize(w, group_size=group_size, bits=bits, mode=mode, axis=axis)
