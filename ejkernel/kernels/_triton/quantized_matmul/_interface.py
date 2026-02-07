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

"""Quantized matmul interface using Triton kernels.

This module provides the Triton GPU implementation of quantized matrix
multiplication. It supports "affine", "nf4", "mxfp4", "mxfp8", "nvfp4",
and "nvfp8" quantization modes with optimized fused dequantization and
matmul kernels.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl import (
    _parse_matmul_precision,
    _resolve_qparams,
    quantized_matmul_dequant_triton,
    quantized_matmul_triton,
)

#: Supported quantization modes for quantized matrix multiplication.
QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


_QMM_DEQUANT_CACHE: OrderedDict[tuple, jax.Array] = OrderedDict()
_QMM_DEQUANT_CACHE_LOCK = threading.Lock()


def _device_key(arr: jax.Array) -> tuple | None:
    try:
        dev = arr.device()
    except Exception:
        return None
    if dev is None:
        return None
    return (getattr(dev, "platform", None), getattr(dev, "id", None), str(dev))


def _dequant_cache_key(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    out_dtype,
    group_size: int,
    bits: int,
    mode: str,
    transpose: bool,
) -> tuple:
    return (
        _device_key(w),
        id(w),
        id(scales),
        id(biases) if biases is not None else None,
        w.shape,
        w.dtype,
        scales.shape,
        scales.dtype,
        biases.shape if biases is not None else None,
        biases.dtype if biases is not None else None,
        out_dtype,
        group_size,
        bits,
        mode,
        transpose,
    )


@ejit(static_argnames=["transpose", "group_size", "bits", "mode", "use_bf16"])
def _dequant_jit(
    w: Array,
    scales: Array,
    biases: Array | None,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    use_bf16: bool,
) -> Array:
    return quantized_matmul_dequant_triton(
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        use_bf16=use_bf16,
    )


@ejit(static_argnames=["transpose", "output_dtype", "precision"])
def _two_stage_matmul(
    x: Array,
    w_deq: Array,
    *,
    transpose: bool,
    output_dtype,
    precision,
) -> Array:
    if transpose:
        dimension_numbers = (((1,), (1,)), ((), ()))
    else:
        dimension_numbers = (((1,), (0,)), ((), ()))
    return jax.lax.dot_general(
        x,
        w_deq,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=output_dtype,
    )


@kernel_registry.register("quantized_matmul", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    biases: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication using Triton GPU kernels.

    This function performs fused dequantization and matrix multiplication
    using optimized Triton kernels. It supports "affine", "nf4", "mxfp4",
    "mxfp8", "nvfp4", and "nvfp8" quantization modes directly.

    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights. Shape is (N, K//values_per_word) for
            transpose=True or (K, N//values_per_word) for transpose=False,
            where values_per_word = 32 // bits.
        scales: Per-group scales array. Shape is (N, K//group_size) for
            transpose=True or (K, N//group_size) for transpose=False.
        biases: Per-group biases (required for affine mode). Must match
            scales shape. Must be None for non-affine modes.
        transpose: If True, weights are in NxK layout (transposed).
            If False, weights are in KxN layout. Default is False.
        group_size: Quantization group size. If None, uses mode defaults
            (affine/nf4: 64, mxfp4/mxfp8: 32, nvfp4/nvfp8: 16).
        bits: Bit-width per element. If None, uses mode defaults
            (affine/nf4/mxfp4/nvfp4: 4, mxfp8/nvfp8: 8). Triton supports
            bits in {4, 8} for affine mode.
        mode: Quantization mode. All supported modes use native Triton
            kernels.
        block_m: Block size for M dimension.
        block_n: Block size for N dimension.
        block_k: Block size for K dimension.
        use_bf16: If True, use BF16 for dot input tiles. If False, use FP16.
        num_warps: Triton warps per program (optional).
        num_stages: Triton pipeline stages (optional).
        split_k: Optional split-K value for parallel reduction.

    Returns:
        Matrix multiplication result of shape (M, N) in bfloat16.

    Raises:
        ValueError: If mode is "affine" but biases is None.
        ValueError: If mode is not "affine" but biases is provided.
        ValueError: If bits/group_size are invalid for the selected mode on Triton.

    Notes:
        - Unsupported shapes or invalid parameters raise an error.
        - Block sizes and split-K are selected heuristically when provided.
    """
    mode_lower = mode.lower()
    use_two_stage = os.getenv("EJKERNEL_QMM_TWO_STAGE", "1").lower() in {"1", "true", "yes", "y"}
    use_cache = os.getenv("EJKERNEL_QMM_DEQUANT_CACHE", "1").lower() in {"1", "true", "yes", "y"}

    if use_two_stage and use_cache:
        group_size_resolved, bits_resolved = _resolve_qparams(mode_lower, group_size, bits)
        M = x.shape[0]
        K = x.shape[1]
        if transpose:
            N = w.shape[0]
        else:
            N = scales.shape[1] * group_size_resolved

        use_large_kernel = M >= 4096 and N >= 4096 and K >= 4096
        if use_large_kernel:
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
            cache_limit = int(os.getenv("EJKERNEL_QMM_DEQUANT_CACHE_MAX_ITEMS", "2"))

            w_deq = None
            cache_key = None
            if cache_limit > 0:
                cache_key = _dequant_cache_key(
                    w,
                    scales,
                    biases,
                    out_dtype,
                    group_size_resolved,
                    bits_resolved,
                    mode_lower,
                    transpose,
                )
                with _QMM_DEQUANT_CACHE_LOCK:
                    w_deq = _QMM_DEQUANT_CACHE.get(cache_key)
                    if w_deq is not None:
                        _QMM_DEQUANT_CACHE.move_to_end(cache_key)

            if w_deq is None:
                w_deq = _dequant_jit(
                    w,
                    scales,
                    biases,
                    transpose=transpose,
                    group_size=group_size_resolved,
                    bits=bits_resolved,
                    mode=mode_lower,
                    use_bf16=use_bf16,
                )
                if cache_key is not None:
                    with _QMM_DEQUANT_CACHE_LOCK:
                        _QMM_DEQUANT_CACHE[cache_key] = w_deq
                        _QMM_DEQUANT_CACHE.move_to_end(cache_key)
                        while len(_QMM_DEQUANT_CACHE) > cache_limit:
                            _QMM_DEQUANT_CACHE.popitem(last=False)

            x_cast = x.astype(out_dtype)
            out = _two_stage_matmul(
                x_cast,
                w_deq,
                transpose=transpose,
                output_dtype=output_dtype,
                precision=matmul_precision,
            )
            return out.astype(jnp.bfloat16)

    out = ejit(
        func=quantized_matmul_triton,
        static_argnames=[
            "transpose",
            "group_size",
            "bits",
            "mode",
            "block_m",
            "block_n",
            "block_k",
            "use_bf16",
            "num_warps",
            "num_stages",
            "split_k",
        ],
    )(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode_lower,
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
    )
    return out.astype(jnp.bfloat16)
