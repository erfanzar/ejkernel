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
multiplication. It supports "affine" and "nf4" quantization modes with
optimized fused dequantization and matmul kernels.

For unsupported modes or shapes, this module automatically falls back to
the XLA implementation for maximum compatibility.
"""

from __future__ import annotations

from typing import Literal

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl import quantized_matmul_triton

#: Supported quantization modes for quantized matrix multiplication.
QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8", "w4a16", "w8a16"]


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
    using optimized Triton kernels. It supports "affine" and "nf4" quantization
    modes directly; other modes fall back to the XLA implementation.

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
        group_size: Quantization group size. If None, uses mode default
            (64 for affine/nf4).
        bits: Bit-width per element. If None, uses mode default (4 for
            affine/nf4). Triton supports bits in {4, 8} for affine mode.
        mode: Quantization mode. "affine" and "nf4" use native Triton
            kernels; other modes fall back to XLA.
        block_m: Block size for M dimension.
        block_n: Block size for N dimension.
        block_k: Block size for K dimension.
        use_bf16: If True, use BF16 for dot input tiles. If False, use FP16.
        num_warps: Triton warps per program (optional).
        num_stages: Triton pipeline stages (optional).
        split_k: Optional split-K value for parallel reduction.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If mode is "affine" but biases is None.
        ValueError: If mode is not "affine" but biases is provided.
        ValueError: If bits is not in {4, 8} for affine mode on Triton.

    Notes:
        - For unsupported modes or shapes, automatically falls back to XLA.
        - Block sizes and split-K are selected heuristically when provided.
    """
    try:
        return ejit(
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
            mode=mode,
            use_bf16=use_bf16,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
            split_k=split_k,
        )
    except ValueError:
        from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as quantized_matmul_xla

        return quantized_matmul_xla(
            x,
            w,
            scales,
            biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )
