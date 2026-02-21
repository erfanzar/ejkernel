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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, GemvMode, QuantizationAxis, RevSplitKMode
from ._xla_impl_fwd import quantized_matmul as _quantized_matmul_impl


@kernel_registry.register("quantized_matmul", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    *,
    tpu_path: str | None = None,
    allow_dense_fallback: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication using XLA.
    This function provides a portable XLA implementation of quantized matmul
    for explicit modes ``affine``, ``nf4``, ``mxfp4``, ``mxfp8``, ``nvfp4``,
    and ``nvfp8``.
    When possible, it uses a blocked algorithm with fused dequantization for
    better performance. For incompatible shapes or bit-widths, it falls back
    to a simple dequantize+matmul approach.
    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights produced by quantize() or prepack_quantized_weights().
            For transpose=True, shape is (N, ceil(K/values_per_word)).
            For transpose=False, shape is (K, ceil(N/values_per_word)).
        scales: Per-group scales array. Shape depends on mode:
            - affine/nf4: float scales, shape (N, K//group_size) or (K, N//group_size)
            - mxfp4/mxfp8: uint8 E8M0 exponents
            - nvfp4/nvfp8: uint8 E4M3 scale codes
        zeros: Per-group affine zero-points (canonical affine metadata). Must
            have the same shape as scales. Must be None for non-affine modes.
            Internally converted to additive offsets for blocked fused kernels.
        transpose: If True, weights are in NxK layout (transposed) and the
            computation is x @ w.T. If False, weights are in KxN layout and
            the computation is x @ w. Default is False.
        group_size: Number of elements per quantization group. If None, uses
            mode default (64 for affine/nf4, 32 for mxfp4/mxfp8, 16 for nvfp4/nvfp8).
        bits: Bit-width per quantized element. Honored for affine only
            (supported values {4,8}); ignored for explicit non-affine modes.
        mode: Quantization mode determining the dequantization formula:
            - "affine": ``(q - zero) * scale``
            - "nf4": 4-bit NormalFloat codebook
            - "mxfp4"/"mxfp8": Microscaling FP4/FP8
            - "nvfp4"/"nvfp8": NVIDIA microscaling FP4/FP8
        block_m: Tile size for M dimension in blocked algorithm. Default 128.
        block_n: Tile size for N dimension in blocked algorithm. Default 128.
        block_k: Tile size for K dimension in blocked algorithm. Default 64.
        use_bf16: If True, use BF16 for intermediate dot products.
            If False, use FP16. Default is True.
        num_warps: Ignored in XLA path (Triton-only).
        num_stages: Ignored in XLA path (Triton-only).
        split_k: Ignored in XLA path (Triton-only).
    Returns:
        Matrix multiplication result of shape (M, N) in float32.
    Raises:
        ValueError: If mode is "affine" and zeros is not provided.
        ValueError: If mode is not "affine" but zeros is provided.
        ValueError: If parameters are invalid for the selected mode.
    Notes:
        - The blocked algorithm is used when bits is 4 or 8 and block sizes
          are compatible with group_size. Otherwise, falls back to
          dequantize+matmul.
        - This implementation serves as the fallback for other backends
          when shapes or runtime constraints are unsupported.
    """
    del tpu_path
    return _quantized_matmul_impl(
        x,
        w,
        scales,
        zeros,
        transpose,
        group_size,
        bits,
        mode,
        axis,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        allow_dense_fallback=allow_dense_fallback,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
    )


__all__ = ("quantized_matmul",)
