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

"""CuTe DSL quantized matrix multiplication interface."""

from __future__ import annotations

from typing import Literal

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._xla.quantized_matmul._interface import _resolve_qparams

from ._cute_impl import get_cute_qmm_call

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


@kernel_registry.register("quantized_matmul", Platform.CUTE, Backend.GPU)
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
    """Quantized matrix multiplication using CuTe DSL.

    This implementation fuses bit-unpacking, dequantization, and GEMM in a
    correctness-first kernel. It supports all quantization modes.

    Notes:
        - This backend targets NVIDIA GPUs only.
    """
    mode_lower = mode.lower()
    group_size_resolved, bits_resolved = _resolve_qparams(mode_lower, group_size, bits)

    if mode_lower == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires biases.")
    if mode_lower != "affine" and biases is not None:
        raise ValueError("biases must be None for non-affine modes.")

    del num_warps, num_stages, split_k

    dev = None
    try:
        dev = x.device()
    except Exception:
        dev = None
    if dev is not None and getattr(dev, "platform", None) != "gpu":
        raise ValueError("CUTE quantized_matmul requires GPU backend.")

    call = get_cute_qmm_call(
        x=x,
        w_q=w,
        scales=scales,
        biases=biases,
        mode=mode_lower,
        bits=bits_resolved,
        group_size=group_size_resolved,
        transpose=transpose,
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    if biases is not None:
        return call(x, w, scales, biases)
    return call(x, w, scales)
