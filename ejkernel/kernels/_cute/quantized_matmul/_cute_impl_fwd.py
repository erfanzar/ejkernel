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

"""Forward CuTe implementation for quantized matrix multiplication."""

from __future__ import annotations

from typing import Literal

from ejkernel.kernels._xla.quantized_matmul._interface import _resolve_qparams

from ._cute_impl import get_cute_qmm_call

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


def quantized_matmul_forward(
    x,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
):
    """Forward CuTe QMM."""
    mode_lower = mode.lower()
    group_size_resolved, bits_resolved = _resolve_qparams(mode_lower, group_size, bits)

    if mode_lower == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires biases.")
    if mode_lower != "affine" and biases is not None:
        raise ValueError("biases must be None for non-affine modes.")

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


__all__ = ("quantized_matmul_forward",)
