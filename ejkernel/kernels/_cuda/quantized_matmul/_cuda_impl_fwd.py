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

"""Forward CUDA implementation for quantized matrix multiplication."""

from __future__ import annotations

from typing import Literal

from ejkernel.callib._ejit import ejit

from ._cuda_impl import quantized_matmul_cuda

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
    ]
)
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
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
):
    """Dispatch to the CUDA custom call implementation."""
    return quantized_matmul_cuda(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )


__all__ = ("quantized_matmul_forward",)
