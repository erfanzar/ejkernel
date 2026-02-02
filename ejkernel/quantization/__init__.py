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

"""Quantization utilities for weight compression and efficient inference.

This module provides tools for quantizing neural network weights to reduce
memory footprint and enable efficient inference. It supports multiple
quantization formats optimized for different use cases:

Supported Quantization Modes:
    - **affine**: Linear scale+bias quantization with configurable bit-width
      (2-8 bits). Best accuracy, most flexible.
    - **nf4**: 4-bit NormalFloat codebook optimized for normally-distributed
      weights. Used in QLoRA.
    - **mxfp4**: Microscaling FP4 (E2M1) with E8M0 shared exponent. Low memory,
      moderate accuracy.
    - **mxfp8**: Microscaling FP8 (E4M3) with E8M0 shared exponent. Good
      accuracy/memory tradeoff.
    - **nvfp4**: NVIDIA FP4 (E2M1) with E4M3 per-group scale. Hardware-friendly.
    - **nvfp8**: NVIDIA FP8 (E4M3) with E4M3 per-group scale. Hardware-friendly.
    - **w4a16**: 4-bit affine quantization with per-channel scale (weights only).
    - **w8a16**: 8-bit affine quantization with per-channel scale (weights only).

Basic Usage:
    >>> from ejkernel.quantization import quantize, dequantize, prepack_quantized_weights
    >>>
    >>> # Quantize weights
    >>> w_q, scales, biases = quantize(weights, mode="affine", bits=4)
    >>>
    >>> # Dequantize for verification
    >>> w_reconstructed = dequantize(w_q, scales, biases, mode="affine", bits=4)
    >>>
    >>> # For optimized matmul kernels, use prepack_quantized_weights
    >>> w_q, scales, biases = prepack_quantized_weights(weights, mode="affine")

For fused quantized matmul kernels with better performance, see
`ejkernel.modules.operations.quantized_matmul`.
"""

from ._quants.quantizations import (
    QuantizationMode,
    dequantize,
    prepack_quantized_weights,
    quantize,
)
from ._quants.quantizations import quantized_matmul as dense_quantized_matmul

__all__ = [
    "QuantizationMode",
    "dense_quantized_matmul",
    "dequantize",
    "prepack_quantized_weights",
    "quantize",
]
