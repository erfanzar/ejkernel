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

"""Backward CuTe implementation for quantized matrix multiplication."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.quantization import dequantize

from ._cute_impl_fwd import quantized_matmul_forward


def quantized_matmul_input_grad(
    dy,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
):
    """Compute gradient with respect to x.

    Uses the same CuTe QMM kernel by flipping the transpose semantic when
    possible, and falls back to exact dequantize+dot when needed.
    """
    try:
        return quantized_matmul_forward(
            dy,
            w,
            scales,
            biases,
            transpose=not transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )
    except ValueError:
        pass

    w_f = dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode)
    if transpose:
        dims = (((1,), (0,)), ((), ()))
    else:
        dims = (((1,), (1,)), ((), ()))
    return jax.lax.dot_general(dy, w_f, dims, preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
