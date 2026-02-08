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

"""CUDA quantized matrix multiplication interface."""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl_bwd import quantized_matmul_input_grad
from ._cuda_impl_fwd import quantized_matmul_forward

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _operate(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
):
    return quantized_matmul_forward(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )


def _operate_fwd(
    x,
    w,
    scales,
    biases,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
):
    out = quantized_matmul_forward(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return out, (w, scales, biases)


def _operate_bwd(
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    residual,
    grad_out,
):
    w, scales, biases = residual
    grad_x = quantized_matmul_input_grad(
        grad_out,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return grad_x, None, None, None


_operate.defvjp(_operate_fwd, _operate_bwd)


@kernel_registry.register("quantized_matmul", Platform.CUDA, Backend.GPU)
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
    """Perform quantized matrix multiplication using CUDA custom call."""
    del block_m, block_n, block_k, use_bf16, num_warps, num_stages, split_k
    return _operate(x, w, scales, biases, transpose, group_size, bits, mode)
