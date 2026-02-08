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

"""Backward CUDA implementation for quantized matrix multiplication."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.callib._ejit import ejit
from ejkernel.quantization import dequantize


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
    gemv_mode: str,
    revsplit_k: str,
    revsplit_k_parts: int | None,
):
    """Compute gradient with respect to the dense input x.

    Note:
        CUDA custom-call QMM currently supports only ``transpose=False`` in the
        forward path, so backward uses exact dequantize+dot for dX.
    """
    del gemv_mode, revsplit_k, revsplit_k_parts
    zeros = None
    if mode == "affine":
        if biases is None:
            raise ValueError("affine input grad requires affine metadata.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        zeros = -biases / safe_scale
    w_f = dequantize(w, scales, zeros, group_size=group_size, bits=bits, mode=mode)
    if transpose:
        # y = x @ w_f.T, so dX = dY @ w_f
        dims = (((1,), (0,)), ((), ()))
    else:
        # y = x @ w_f, so dX = dY @ w_f.T
        dims = (((1,), (1,)), ((), ()))
    return jax.lax.dot_general(dy, w_f, dims, preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
