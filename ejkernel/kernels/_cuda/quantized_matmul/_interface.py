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

"""CUDA quantized matrix multiplication interface.

This module provides the public, registry-registered entry point for
CUDA-backed quantized matrix multiplication. The function
:func:`quantized_matmul` is decorated with the ejKernel kernel registry
(platform ``CUDA``, backend ``GPU``) and with ``jaxtyping``/``beartype``
runtime type checking.

It supports the following quantization modes via the underlying CUDA kernel:

* **affine** -- Uniform affine quantization (2--8 bit, per-group scales
  and biases).
* **nf4** -- 4-bit NormalFloat quantization.
* **mxfp4** / **mxfp8** -- Microscaling FP4/FP8 formats.
* **nvfp4** / **nvfp8** -- NVIDIA FP4/FP8 formats.

For modes not natively supported by the CUDA kernel, the function
transparently falls back to the XLA-based implementation.

Weights are expected as packed ``uint32`` arrays; the output dtype matches
the input ``x`` dtype.
"""

from __future__ import annotations

from typing import Literal

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._cuda_impl import quantized_matmul_cuda

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
"""Type alias for the supported quantization mode strings."""


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
    """Perform quantized matrix multiplication using a CUDA custom call.

    Computes ``x @ dequantize(w, scales, biases)`` on the GPU. For modes
    not natively supported by the CUDA kernel (i.e., any value of *mode*
    outside the six CUDA-supported names), the call is transparently
    forwarded to the XLA-based fallback implementation.

    The function is registered in the ejKernel kernel registry under the
    name ``"quantized_matmul"`` for ``Platform.CUDA`` / ``Backend.GPU``
    and benefits from ``ejit`` persistent JIT caching.

    Note:
        The keyword-only tuning parameters (``block_m``, ``block_n``, etc.)
        are accepted for API compatibility with the Triton backend but are
        silently ignored by the CUDA implementation.

    Args:
        x: Input activation matrix of shape ``(M, K)`` with a float dtype.
        w: Packed quantized weight matrix of shape
            ``(K, ceil(N * bits / 32))`` stored as ``uint32``.
        scales: Per-group scale factors of shape ``(K, N / group_size)``.
        biases: Per-group bias values with the same shape as *scales*.
            Required for ``"affine"`` mode. Defaults to ``None``.
        transpose: Whether the weight matrix uses ``(N, K)`` layout.
            The CUDA backend currently requires ``transpose=False``.
        group_size: Number of output features per quantization group.
            Inferred from *mode* when ``None``: 32 for ``mxfp4``/``mxfp8``,
            16 for ``nvfp4``/``nvfp8``, 64 otherwise.
        bits: Bit-width per quantized element. Inferred from *mode* when
            ``None``: 8 for 8-bit modes, 4 for all others. Affine mode
            accepts 2--8.
        mode: Quantization scheme. One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``. Other
            values trigger an XLA fallback.
        block_m: Ignored (Triton API compatibility). Defaults to 128.
        block_n: Ignored (Triton API compatibility). Defaults to 128.
        block_k: Ignored (Triton API compatibility). Defaults to 64.
        use_bf16: Ignored (Triton API compatibility). Defaults to ``True``.
        num_warps: Ignored (Triton API compatibility). Defaults to ``None``.
        num_stages: Ignored (Triton API compatibility). Defaults to ``None``.
        split_k: Ignored (Triton API compatibility). Defaults to ``None``.

    Returns:
        Result matrix of shape ``(M, N)`` with the same dtype as ``x``.

    Raises:
        ValueError: Propagated from the CUDA implementation when inputs
            are invalid (e.g., unsupported *bits*, shape mismatches, or
            unsupported ``transpose``).
    """
    del block_m, block_n, block_k, use_bf16, num_warps, num_stages, split_k

    return ejit(func=quantized_matmul_cuda, static_argnames=["transpose", "group_size", "bits", "mode"])(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
