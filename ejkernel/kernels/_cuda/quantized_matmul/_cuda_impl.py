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

"""Low-level CUDA implementation of quantized matrix multiplication.

This module contains the direct JAX FFI binding to the ``ejk_qmm_cuda``
custom call. It handles:

* One-time registration of the CUDA shared library via :func:`_register_cuda_qmm`.
* Mapping string quantization mode names to integer IDs consumed by the
  CUDA kernel.
* Input validation (shapes, bit-widths, and mode constraints).
* Construction and dispatch of the FFI call that executes the CUDA kernel.

The main entry point is :func:`quantized_matmul_cuda`, which is invoked by
the higher-level :func:`~ejkernel.kernels._cuda.quantized_matmul._interface.quantized_matmul`.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Literal

import jax
import jax.ffi as ffi
import jax.numpy as jnp

from ._build import build_cuda_lib

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
"""Type alias for supported quantization mode strings."""


@lru_cache(maxsize=1)
def _register_cuda_qmm() -> None:
    """Build and register the ``ejk_qmm_cuda`` FFI target with JAX.

    On the first call this function:

    1. Compiles the CUDA shared library (if not already cached) via
       :func:`~._build.build_cuda_lib`.
    2. Loads it with :mod:`ctypes` to obtain the ``ejk_qmm_cuda`` symbol.
    3. Registers that symbol as a JAX FFI target on the ``"gpu"`` platform.

    Subsequent calls are no-ops thanks to :func:`functools.lru_cache`.

    Raises:
        RuntimeError: If the CUDA library cannot be built or loaded.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    handler = lib.ejk_qmm_cuda
    ffi.register_ffi_target(
        "ejk_qmm_cuda",
        ffi.pycapsule(handler),
        platform="gpu",
        api_version=1,
    )


def _mode_to_id(mode: str) -> int:
    """Convert a quantization mode name to the integer ID expected by the CUDA kernel.

    Mapping:
        ========  ==
        Mode      ID
        ========  ==
        affine     0
        nf4        1
        mxfp4      2
        mxfp8      3
        nvfp4      4
        nvfp8      5
        w4a16      6
        w8a16      7
        ========  ==

    Args:
        mode: Case-insensitive quantization mode name.

    Returns:
        Integer identifier consumed by the CUDA kernel.

    Raises:
        ValueError: If *mode* is not one of the supported names.
    """
    mode = mode.lower()
    if mode == "affine":
        return 0
    if mode == "nf4":
        return 1
    if mode == "mxfp4":
        return 2
    if mode == "mxfp8":
        return 3
    if mode == "nvfp4":
        return 4
    if mode == "nvfp8":
        return 5
    if mode == "w4a16":
        return 6
    if mode == "w8a16":
        return 7
    raise ValueError(
        "CUDA quantized_matmul supports affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8, w4a16, w8a16 modes."
    )


def _expected_words(n: int, bits: int) -> int:
    """Compute the number of 32-bit words required to store *n* elements at *bits* each.

    This is the expected second dimension of the packed weight tensor
    ``w`` (shape ``(K, expected_words)``), where each ``uint32`` word
    packs ``floor(32 / bits)`` quantized values.

    Args:
        n: Number of output features (columns).
        bits: Bit-width per quantized element (2--8).

    Returns:
        Number of ``uint32`` words per row of the packed weight matrix.
    """
    return int((n * bits + 31) // 32)


def quantized_matmul_cuda(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
) -> jax.Array:
    """Execute quantized matrix multiplication on CUDA via JAX FFI.

    Performs ``x @ dequantize(w, scales, biases)`` entirely on the GPU
    using a custom CUDA kernel registered through JAX's FFI mechanism.
    Weights are stored in a packed ``uint32`` format and dequantized
    on-the-fly during the kernel execution.

    The function validates all inputs, infers default values for
    *bits* and *group_size* based on the selected *mode*, registers
    the CUDA kernel (on the first call), and dispatches the FFI call.

    Args:
        x: Input activation matrix of shape ``(M, K)`` with a float dtype.
        w: Packed quantized weight matrix of shape ``(K, ceil(N * bits / 32))``
            with ``uint32`` dtype, where *N* is the number of output
            features.
        scales: Per-group scale factors of shape ``(K, N / group_size)``.
        biases: Per-group bias values with the same shape as *scales*.
            Required for ``"affine"`` mode; optional for other modes.
            Defaults to ``None``.
        transpose: Whether the weight matrix is stored in ``(N, K)``
            layout. Currently **must** be ``False``; passing ``True``
            raises :class:`ValueError`.
        group_size: Number of output features per quantization group.
            Defaults depend on *mode*: 32 for ``mxfp4``/``mxfp8``,
            16 for ``nvfp4``/``nvfp8``, and 64 for ``affine``/``nf4``.
        bits: Bit-width per quantized element. Defaults to 8 for
            ``mxfp8``/``nvfp8`` and 4 for all other modes. Affine mode
            accepts values in {2, 3, 4, 5, 6, 7, 8}.
        mode: Quantization scheme. One of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``,
            ``"w4a16"``, or ``"w8a16"``.

    Returns:
        Result matrix of shape ``(M, N)`` with the same dtype as ``x``.

    Raises:
        ValueError: If *bits* is outside the range [2, 8] or incompatible
            with the chosen *mode*.
        ValueError: If *transpose* is ``True`` (currently unsupported).
        ValueError: If *scales* is not rank-2 or its shape is inconsistent
            with *x* and *w*.
        ValueError: If the packed weight width does not match the expected
            number of ``uint32`` words for the given *N* and *bits*.
        ValueError: If the K dimensions of *x* and *w* disagree.
    """
    # Avoid device queries on tracers; runtime will error if no GPU backend.

    mode_id = _mode_to_id(mode)
    mode = mode.lower()
    if bits is None:
        if mode in ("mxfp8", "nvfp8", "w8a16"):
            bits = 8
        else:
            bits = 4
    bits = int(bits)
    if group_size is None:
        if mode in ("mxfp4", "mxfp8"):
            group_size = 32
        elif mode in ("nvfp4", "nvfp8"):
            group_size = 16
        elif mode in ("w4a16", "w8a16"):
            group_size = int(x.shape[1])
        else:
            group_size = 64
    group_size = int(group_size)

    if bits < 2 or bits > 8:
        raise ValueError("CUDA quantized_matmul supports bits in [2, 8].")
    if mode == "affine" and bits not in (2, 3, 4, 5, 6, 7, 8):
        raise ValueError("CUDA quantized_matmul affine supports bits in {2, 3, 4, 5, 6, 7, 8}.")
    if mode == "nf4" and bits != 4:
        raise ValueError("CUDA quantized_matmul nf4 requires bits=4.")
    if mode == "mxfp4" and bits != 4:
        raise ValueError("CUDA quantized_matmul mxfp4 requires bits=4.")
    if mode == "mxfp8" and bits != 8:
        raise ValueError("CUDA quantized_matmul mxfp8 requires bits=8.")
    if mode == "nvfp4" and bits != 4:
        raise ValueError("CUDA quantized_matmul nvfp4 requires bits=4.")
    if mode == "nvfp8" and bits != 8:
        raise ValueError("CUDA quantized_matmul nvfp8 requires bits=8.")
    if mode == "w4a16" and bits != 4:
        raise ValueError("CUDA quantized_matmul w4a16 requires bits=4.")
    if mode == "w8a16" and bits != 8:
        raise ValueError("CUDA quantized_matmul w8a16 requires bits=8.")
    if mode in ("w4a16", "w8a16"):
        if not transpose:
            raise ValueError("CUDA quantized_matmul w4a16/w8a16 requires transpose=True.")
        if group_size != int(x.shape[1]):
            raise ValueError("CUDA quantized_matmul w4a16/w8a16 requires group_size=K.")
    if transpose and mode not in ("w4a16", "w8a16"):
        raise ValueError("CUDA quantized_matmul transpose=True only supports w4a16/w8a16.")

    m, k = int(x.shape[0]), int(x.shape[1])
    if scales.ndim != 2:
        raise ValueError("CUDA quantized_matmul scales must be rank-2.")
    if transpose:
        if scales.shape[0] != int(w.shape[0]):
            raise ValueError("CUDA quantized_matmul scales shape must be (N, K/group_size) for transpose=True.")
        if k % group_size != 0:
            raise ValueError("CUDA quantized_matmul K must be divisible by group_size for transpose=True.")
        n_groups = int(scales.shape[1])
        if n_groups != k // group_size:
            raise ValueError("CUDA quantized_matmul scales second dimension must be K/group_size.")
        n = int(w.shape[0])
        expected_words = _expected_words(k, bits)
        if int(w.shape[1]) != expected_words:
            raise ValueError("CUDA quantized_matmul packed weight shape does not match K and bits.")
    else:
        n_groups = int(scales.shape[1])
        n = n_groups * group_size
        if scales.shape[0] != k:
            raise ValueError("CUDA quantized_matmul scales shape must be (K, N/group_size).")
        expected_words = _expected_words(n, bits)
        k_w = int(w.shape[0])
        if int(w.shape[1]) != expected_words:
            raise ValueError("CUDA quantized_matmul packed weight shape does not match N and bits.")
        if k != k_w:
            raise ValueError("Input K dimension does not match packed weight K.")

    out_dtype = jnp.dtype(x.dtype)
    if out_dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise ValueError("CUDA quantized_matmul supports float16/float32/bfloat16 outputs.")
    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)
    _register_cuda_qmm()

    call = ffi.ffi_call("ejk_qmm_cuda", out_shape, vmap_method="sequential")

    attrs = {
        "group_size": group_size,
        "bits": bits,
        "mode": mode_id,
        "transpose": int(transpose),
    }

    if biases is None:
        return call(x, w, scales, **attrs)
    return call(x, w, scales, biases, **attrs)
