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

"""TPU Pallas quantized matrix multiplication interface."""

from __future__ import annotations

import functools
from typing import Literal

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit
from ejkernel.quantization._utils.grouping import _require_bits

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.quantized_matmul import quantized_matmul as _xla_quantized_matmul
from ._pallas_impl_bwd import quantized_matmul_input_grad
from ._pallas_impl_core import (
    get_qmm_tpu_path,
    is_packed_tpu_legal_forward,
    is_packed_tpu_legal_input_grad,
)
from ._pallas_impl_fwd import _pallas_qmm_transpose_false

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    mode = mode.lower()
    if mode == "affine":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else _require_bits(bits, {2, 3, 4, 5, 6, 7, 8})
        if group_size not in {32, 64, 128}:
            raise ValueError("affine mode supports group_size in {32, 64, 128}.")
        return group_size, bits
    if mode == "mxfp4":
        group_size = 32 if group_size is None else int(group_size)
        bits = 4 if bits is None else int(bits)
        if group_size != 32 or bits != 4:
            raise ValueError("mxfp4 requires group_size=32 and bits=4.")
        return group_size, bits
    if mode == "mxfp8":
        group_size = 32 if group_size is None else int(group_size)
        bits = 8 if bits is None else int(bits)
        if group_size != 32 or bits != 8:
            raise ValueError("mxfp8 requires group_size=32 and bits=8.")
        return group_size, bits
    if mode == "nvfp4":
        group_size = 16 if group_size is None else int(group_size)
        bits = 4 if bits is None else int(bits)
        if group_size != 16 or bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4.")
        return group_size, bits
    if mode == "nvfp8":
        group_size = 16 if group_size is None else int(group_size)
        bits = 8 if bits is None else int(bits)
        if group_size != 16 or bits != 8:
            raise ValueError("nvfp8 requires group_size=16 and bits=8.")
        return group_size, bits
    if mode == "nf4":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else int(bits)
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        return group_size, bits
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _is_packed_tpu_legal(
    *,
    is_input_grad: bool,
    x_or_dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    """Strict legality gate for packed TPU Pallas QMM BlockSpecs."""
    if is_input_grad:
        return is_packed_tpu_legal_input_grad(
            x_or_dy,
            w_q,
            scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    return is_packed_tpu_legal_forward(
        x_or_dy,
        w_q,
        scales,
        group_size=group_size,
        bits=bits,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
    ],
)
def _operate_impl(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> jax.Array:
    del use_bf16
    compute_in_bf16 = True

    if transpose:
        # Keep transpose=True on XLA baseline for now.
        return _xla_quantized_matmul(
            x,
            w,
            scales,
            biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=compute_in_bf16,
        )

    path = get_qmm_tpu_path()
    packed_legal = _is_packed_tpu_legal(
        is_input_grad=False,
        x_or_dy=x,
        w_q=w,
        scales=scales,
        group_size=group_size,
        bits=bits,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )

    if bits in (4, 8):
        try:
            return _pallas_qmm_transpose_false(
                x,
                w,
                scales,
                biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                use_bf16=compute_in_bf16,
                path=path,
                packed_legal=packed_legal,
            )
        except Exception:
            pass

    return _xla_quantized_matmul(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=compute_in_bf16,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(4, 12))
def _operate(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> jax.Array:
    return _operate_impl(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
    )


def _operate_fwd(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array | None]]:
    out = _operate_impl(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
    )
    return out, (w, scales, biases)


def _operate_bwd(
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    residual: tuple[jax.Array, jax.Array, jax.Array | None],
    grad_out: jax.Array,
) -> tuple[jax.Array, None, None, None]:
    w, scales, biases = residual
    path = get_qmm_tpu_path()
    packed_legal = False
    if not transpose:
        packed_legal = _is_packed_tpu_legal(
            is_input_grad=True,
            x_or_dy=grad_out,
            w_q=w,
            scales=scales,
            group_size=group_size,
            bits=bits,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
        )
    grad_x = quantized_matmul_input_grad(
        grad_out,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        path=path,
        packed_legal=packed_legal,
    )
    return grad_x, None, None, None


_operate.defvjp(_operate_fwd, _operate_bwd)


@kernel_registry.register("quantized_matmul", Platform.PALLAS, Backend.TPU)
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
    """Quantized matmul on TPU via Pallas with custom backward support."""
    del num_warps, num_stages, split_k
    del use_bf16

    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)
    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires biases.")
    if mode != "affine" and biases is not None:
        raise ValueError("biases must be None for non-affine modes.")

    return _operate(
        x,
        w,
        scales,
        biases,
        transpose,
        group_size,
        bits,
        mode,
        block_m,
        block_n,
        block_k,
        True,
    )


__all__ = ("quantized_matmul",)
