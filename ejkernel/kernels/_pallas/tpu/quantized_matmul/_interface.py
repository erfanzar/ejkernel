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

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit
from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    to_backend_mode,
)

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.quantized_matmul import quantized_matmul as _xla_quantized_matmul
from ._pallas_impl_bwd import quantized_matmul_input_grad
from ._pallas_impl_core import (
    get_qmm_tpu_path,
    is_packed_tpu_legal_forward,
    is_packed_tpu_legal_input_grad,
)
from ._pallas_impl_fwd import _pallas_qmm_transpose_false


def _biases_to_zeros(scales: jax.Array, biases: jax.Array | None) -> jax.Array | None:
    """Convert internal affine additive biases back to canonical affine zeros."""
    if biases is None:
        return None
    safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
    return -biases / safe_scale


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
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
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
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> jax.Array:
    del gemv_mode, revsplit_k, revsplit_k_parts
    del use_bf16
    compute_in_bf16 = True
    zeros = _biases_to_zeros(scales, biases)

    if transpose:
        # Keep transpose=True on XLA baseline for now.
        return _xla_quantized_matmul(
            x,
            w,
            scales,
            zeros,
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
        zeros,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=compute_in_bf16,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=range(4, 15))
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
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
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
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
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
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
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
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
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
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    residual: tuple[jax.Array, jax.Array, jax.Array | None],
    grad_out: jax.Array,
) -> tuple[jax.Array, None, None, None]:
    del gemv_mode, revsplit_k, revsplit_k_parts
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
    zeros: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Quantized matmul on TPU via Pallas with custom backward support.

    ``zeros`` is used only for affine mode and is converted to per-group
    additive offsets before entering Pallas/XLA kernels.
    """
    del num_warps, num_stages, split_k
    del use_bf16

    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    _, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine quantized_matmul requires `zeros`.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        affine_biases = -zeros * safe_scale
    else:
        if zeros is not None:
            raise ValueError("zeros must be None for non-affine modes.")
        affine_biases = None

    backend_mode = to_backend_mode(mode, bits)

    return _operate(
        x,
        w,
        scales,
        affine_biases,
        transpose,
        group_size,
        bits,
        backend_mode,
        block_m,
        block_n,
        block_k,
        True,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
    )


__all__ = ("quantized_matmul",)
