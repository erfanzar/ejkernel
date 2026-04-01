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

"""XLA registry wrapper for Grouped Matrix Multiplication v3."""

from __future__ import annotations

from functools import partial

import jaxtyping
from beartype import beartype

from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

from ..._registry import Backend, Platform, kernel_registry
from ..grouped_matmul._xla_impl_fwd import Array, DTypeLike, Float, Int, jax, jnp
from ..grouped_matmul._xla_impl_fwd import grouped_matmul as _grouped_matmul_impl


def _apply_rhs_scale_bias(
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    *,
    transpose_rhs: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Apply v3's optional rhs scale/bias in a backend-agnostic way."""
    rhs_prepped = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    bias = None

    if rhs_scale is not None:
        if rhs_scale.ndim != 4 or rhs_scale.shape[2] != 1:
            raise ValueError("rhs_scale must have shape [num_groups, num_blocks, 1, n].")
        num_groups, size_k, size_n = rhs_prepped.shape
        if rhs_scale.shape[0] != num_groups or rhs_scale.shape[3] != size_n:
            raise ValueError("rhs_scale group/out dimensions must match rhs.")
        num_blocks = int(rhs_scale.shape[1])
        if size_k % num_blocks != 0:
            raise ValueError("rhs.shape[1] must be divisible by rhs_scale.shape[1].")
        block_size = size_k // num_blocks
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        rhs_prepped = rhs_prepped * scale.astype(rhs_prepped.dtype)

    if rhs_bias is not None:
        if rhs_bias.ndim != 3 or rhs_bias.shape[1] != 1:
            raise ValueError("rhs_bias must have shape [num_groups, 1, n].")
        if rhs_bias.shape[0] != rhs_prepped.shape[0] or rhs_bias.shape[2] != rhs_prepped.shape[2]:
            raise ValueError("rhs_bias group/out dimensions must match rhs.")
        bias = rhs_bias[:, 0, :]

    return rhs_prepped, bias


def _active_group_ids(
    group_sizes: jax.Array,
    num_groups: int,
    total_rows: int,
    group_offset: jax.Array | None,
) -> jax.Array:
    """Map each lhs row to its active group id."""
    offset = (
        group_offset.reshape(-1)[0].astype(group_sizes.dtype)
        if group_offset is not None
        else jnp.array(0, dtype=group_sizes.dtype)
    )
    active_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, offset, num_groups, axis=0)
    return jnp.repeat(
        jnp.arange(num_groups, dtype=group_sizes.dtype),
        active_sizes,
        total_repeat_length=total_rows,
    )


def grouped_matmulv3_autodiff_reference(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Pure-JAX reference used for stable autodiff."""
    del tiling, interpret
    rhs_prepped, bias = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    group_ids = _active_group_ids(group_sizes, rhs_prepped.shape[0], lhs.shape[0], group_offset)
    out = jax.vmap(
        lambda row, mat: jnp.matmul(
            row,
            mat,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
    )(lhs, rhs_prepped[group_ids])
    if bias is not None:
        out = out + bias[group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + jnp.asarray(existing_out, dtype=out.dtype)
    return out


def grouped_matmulv3_reference(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Reference v3 forward for both XLA execution and TPU backward helpers."""
    if rhs_scale is not None or rhs_bias is not None:
        return grouped_matmulv3_autodiff_reference(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
            group_offset,
            existing_out,
            rhs_scale,
            rhs_bias,
            transpose_rhs,
            interpret,
            precision,
        )
    rhs_prepped, bias = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    out = _grouped_matmul_impl(
        lhs,
        rhs_prepped,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out=None,
        transpose_rhs=False,
        interpret=interpret,
        precision=precision,
    )
    if bias is not None:
        group_ids = _active_group_ids(group_sizes, bias.shape[0], lhs.shape[0], group_offset)
        out = out + bias[group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + jnp.asarray(existing_out, dtype=out.dtype)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 9, 10, 11))
def _grouped_matmulv3_core(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    existing_out: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
) -> jax.Array:
    return grouped_matmulv3_reference(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )


def _grouped_matmulv3_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    existing_out: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
):
    out = grouped_matmulv3_reference(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    return out, (lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias)


def _grouped_matmulv3_bwd(
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    grad: jax.Array,
):
    lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias = residual

    _, pullback = jax.vjp(
        lambda lhs, rhs: grouped_matmulv3_autodiff_reference(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
            group_offset,
            existing_out,
            rhs_scale,
            rhs_bias,
            transpose_rhs,
            interpret,
            precision,
        ),
        lhs,
        rhs,
    )
    grad_lhs, grad_rhs = pullback(grad)

    grad_existing_out = grad if existing_out is not None else None
    grad_rhs_scale = None
    grad_rhs_bias = None

    if rhs_scale is not None:
        _, pullback = jax.vjp(
            lambda scale: grouped_matmulv3_autodiff_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                existing_out,
                scale,
                rhs_bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_scale,
        )
        (grad_rhs_scale,) = pullback(grad)

    if rhs_bias is not None:
        _, pullback = jax.vjp(
            lambda bias: grouped_matmulv3_autodiff_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                existing_out,
                rhs_scale,
                bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_bias,
        )
        (grad_rhs_bias,) = pullback(grad)

    return grad_lhs, grad_rhs, None, None, grad_existing_out, grad_rhs_scale, grad_rhs_bias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


@kernel_registry.register("grouped_matmulv3", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmulv3(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None,
    rhs_bias: Float[Array, "num_groups 1 n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n"]:
    """Grouped Matrix Multiplication v3 on XLA, reusing the reference fallback."""
    preferred_element_type = jnp.dtype(preferred_element_type) if preferred_element_type is not None else None
    return _grouped_matmulv3_core(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )


__all__ = ("grouped_matmulv3", "grouped_matmulv3_reference")
