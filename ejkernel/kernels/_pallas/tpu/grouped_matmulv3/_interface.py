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

"""Grouped Matrix Multiplication v3 for TPU using upstream-style Pallas kernels."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.grouped_matmulv3._interface import _apply_rhs_scale_bias, grouped_matmulv3_reference
from ..grouped_matmul._pallas_impl import LutFn
from ..grouped_matmulv2._pallas_impl import grouped_matmul as back_grouped_matmul
from ..grouped_matmulv2._pallas_impl import transposed_grouped_matmul as back_tgrouped_matmul
from ._pallas_impl import TileSizes, calculate_tiling, grouped_matmulv3_pallas_impl


def _normalize_tiling(
    tiling: tuple[int, int, int] | LutFn | None,
    lhs: jax.Array,
    rhs: jax.Array,
) -> TileSizes | Callable:
    """Normalize repo-style tiling into the upstream tile-info format."""
    if tiling is None:
        return calculate_tiling
    if isinstance(tiling, tuple):
        return TileSizes(tile_m=int(tiling[0]), tile_k=int(tiling[1]), tile_n=int(tiling[2]))

    def _wrapped_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act):
        tile = tiling(dims.size_m, dims.size_k, dims.size_n)
        if tile is None:
            return calculate_tiling(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        return TileSizes(tile_m=int(tile[0]), tile_k=int(tile[1]), tile_n=int(tile[2]))

    return _wrapped_tile_fn


def _call_grouped_matmulv3(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
) -> tuple[jax.Array, jax.Array]:
    """Run the v3 kernel and return the prepared rhs used for backward."""
    del precision
    rhs_prepped = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    out = grouped_matmulv3_pallas_impl(
        lhs=lhs,
        rhs=rhs_prepped,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        tile_info=_normalize_tiling(tiling, lhs, rhs_prepped),
        preferred_element_type=preferred_element_type,
        interpret=interpret,
    )
    return out, rhs_prepped


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
    out, _ = _call_grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    return out if existing_out is None else out + existing_out


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
    out, rhs_prepped = _call_grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    if existing_out is not None:
        out = out + existing_out
    return out, (
        lhs,
        rhs,
        rhs_prepped,
        group_sizes,
        group_offset,
        rhs_scale,
        rhs_bias,
        rhs.shape[0],
        existing_out is not None,
    )


def _grouped_matmulv3_bwd(
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    grad: jax.Array,
):
    (
        lhs,
        rhs,
        rhs_prepped,
        group_sizes,
        group_offset,
        rhs_scale,
        rhs_bias,
        num_actual_groups,
        has_existing_out,
    ) = residual

    rhs_effective, _ = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        None,
        transpose_rhs=transpose_rhs,
    )
    resolved_tiling = tiling if isinstance(tiling, tuple) else (128, 128, 128)
    grad_lhs = back_grouped_matmul(
        grad,
        rhs_effective,
        group_sizes,
        lhs[0].dtype,
        resolved_tiling,
        input_buffer_count=2,
        group_offset=group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    grad_rhs = back_tgrouped_matmul(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        rhs_prepped.dtype,
        resolved_tiling,
        group_offset=group_offset,
        num_actual_groups=num_actual_groups,
        interpret=interpret,
    )
    if rhs_scale is not None:
        block_size = rhs_prepped.shape[1] // rhs_scale.shape[1]
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        grad_rhs = grad_rhs * scale.astype(grad_rhs.dtype)
    grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs

    grad_rhs_scale = None
    grad_rhs_bias = None
    if rhs_scale is not None:
        _, pullback = jax.vjp(
            lambda scale: grouped_matmulv3_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                None,
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
            lambda bias: grouped_matmulv3_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                None,
                rhs_scale,
                bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_bias,
        )
        (grad_rhs_bias,) = pullback(grad)

    grad_existing_out = grad if has_existing_out else None
    return grad_lhs, grad_rhs, None, None, grad_existing_out, grad_rhs_scale, grad_rhs_bias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


@kernel_registry.register("grouped_matmulv3", Platform.PALLAS, Backend.TPU)
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
    """Grouped Matrix Multiplication v3 on TPU using the vendored upstream kernel."""
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


__all__ = ("grouped_matmulv3",)
