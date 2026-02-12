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

"""Pallas TPU all-gather matmul interface with explicit backward kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import all_gather_matmul as _all_gather_matmul_impl


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active mapped context when available."""
    try:
        return jax.core.concrete_or_error(
            int,
            lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size using explicit value, axis context, then global device count."""
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def _largest_divisor_leq(x: int, candidates: tuple[int, ...] = (512, 256, 128, 64, 32, 16, 8, 4, 2, 1)) -> int:
    x = int(max(1, x))
    for candidate in candidates:
        if candidate <= x and x % candidate == 0:
            return candidate
    return 1


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _all_gather_matmul_core(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m n_local"]:
    del precision
    return _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
    )


def _all_gather_matmul_core_fwd(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
):
    del precision
    out = _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
    )
    return out, (x, y)


def _all_gather_matmul_core_bwd(
    axis_name: str,
    rhs_transpose: bool,
    bn: int | None,
    bk: int | None,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m n_local"],
):
    del bn, bk
    x, y = residual
    tp = _resolve_tp_size(tp_size, axis_name)

    if rhs_transpose:
        y_for_dx = y.T
    else:
        y_for_dx = y

    m_total = int(dy.shape[0])
    n_local = int(dy.shape[1])
    m_block = m_total // int(tp)
    m_half_block = max(1, m_block // 2)

    bm = _largest_divisor_leq(m_half_block)
    rs_bn = _largest_divisor_leq(int(y_for_dx.shape[0]))
    rs_bk = _largest_divisor_leq(n_local)

    from ..reduce_scatter_matmul._pallas_impl import reduce_scatter_matmul as _reduce_scatter_matmul_impl

    grad_x = _reduce_scatter_matmul_impl(
        dy,
        y_for_dx,
        axis_name=axis_name,
        tp_size=tp,
        collective_id=collective_id,
        bm=bm,
        bn=rs_bn,
        bk=rs_bk,
    )

    x_full = lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
    if rhs_transpose:
        grad_y = jnp.dot(dy.T, x_full, precision=precision)
    else:
        grad_y = jnp.dot(x_full.T, dy, precision=precision)

    return grad_x, grad_y


_all_gather_matmul_core.defvjp(_all_gather_matmul_core_fwd, _all_gather_matmul_core_bwd)


@kernel_registry.register("all_gather_matmul", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def all_gather_matmul(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool = False,
    bn: int | None = None,
    bk: int | None = None,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n_local"]:
    """Bidirectional ring all-gather matmul for TPU."""
    return _all_gather_matmul_core(x, y, axis_name, rhs_transpose, bn, bk, tp_size, collective_id, precision)


__all__ = ("all_gather_matmul",)
