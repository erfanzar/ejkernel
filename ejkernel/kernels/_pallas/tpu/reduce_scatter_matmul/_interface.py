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

"""Pallas TPU reduce-scatter matmul interface with explicit backward kernels."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl import reduce_scatter_matmul as _reduce_scatter_matmul_impl


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5, 6, 7, 8))
def _reduce_scatter_matmul_core(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m_local n"]:
    del precision
    return _reduce_scatter_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        tp_size=tp_size,
        collective_id=collective_id,
        bm=bm,
        bn=bn,
        bk=bk,
    )


def _reduce_scatter_matmul_core_fwd(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
):
    del precision
    out = _reduce_scatter_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        tp_size=tp_size,
        collective_id=collective_id,
        bm=bm,
        bn=bn,
        bk=bk,
    )
    return out, (x, y)


def _reduce_scatter_matmul_core_bwd(
    axis_name: str,
    bm: int,
    bn: int,
    bk: int,
    tp_size: int | None,
    collective_id: int | None,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m_local n"],
):
    del bm, bn, bk, tp_size, collective_id
    x, y = residual
    dy_full = lax.all_gather(dy, axis_name=axis_name, axis=0, tiled=True)
    grad_x = jnp.dot(dy_full, y, precision=precision)
    grad_y = jnp.dot(dy_full.T, x, precision=precision)
    return grad_x, grad_y


_reduce_scatter_matmul_core.defvjp(_reduce_scatter_matmul_core_fwd, _reduce_scatter_matmul_core_bwd)


@kernel_registry.register("reduce_scatter_matmul", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def reduce_scatter_matmul(
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    axis_name: str,
    bm: int = 128,
    bn: int = 128,
    bk: int = 128,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m_local n"]:
    """Bidirectional M-split reduce-scatter matmul for TPU."""
    return _reduce_scatter_matmul_core(x, y, axis_name, bm, bn, bk, tp_size, collective_id, precision)


__all__ = ("reduce_scatter_matmul",)
