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

"""XLA all-gather matmul implementation with explicit VJP."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

from ._xla_impl_bwd import all_gather_matmul_backward


def _validate_inputs(
    x: jax.Array,
    y: jax.Array,
    *,
    rhs_transpose: bool,
    tp_size: int | None,
) -> None:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"x and y must be rank-2 tensors, got {x.ndim=} and {y.ndim=}.")
    if x.dtype != y.dtype:
        raise ValueError(f"x and y must share dtype, got {x.dtype=} and {y.dtype=}.")

    k = x.shape[1]
    if rhs_transpose:
        k_from_y = y.shape[1]
    else:
        k_from_y = y.shape[0]

    if k != k_from_y:
        raise ValueError(
            "Incompatible matmul shapes for all_gather_matmul: "
            f"x.shape={x.shape}, y.shape={y.shape}, rhs_transpose={rhs_transpose}."
        )

    if tp_size is not None and tp_size < 1:
        raise ValueError(f"tp_size must be >= 1 when provided, got {tp_size}.")


def _forward_impl(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    *,
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "m n_local"], Float[Array, "m k"]]:
    x_full = lax.all_gather(x, axis_name=axis_name, axis=0, tiled=True)
    if rhs_transpose:
        out = jnp.dot(x_full, y.T, precision=precision)
    else:
        out = jnp.dot(x_full, y, precision=precision)
    return out, x_full


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4))
def _all_gather_matmul_core(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
) -> Float[Array, "m n_local"]:
    out, _ = _forward_impl(x, y, axis_name=axis_name, rhs_transpose=rhs_transpose, precision=precision)
    return out


def _all_gather_matmul_core_fwd(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
):
    out, x_full = _forward_impl(x, y, axis_name=axis_name, rhs_transpose=rhs_transpose, precision=precision)
    return out, (x, y, x_full)


def _all_gather_matmul_core_bwd(
    axis_name: str,
    rhs_transpose: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    dy: Float[Array, "m n_local"],
):
    _, y, x_full = residual
    grad_x, grad_y = all_gather_matmul_backward(
        dy,
        y,
        x_full,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        precision=precision,
    )
    return grad_x, grad_y


_all_gather_matmul_core.defvjp(_all_gather_matmul_core_fwd, _all_gather_matmul_core_bwd)


def all_gather_matmul(
    x: Float[Array, "m_local k"],
    y: Float[Array, "k n_local"] | Float[Array, "n_local k"],
    *,
    axis_name: str,
    rhs_transpose: bool = False,
    bn: int | None = None,
    bk: int | None = None,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n_local"]:
    """Compute all_gather(x, axis=0) @ y_shard with explicit backward kernels."""
    del bn, bk, collective_id
    _validate_inputs(x, y, rhs_transpose=rhs_transpose, tp_size=tp_size)
    return _all_gather_matmul_core(x, y, axis_name, rhs_transpose, precision)


__all__ = ("all_gather_matmul",)
