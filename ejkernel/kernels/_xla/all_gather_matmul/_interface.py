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

"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import all_gather_matmul as _all_gather_matmul_impl


@kernel_registry.register("all_gather_matmul", Platform.XLA, Backend.ANY)
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
    """XLA all-gather matmul."""
    return _all_gather_matmul_impl(
        x,
        y,
        axis_name=axis_name,
        rhs_transpose=rhs_transpose,
        bn=bn,
        bk=bk,
        tp_size=tp_size,
        collective_id=collective_id,
        precision=precision,
    )


__all__ = ("all_gather_matmul",)
