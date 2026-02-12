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

"""Backward utilities for XLA reduce-scatter matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float


def reduce_scatter_matmul_backward(
    dy: Float[Array, "m_local n"],
    x: Float[Array, "m k_shard"],
    y: Float[Array, "n k_shard"],
    *,
    axis_name: str,
    precision: jax.lax.PrecisionLike,
) -> tuple[Float[Array, "m k_shard"], Float[Array, "n k_shard"]]:
    """Compute gradients for reduce_scatter_matmul."""
    dy_full = lax.all_gather(dy, axis_name=axis_name, axis=0, tiled=True)
    grad_x = jnp.dot(dy_full, y, precision=precision)
    grad_y = jnp.dot(dy_full.T, x, precision=precision)
    return grad_x, grad_y


__all__ = ("reduce_scatter_matmul_backward",)
