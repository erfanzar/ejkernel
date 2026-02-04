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

"""Tests for XLA grouped matrix multiplication (ragged_dot)."""

import inspect

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels import Platform, kernel_registry
from ejkernel.kernels._xla.grouped_matmul import grouped_matmul


def _naive_grouped_matmul(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, transpose_rhs: bool) -> jax.Array:
    outs = []
    start = 0
    for group_idx in range(int(group_sizes.shape[0])):
        rows = int(group_sizes[group_idx])
        end = start + rows
        if rows == 0:
            continue
        if transpose_rhs:
            # rhs[group_idx] is (n, k)
            w = rhs[group_idx].T
        else:
            # rhs[group_idx] is (k, n)
            w = rhs[group_idx]
        outs.append(lhs[start:end] @ w)
        start = end
    return jnp.concatenate(outs, axis=0) if outs else jnp.zeros((0, rhs.shape[-2 if transpose_rhs else -1]), lhs.dtype)


@pytest.mark.parametrize("transpose_rhs", [False, True])
@pytest.mark.parametrize("use_jit", [False, True])
def test_matches_naive(transpose_rhs, use_jit):
    key = jax.random.PRNGKey(0)
    key, kl, kr = jax.random.split(key, 3)

    group_sizes = jnp.array([3, 5, 2], dtype=jnp.int32)
    m = int(group_sizes.sum())
    k = 7
    n = 4
    lhs = jax.random.normal(kl, (m, k), dtype=jnp.float32)

    if transpose_rhs:
        rhs = jax.random.normal(kr, (len(group_sizes), n, k), dtype=jnp.float32)
    else:
        rhs = jax.random.normal(kr, (len(group_sizes), k, n), dtype=jnp.float32)

    if use_jit:

        def _run(lhs, rhs, group_sizes):
            return grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)

        fn = jax.jit(_run)
        out = fn(lhs, rhs, group_sizes)
    else:
        out = grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, tiling=None)
    expected = _naive_grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs)

    assert out.shape == expected.shape
    assert jnp.allclose(out, expected, rtol=0, atol=0.125)


def test_registry_alias_for_grouped_matmulv2():
    impl = kernel_registry.get("grouped_matmulv2", platform=Platform.XLA)
    assert inspect.unwrap(impl) is inspect.unwrap(grouped_matmul)
