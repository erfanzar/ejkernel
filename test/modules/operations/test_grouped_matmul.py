from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import grouped_matmul

from ._utils import assert_allclose


def _grouped_matmul_ref(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array, *, transpose_rhs: bool, existing_out: jax.Array | None):
    sizes = [int(x) for x in list(group_sizes)]
    offset = 0
    chunks = []
    for g, sz in enumerate(sizes):
        a = lhs[offset : offset + sz]
        b = rhs[g].T if transpose_rhs else rhs[g]
        chunks.append(a @ b)
        offset += sz
    out = jnp.concatenate(chunks, axis=0) if chunks else jnp.zeros((0, rhs.shape[2] if not transpose_rhs else rhs.shape[1]))
    if existing_out is not None:
        out = out + existing_out
    return out


def test_grouped_matmul_matches_reference_basic_and_transpose_rhs():
    M, K, N = 32, 16, 8
    group_sizes = jnp.array([16, 16], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(1), (2, K, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, platform="xla")
    ref = _grouped_matmul_ref(lhs, rhs, group_sizes, transpose_rhs=False, existing_out=None)
    assert out.shape == (M, N)
    assert_allclose(out, ref, atol=0.2)

    rhs_t = rhs.transpose(0, 2, 1)
    out_t = grouped_matmul(lhs, rhs_t, group_sizes, transpose_rhs=True, platform="xla")
    ref_t = _grouped_matmul_ref(lhs, rhs_t, group_sizes, transpose_rhs=True, existing_out=None)
    assert_allclose(out_t, ref_t, atol=0.2)


def test_grouped_matmul_variable_sizes_existing_out_and_v2():
    M, K, N = 40, 16, 8
    group_sizes = jnp.array([10, 6, 24], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(2), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(3), (3, K, N), dtype=jnp.float32)
    existing = jax.random.normal(jax.random.PRNGKey(4), (M, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, None, existing, do_padding=False, platform="xla")
    ref = _grouped_matmul_ref(lhs, rhs, group_sizes, transpose_rhs=False, existing_out=existing)
    assert_allclose(out, ref, atol=0.25)

    out_v2 = grouped_matmul(lhs, rhs, group_sizes, None, existing, do_padding=False, use_v2=True, platform="xla")
    assert_allclose(out_v2, ref, atol=0.25)


def test_grouped_matmul_interpret_mode_runs():
    M, K, N = 16, 8, 4
    group_sizes = jnp.array([8, 8], dtype=jnp.int32)
    lhs = jax.random.normal(jax.random.PRNGKey(5), (M, K), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.PRNGKey(6), (2, K, N), dtype=jnp.float32)

    out = grouped_matmul(lhs, rhs, group_sizes, interpret=True, platform="xla")
    assert out.shape == (M, N)
