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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

from ejkernel.kernels._triton import rwkv4 as triton_rwkv4
from ejkernel.kernels._triton import rwkv6 as triton_rwkv6
from ejkernel.kernels._triton import rwkv7 as triton_rwkv7
from ejkernel.kernels._triton import rwkv7_mul as triton_rwkv7_mul
from ejkernel.kernels._xla.rwkv4 import rwkv4 as xla_rwkv4
from ejkernel.kernels._xla.rwkv6 import rwkv6 as xla_rwkv6
from ejkernel.kernels._xla.rwkv7 import rwkv7 as xla_rwkv7
from ejkernel.kernels._xla.rwkv7 import rwkv7_mul as xla_rwkv7_mul

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def test_rwkv4_matches_xla_and_grad():
    B, T, C = 2, 16, 64
    key = jax.random.PRNGKey(0)
    w = jax.random.normal(key, (C,), dtype=jnp.float16)
    u = jax.random.normal(jax.random.PRNGKey(1), (C,), dtype=jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(2), (B, T, C), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(3), (B, T, C), dtype=jnp.float16)

    out_tri, st_tri = triton_rwkv4(w, u, k, v, state=None)
    out_xla, st_xla = xla_rwkv4(w, u, k, v, state=None)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)

    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    def loss_tri(w_):
        o, st = triton_rwkv4(w_, u, k, v, state=None)
        return jnp.sum(o) + jnp.sum(st)

    def loss_xla(w_):
        o, st = xla_rwkv4(w_, u, k, v, state=None)
        return jnp.sum(o) + jnp.sum(st)

    dw_tri = jax.grad(loss_tri)(w)
    dw_xla = jax.grad(loss_xla)(w)
    np.testing.assert_allclose(np.asarray(dw_tri, np.float32), np.asarray(dw_xla, np.float32), rtol=2e-2, atol=2e-2)


def test_rwkv6_matches_xla_and_varlen():
    B, T, H, K, V = 2, 16, 2, 32, 32
    key = jax.random.PRNGKey(0)
    r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float16)
    k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, V), dtype=jnp.float16)
    w = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, K), dtype=jnp.float16) * -0.01
    u = jax.random.normal(jax.random.PRNGKey(4), (H, K), dtype=jnp.float16)
    h0 = jax.random.normal(jax.random.PRNGKey(5), (B, H, K, V), dtype=jnp.float32)

    out_tri, st_tri = triton_rwkv6(r, k, v, w, u, initial_state=h0)
    out_xla, st_xla = xla_rwkv6(r, k, v, w, u, initial_state=h0)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)

    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    r_p, k_p, v_p, w_p = map(lambda x: rearrange(x, "b t h d -> 1 (b t) h d"), (r, k, v, w))
    cu = jnp.arange(0, (B + 1) * T, T, dtype=jnp.int32)
    out_var, st_var = triton_rwkv6(r_p, k_p, v_p, w_p, u, initial_state=h0, cu_seqlens=cu)
    out_var, st_var = jax.block_until_ready(out_var), jax.block_until_ready(st_var)
    np.testing.assert_allclose(
        np.asarray(out_var.reshape(B, T, H, V), np.float32),
        np.asarray(out_tri, np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(np.asarray(st_var, np.float32), np.asarray(st_tri, np.float32), rtol=2e-2, atol=2e-2)


def test_rwkv7_matches_xla_and_mul_wrapper():
    B, T, H, K, V = 2, 16, 2, 32, 32
    key = jax.random.PRNGKey(0)
    r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float16)
    w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float16) * -0.01
    k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float16)
    v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float16)
    a = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float16) * 0.01
    b = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float16) * 0.01
    kk = jax.random.normal(jax.random.PRNGKey(6), (B, T, H, K), dtype=jnp.float16) * 0.01
    h0 = jax.random.normal(jax.random.PRNGKey(7), (B, H, K, V), dtype=jnp.float32)

    out_tri, st_tri = triton_rwkv7(r, w, k, v, a, b, initial_state=h0)
    out_xla, st_xla = xla_rwkv7(r, w, k, v, a, b, initial_state=h0)

    out_tri, st_tri = jax.block_until_ready(out_tri), jax.block_until_ready(st_tri)
    out_xla, st_xla = jax.block_until_ready(out_xla), jax.block_until_ready(st_xla)
    np.testing.assert_allclose(np.asarray(out_tri, np.float32), np.asarray(out_xla, np.float32), rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.asarray(st_tri, np.float32), np.asarray(st_xla, np.float32), rtol=2e-2, atol=2e-2)

    out_mul_tri, st_mul_tri = triton_rwkv7_mul(r, w, k, v, kk, a, initial_state=h0)
    out_mul_xla, st_mul_xla = xla_rwkv7_mul(r, w, k, v, kk, a, initial_state=h0)
    out_mul_tri, st_mul_tri = jax.block_until_ready(out_mul_tri), jax.block_until_ready(st_mul_tri)
    out_mul_xla, st_mul_xla = jax.block_until_ready(out_mul_xla), jax.block_until_ready(st_mul_xla)
    np.testing.assert_allclose(
        np.asarray(out_mul_tri, np.float32), np.asarray(out_mul_xla, np.float32), rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        np.asarray(st_mul_tri, np.float32), np.asarray(st_mul_xla, np.float32), rtol=2e-2, atol=2e-2
    )
