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

from ejkernel.kernels._xla.rwkv4 import rwkv4
from ejkernel.kernels._xla.rwkv6 import rwkv6
from ejkernel.kernels._xla.rwkv7 import rwkv7, rwkv7_mul


class TestRWKV4:
    def test_forward_shapes(self):
        B, T, C = 2, 8, 32
        w = jnp.zeros((C,), dtype=jnp.float32)
        u = jnp.zeros((C,), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(0), (B, T, C), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(1), (B, T, C), dtype=jnp.float32)
        out, state = rwkv4(w, u, k, v, state=None)
        assert out.shape == (B, T, C)
        assert state.shape == (B, 3, C)

    def test_gradient_shapes(self):
        B, T, C = 1, 4, 16
        key = jax.random.PRNGKey(0)
        w = jax.random.normal(key, (C,), dtype=jnp.float32)
        u = jax.random.normal(jax.random.PRNGKey(1), (C,), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(2), (B, T, C), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (B, T, C), dtype=jnp.float32)
        state = jnp.stack(
            [
                jnp.zeros((B, C), dtype=jnp.float32),
                jnp.zeros((B, C), dtype=jnp.float32),
                jnp.full((B, C), -1e30, dtype=jnp.float32),
            ],
            axis=1,
        )

        def loss_fn(w_, u_, k_, v_, s_):
            o, st = rwkv4(w_, u_, k_, v_, s_)
            return jnp.sum(o) + jnp.sum(st)

        dw, du, dk, dv, ds = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(w, u, k, v, state)
        assert dw.shape == w.shape
        assert du.shape == u.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert ds.shape == state.shape


class TestRWKV6:
    def test_forward_shapes(self):
        B, T, H, K, V = 2, 8, 2, 16, 16
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, V), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, K), dtype=jnp.float32) * -0.01
        u = jax.random.normal(jax.random.PRNGKey(4), (H, K), dtype=jnp.float32)
        o, st = rwkv6(r, k, v, w, u)
        assert o.shape == (B, T, H, V)
        assert st.shape == (B, H, K, V)

    def test_varlen_matches_batched(self):
        B, T, H, K, V = 3, 5, 2, 8, 8
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, V), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, K), dtype=jnp.float32) * -0.01
        u = jax.random.normal(jax.random.PRNGKey(4), (H, K), dtype=jnp.float32)
        h0 = jax.random.normal(jax.random.PRNGKey(5), (B, H, K, V), dtype=jnp.float32)

        o_b, st_b = rwkv6(r, k, v, w, u, initial_state=h0)

        r_p, k_p, v_p, w_p = map(lambda x: rearrange(x, "b t h d -> 1 (b t) h d"), (r, k, v, w))
        cu = jnp.arange(0, (B + 1) * T, T, dtype=jnp.int32)
        o_p, st_p = rwkv6(r_p, k_p, v_p, w_p, u, initial_state=h0, cu_seqlens=cu)

        np.testing.assert_allclose(o_b, o_p.reshape(B, T, H, V), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(st_b, st_p, rtol=1e-4, atol=1e-4)

    def test_gradient_shapes(self):
        B, T, H, K, V = 1, 4, 2, 8, 8
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        k = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, V), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, K), dtype=jnp.float32) * -0.01
        u = jax.random.normal(jax.random.PRNGKey(4), (H, K), dtype=jnp.float32)
        h0 = jax.random.normal(jax.random.PRNGKey(5), (B, H, K, V), dtype=jnp.float32)

        def loss_fn(r_, k_, v_, w_, u_, h0_):
            o, st = rwkv6(r_, k_, v_, w_, u_, initial_state=h0_)
            return jnp.sum(o) + jnp.sum(st)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(r, k, v, w, u, h0)
        assert grads[0].shape == r.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape
        assert grads[3].shape == w.shape
        assert grads[4].shape == u.shape
        assert grads[5].shape == h0.shape


class TestRWKV7:
    def test_forward_shapes(self):
        B, T, H, K, V = 2, 8, 2, 16, 16
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32) * -0.01
        k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float32)
        a = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float32) * 0.01
        b = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float32) * 0.01
        o, st = rwkv7(r, w, k, v, a, b)
        assert o.shape == (B, T, H, V)
        assert st.shape == (B, H, K, V)

    def test_mul_wrapper_matches_ab(self):
        B, T, H, K, V = 2, 6, 2, 8, 8
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32) * -0.01
        k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float32)
        kk = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float32) * 0.01
        a = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float32) * 0.01

        o_mul, st_mul = rwkv7_mul(r, w, k, v, kk, a)
        o_ab, st_ab = rwkv7(r, w, k, v, kk * a, -kk)

        np.testing.assert_allclose(o_mul, o_ab, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(st_mul, st_ab, rtol=1e-5, atol=1e-5)

    def test_varlen_matches_batched(self):
        B, T, H, K, V = 3, 5, 2, 8, 8
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32) * -0.01
        k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float32)
        a = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float32) * 0.01
        b = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float32) * 0.01
        h0 = jax.random.normal(jax.random.PRNGKey(6), (B, H, K, V), dtype=jnp.float32)

        o_b, st_b = rwkv7(r, w, k, v, a, b, initial_state=h0)
        r_p, w_p, k_p, v_p, a_p, b_p = map(
            lambda x: rearrange(x, "b t h d -> 1 (b t) h d"),
            (r, w, k, v, a, b),
        )
        cu = jnp.arange(0, (B + 1) * T, T, dtype=jnp.int32)
        o_p, st_p = rwkv7(r_p, w_p, k_p, v_p, a_p, b_p, initial_state=h0, cu_seqlens=cu)

        np.testing.assert_allclose(o_b, o_p.reshape(B, T, H, V), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(st_b, st_p, rtol=1e-4, atol=1e-4)

    def test_gradient_shapes(self):
        B, T, H, K, V = 1, 4, 2, 8, 8
        key = jax.random.PRNGKey(0)
        r = jax.random.normal(key, (B, T, H, K), dtype=jnp.float32)
        w = jax.random.normal(jax.random.PRNGKey(1), (B, T, H, K), dtype=jnp.float32) * -0.01
        k = jax.random.normal(jax.random.PRNGKey(2), (B, T, H, K), dtype=jnp.float32)
        v = jax.random.normal(jax.random.PRNGKey(3), (B, T, H, V), dtype=jnp.float32)
        a = jax.random.normal(jax.random.PRNGKey(4), (B, T, H, K), dtype=jnp.float32) * 0.01
        b = jax.random.normal(jax.random.PRNGKey(5), (B, T, H, K), dtype=jnp.float32) * 0.01
        h0 = jax.random.normal(jax.random.PRNGKey(6), (B, H, K, V), dtype=jnp.float32)

        def loss_fn(r_, w_, k_, v_, a_, b_, h0_):
            o, st = rwkv7(r_, w_, k_, v_, a_, b_, initial_state=h0_)
            return jnp.sum(o) + jnp.sum(st)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))(r, w, k, v, a, b, h0)
        assert grads[0].shape == r.shape
        assert grads[1].shape == w.shape
        assert grads[2].shape == k.shape
        assert grads[3].shape == v.shape
        assert grads[4].shape == a.shape
        assert grads[5].shape == b.shape
        assert grads[6].shape == h0.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
