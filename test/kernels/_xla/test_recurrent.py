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


"""Tests for XLA recurrent attention implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import lightning_attn, recurrent, recurrent_gla


class TestRecurrentAttention:
    """Test suite for recurrent attention kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output, final_state = recurrent(q, k, v)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        k = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = recurrent(q, k, v)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

    def test_orthogonal_vectors(self):
        """Test with orthogonal query/key vectors."""

        q = jnp.array([[[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0, 0.0]]]])
        k = q
        v = jnp.array([[[[2.0, 0.0, 0.0, 0.0]], [[0.0, 3.0, 0.0, 0.0]], [[0.0, 0.0, 4.0, 0.0]]]])

        output, _ = recurrent(q, k, v, softmax_scale=1.0)

        expected = jnp.array([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
        assert jnp.allclose(output[0, :, 0, :], expected, atol=1e-5)

    def test_with_initial_state(self):
        """Test with non-zero initial state."""
        batch, seq_len, num_heads, head_dim = 1, 3, 1, 4
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        initial_state = jnp.ones((batch, num_heads, head_dim, head_dim)) * 0.5

        output, final_state = recurrent(q, k, v, initial_state=initial_state)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

        assert not jnp.allclose(final_state, initial_state)

    def test_reverse_mode(self):
        """Test reverse processing."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output_fwd, _ = recurrent(q, k, v, reverse=False)
        output_rev, _ = recurrent(q, k, v, reverse=True)

        assert output_fwd.shape == output_rev.shape

        assert not jnp.allclose(output_fwd, output_rev)


class TestGLA:
    """Test suite for Gated Linear Attention."""

    def test_forward_shape(self):
        """Test GLA output shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))
        g = jnp.zeros((batch, seq_len, num_heads, head_dim))

        output, final_state = recurrent_gla(q, k, v, g=g)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_gradient_shapes(self):
        """Test GLA gradient shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))
        g = jnp.zeros((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = recurrent_gla(q, k, v, g=g)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape


class TestLightningAttention:
    """Test suite for Lightning Attention."""

    def test_forward_shape(self):
        """Test Lightning attention output shapes."""
        batch, seq_len, num_heads, head_dim = 2, 10, 4, 16
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output, final_state = lightning_attn(q, k, v, layer_idx=5, num_layers=24)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        assert final_state.shape == (batch, num_heads, head_dim, head_dim)

    def test_layer_dependent_decay(self):
        """Test that different layers produce different outputs."""
        batch, seq_len, num_heads, head_dim = 1, 5, 4, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output_layer0, _ = lightning_attn(q, k, v, layer_idx=0, num_layers=12)
        output_layer11, _ = lightning_attn(q, k, v, layer_idx=11, num_layers=12)

        assert not jnp.allclose(output_layer0, output_layer11)

    def test_gradient_shapes(self):
        """Test Lightning attention gradient shapes."""
        batch, seq_len, num_heads, head_dim = 1, 5, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = lightning_attn(q, k, v, layer_idx=3, num_layers=12)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
