# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Tests for advanced flash attention features (Triton backend)."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.flash_attention import flash_attention


class TestLogitsSoftCap:
    """Test logits soft cap feature."""

    def test_soft_cap_basic(self):
        """Test that soft cap is applied correctly."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        # Run with soft cap
        out_capped = flash_attention(q, k, v, logits_soft_cap=20.0)

        # Run without soft cap
        out_uncapped = flash_attention(q, k, v)

        # Outputs should be different
        assert out_capped.shape == out_uncapped.shape
        assert not jnp.allclose(out_capped, out_uncapped, rtol=1e-2)

    def test_soft_cap_gradient(self):
        """Test that gradients work with soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        def loss_fn(q, k, v):
            out = flash_attention(q, k, v, logits_soft_cap=20.0)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        # Check gradients are finite
        assert jnp.all(jnp.isfinite(grads[0]))
        assert jnp.all(jnp.isfinite(grads[1]))
        assert jnp.all(jnp.isfinite(grads[2]))

    def test_soft_cap_values(self):
        """Test different soft cap values."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.bfloat16)

        out_10 = flash_attention(q, k, v, logits_soft_cap=10.0)
        out_20 = flash_attention(q, k, v, logits_soft_cap=20.0)
        out_30 = flash_attention(q, k, v, logits_soft_cap=30.0)

        # Different soft cap values should produce different results
        assert not jnp.allclose(out_10, out_20, rtol=1e-2)
        assert not jnp.allclose(out_20, out_30, rtol=1e-2)


class TestSoftmaxAux:
    """Test softmax_aux (attention sinks) feature."""

    def test_attention_sinks_basic(self):
        """Test basic attention sinks functionality."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        # Create sink logits (per-head)
        sinks = jax.random.normal(key, (num_heads, num_sinks), dtype=jnp.float16) * 0.1

        out_with_sinks = flash_attention(q, k, v, softmax_aux=sinks)
        out_no_sinks = flash_attention(q, k, v)

        # Outputs should be different (sinks affect normalization)
        assert out_with_sinks.shape == out_no_sinks.shape
        assert not jnp.allclose(out_with_sinks, out_no_sinks, rtol=1e-2)

    def test_attention_sinks_broadcast(self):
        """Test that 1D sinks broadcast correctly."""
        batch, seq_len, num_heads, head_dim = 1, 16, 4, 32
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)

        # 1D sinks (broadcast to all heads)
        sinks_1d = jax.random.normal(key, (num_sinks,), dtype=jnp.float16)
        out_1d = flash_attention(q, k, v, softmax_aux=sinks_1d)

        # 2D sinks (same value for all heads)
        sinks_2d = jnp.broadcast_to(sinks_1d[None, :], (num_heads, num_sinks))
        out_2d = flash_attention(q, k, v, softmax_aux=sinks_2d)

        # Should be close (within float16 precision)
        assert jnp.allclose(out_1d, out_2d, rtol=1e-2)

    def test_attention_sinks_gradient(self):
        """Test gradients with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 16, 2, 32
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_heads, num_sinks), dtype=jnp.float16)

        def loss_fn(q, k, v):
            # Note: sinks are not differentiable in the current implementation
            out = flash_attention(q, k, v, softmax_aux=sinks)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        # Check all gradients are finite
        for grad in grads:
            assert jnp.all(jnp.isfinite(grad))


class TestCombinedFeatures:
    """Test combinations of advanced features."""

    def test_soft_cap_and_sinks(self):
        """Test using both soft cap and sinks together."""
        batch, seq_len, num_heads, head_dim = 2, 16, 4, 32
        num_sinks = 4
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_heads, num_sinks), dtype=jnp.float16)

        out = flash_attention(q, k, v, logits_soft_cap=20.0, softmax_aux=sinks)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.all(jnp.isfinite(out))

    def test_causal_with_features(self):
        """Test causal attention with advanced features."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(key, (batch, seq_len, num_heads, head_dim), dtype=jnp.float16)
        sinks = jax.random.normal(key, (num_heads, num_sinks), dtype=jnp.float16)

        out = flash_attention(q, k, v, causal=True, logits_soft_cap=20.0, softmax_aux=sinks)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert jnp.all(jnp.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
