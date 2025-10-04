"""Internal consistency tests for XLA implementations.

These tests verify that XLA implementations are numerically stable,
deterministic, and handle various input conditions correctly.
"""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import apply_native_sparse_attention, mean_pooling, recurrent


class TestNumericalStability:
    """Test numerical stability of XLA implementations."""

    def test_recurrent_no_nan_inf(self):
        """Test recurrent doesn't produce NaN/Inf with random inputs."""
        key = jax.random.PRNGKey(42)
        batch, seq_len, num_heads, head_dim = 2, 16, 4, 32

        key, *subkeys = jax.random.split(key, 4)
        q = jax.random.normal(subkeys[0], (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(subkeys[1], (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(subkeys[2], (batch, seq_len, num_heads, head_dim))

        output, _ = recurrent(q, k, v)
        assert jnp.all(jnp.isfinite(output)), "Recurrent produced NaN or Inf"

    def test_mean_pooling_no_nan_inf(self):
        """Test mean pooling doesn't produce NaN/Inf."""
        key = jax.random.PRNGKey(42)
        batch, seq_len, hidden_dim = 4, 20, 64

        x = jax.random.normal(key, (batch, seq_len, hidden_dim))
        output = mean_pooling(x)
        assert jnp.all(jnp.isfinite(output)), "Mean pooling produced NaN or Inf"

    def test_sparse_attention_no_nan_inf(self):
        """Test sparse attention doesn't produce NaN/Inf."""
        key = jax.random.PRNGKey(42)
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32
        num_blocks = seq_len // block_size

        key, *subkeys = jax.random.split(key, 4)
        q = jax.random.normal(subkeys[0], (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(subkeys[1], (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(subkeys[2], (batch, seq_len, num_heads, head_dim))

        block_indices = jnp.tile(jnp.arange(1)[None, None, None, :], (batch, num_heads, num_blocks, 1))
        output = apply_native_sparse_attention(q, k, v, block_indices, 1, block_size)

        assert jnp.all(jnp.isfinite(output)), "Sparse attention produced NaN or Inf"


class TestDeterminism:
    """Test that implementations are deterministic."""

    def test_recurrent_deterministic(self):
        """Test recurrent produces same output on multiple runs."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 8
        q = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        k = jnp.ones((batch, seq_len, num_heads, head_dim)) * 0.5
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        output1, _ = recurrent(q, k, v)
        output2, _ = recurrent(q, k, v)
        output3, _ = recurrent(q, k, v)

        assert jnp.allclose(output1, output2, rtol=0, atol=0)
        assert jnp.allclose(output2, output3, rtol=0, atol=0)

    def test_mean_pooling_deterministic(self):
        """Test mean pooling produces same output on multiple runs."""
        batch, seq_len, hidden_dim = 2, 10, 16
        x = jnp.ones((batch, seq_len, hidden_dim))

        output1 = mean_pooling(x)
        output2 = mean_pooling(x)
        output3 = mean_pooling(x)

        assert jnp.allclose(output1, output2, rtol=0, atol=0)
        assert jnp.allclose(output2, output3, rtol=0, atol=0)


class TestGradientFiniteness:
    """Test that gradients are finite."""

    def test_recurrent_gradients_finite(self):
        """Test recurrent gradients are finite."""
        key = jax.random.PRNGKey(42)
        batch, seq_len, num_heads, head_dim = 1, 8, 2, 16

        key, *subkeys = jax.random.split(key, 4)
        q = jax.random.normal(subkeys[0], (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(subkeys[1], (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(subkeys[2], (batch, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            o, _ = recurrent(q, k, v)
            return jnp.sum(o)

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.all(jnp.isfinite(dq)), "dq contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dk)), "dk contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dv)), "dv contains NaN or Inf"

    def test_sparse_attention_gradients_finite(self):
        """Test sparse attention gradients are finite."""
        key = jax.random.PRNGKey(42)
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32
        num_blocks = seq_len // block_size

        key, *subkeys = jax.random.split(key, 4)
        q = jax.random.normal(subkeys[0], (batch, seq_len, num_heads, head_dim))
        k = jax.random.normal(subkeys[1], (batch, seq_len, num_heads, head_dim))
        v = jax.random.normal(subkeys[2], (batch, seq_len, num_heads, head_dim))

        block_indices = jnp.tile(jnp.arange(1)[None, None, None, :], (batch, num_heads, num_blocks, 1))

        def loss_fn(q, k, v):
            return jnp.sum(apply_native_sparse_attention(q, k, v, block_indices, 1, block_size))

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.all(jnp.isfinite(dq)), "dq contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dk)), "dk contains NaN or Inf"
        assert jnp.all(jnp.isfinite(dv)), "dv contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
