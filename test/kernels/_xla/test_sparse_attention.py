"""Tests for XLA native sparse attention implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import apply_native_sparse_attention


class TestNativeSparseAttention:
    """Test suite for native sparse attention kernel."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
        block_size = 32
        num_blocks = seq_len // block_size

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        # Each query block attends to 2 key blocks
        block_counts = 2
        block_indices = jnp.tile(jnp.arange(2)[None, None, None, :], (batch, num_heads, num_blocks, 1))

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)

        assert output.shape == (batch, seq_len, num_heads, head_dim)

    def test_gradient_shapes(self):
        """Test gradient shapes match input shapes."""
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32
        num_blocks = seq_len // block_size

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_counts = 1
        block_indices = jnp.tile(jnp.arange(1)[None, None, None, :], (batch, num_heads, num_blocks, 1))

        def loss_fn(q, k, v):
            return jnp.sum(apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size))

        dq, dk, dv = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape

    def test_local_attention_pattern(self):
        """Test local attention pattern (each block attends to itself)."""
        batch, seq_len, num_heads, head_dim = 1, 64, 1, 8
        block_size = 16
        num_blocks = seq_len // block_size

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        # Each block attends only to itself
        block_indices = jnp.arange(num_blocks)[None, None, :, None]
        block_indices = jnp.tile(block_indices, (batch, num_heads, 1, 1))
        block_counts = 1

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)

        assert output.shape == (batch, seq_len, num_heads, head_dim)
        # With uniform q, k, v, output should be close to v
        assert jnp.allclose(output, v, atol=1e-5)

    def test_custom_scale(self):
        """Test with custom attention scale parameter."""
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32
        num_blocks = seq_len // block_size

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        block_indices = jnp.tile(jnp.arange(1)[None, None, None, :], (batch, num_heads, num_blocks, 1))
        block_counts = 1

        # Test that custom scale doesn't error
        output_custom = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size, scale=0.5)
        assert output_custom.shape == (batch, seq_len, num_heads, head_dim)

    def test_variable_block_counts(self):
        """Test with different block counts per query block."""
        batch, seq_len, num_heads, head_dim = 1, 64, 2, 16
        block_size = 32
        num_blocks = seq_len // block_size

        q = jnp.ones((batch, seq_len, num_heads, head_dim))
        k = jnp.ones((batch, seq_len, num_heads, head_dim))
        v = jnp.ones((batch, seq_len, num_heads, head_dim))

        # First query block attends to 1 key block, second to 2
        block_counts = jnp.array([[[1, 2]]])  # [batch=1, num_heads=1, num_query_blocks=2]
        block_counts = jnp.tile(block_counts, (batch, num_heads, 1))

        block_indices = jnp.tile(jnp.arange(2)[None, None, None, :], (batch, num_heads, num_blocks, 1))

        output = apply_native_sparse_attention(q, k, v, block_indices, block_counts, block_size)

        assert output.shape == (batch, seq_len, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
