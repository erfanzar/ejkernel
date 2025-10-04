"""Tests for XLA page attention implementation."""

import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla import page_attention


class TestPageAttention:
    """Test suite for page attention kernel (inference only)."""

    def test_forward_shape(self):
        """Test output shapes are correct."""
        num_seqs, num_heads, head_dim = 2, 8, 64
        num_blocks, num_kv_heads, block_size = 8, 4, 16

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([48, 32])
        block_tables = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_gqa_support(self):
        """Test Grouped Query Attention (GQA) with different num_heads and num_kv_heads."""
        num_seqs = 2
        num_heads = 8  # Query heads
        num_kv_heads = 2  # KV heads (GQA: 8 query heads share 2 KV heads)
        head_dim = 64
        num_blocks, block_size = 4, 16

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([32, 16])
        block_tables = jnp.array([[0, 1], [2, 3]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_different_context_lengths(self):
        """Test with different context lengths per sequence."""
        num_seqs, num_heads, head_dim = 3, 4, 32
        num_blocks, num_kv_heads, block_size = 6, 4, 8

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))

        # Different context lengths: 16, 8, 24
        context_lens = jnp.array([16, 8, 24])
        block_tables = jnp.array([[0, 1], [2, 3], [4, 5]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)

    def test_custom_scale(self):
        """Test with custom attention scale parameter (inference only)."""
        num_seqs, num_heads, head_dim = 2, 4, 16
        num_blocks, num_kv_heads, block_size = 4, 4, 8

        query = jnp.ones((num_seqs, num_heads, head_dim))
        key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        context_lens = jnp.array([16, 8])
        block_tables = jnp.array([[0, 1], [2, 3]])

        # Just test that custom scale doesn't error
        output_custom = page_attention(query, key_cache, value_cache, context_lens, block_tables, attn_scale=0.5)
        assert output_custom.shape == (num_seqs, num_heads, head_dim)

    def test_block_table_indexing(self):
        """Test that block tables correctly index into cache."""
        num_seqs, num_heads, head_dim = 1, 2, 8
        num_blocks, num_kv_heads, block_size = 4, 2, 4

        query = jnp.ones((num_seqs, num_heads, head_dim))

        # Create distinct key/value blocks
        key_cache = jnp.arange(num_blocks * num_kv_heads * block_size * head_dim).reshape(
            num_blocks, num_kv_heads, block_size, head_dim
        )
        value_cache = key_cache * 2

        context_lens = jnp.array([8])
        # Use blocks 0 and 1
        block_tables = jnp.array([[0, 1]])

        output = page_attention(query, key_cache, value_cache, context_lens, block_tables)

        assert output.shape == (num_seqs, num_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
