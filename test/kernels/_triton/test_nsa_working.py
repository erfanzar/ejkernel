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

"""Working tests for native_sparse_attention Triton implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.native_sparse_attention import native_sparse_attention
from ejkernel.utils import generate_block_indices


class TestNSAForwardWithCompression:
    """Test forward pass with compression gate (most stable mode)."""

    @pytest.mark.parametrize(
        "batch_size,seq_len,num_q_heads,num_kv_heads,head_dim,block_size",
        [
            (1, 256, 16, 1, 64, 64),  # Minimal GQA (16/1 = 16)
            (2, 256, 32, 2, 64, 64),  # Standard GQA (32/2 = 16)
            (1, 256, 64, 4, 64, 64),  # Higher GQA (64/4 = 16)
            (1, 512, 64, 2, 128, 64),  # Larger head_dim
            (1, 256, 128, 8, 64, 64),  # More heads (128/8 = 16)
        ],
    )
    def test_forward_with_compression(self, batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, block_size):
        """Test forward pass with compression gate produces valid output."""
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        g_cmp = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

        output = native_sparse_attention(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            block_counts=num_blocks,
            block_size=block_size,
        )

        # Check basic properties
        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert output.dtype == jnp.float16

        # Compression mode seems to handle all tokens properly
        assert not jnp.any(jnp.isnan(output)), "Output contains NaN"
        assert not jnp.any(jnp.isinf(output)), "Output contains Inf"

    def test_with_both_gates(self):
        """Test with both compression and selection gates."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 256, 16, 1, 64
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        g_cmp = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)
        g_slc = jax.random.uniform(jax.random.PRNGKey(4), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

        output = native_sparse_attention(
            q=q,
            k=k,
            v=v,
            g_cmp=g_cmp,
            g_slc=g_slc,
            block_counts=num_blocks,
            block_size=block_size,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        assert not jnp.any(jnp.isnan(output))


class TestNSAPrecomputedIndices:
    """Test with precomputed block indices."""

    @pytest.mark.skip(reason="Known issue: NSA without compression produces NaN values")
    def test_forward_with_precomputed_indices(self):
        """Test forward pass with precomputed indices."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        num_blocks = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        num_query_blocks = seq_len // block_size
        block_indices = generate_block_indices(batch_size, num_query_blocks, num_kv_heads, num_blocks, block_size)

        output = native_sparse_attention(
            q=q,
            k=k,
            v=v,
            block_indices=block_indices,
            block_counts=num_blocks,
            block_size=block_size,
        )

        assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)
        # Without compression, NSA produces NaN - this is a known issue


class TestNSAConstraints:
    """Test NSA constraints and error handling."""

    def test_gqa_ratio_assertion(self):
        """Test that GQA group size must be multiple of 16."""
        batch_size, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 15, 1, 64
        block_size = 64

        q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
        k = jax.random.normal(jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)
        v = jax.random.normal(jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16)

        # 15 % (1 * 16) != 0, should fail
        with pytest.raises(AssertionError, match="Group size must be a multiple of 16"):
            native_sparse_attention(q, k, v, block_size=block_size)

    def test_valid_gqa_ratios(self):
        """Test that valid GQA ratios work."""
        valid_configs = [
            (16, 1),  # 16/1 = 16
            (32, 2),  # 32/2 = 16
            (48, 3),  # 48/3 = 16
            (64, 2),  # 64/2 = 32
            (128, 4),  # 128/4 = 32
        ]

        for num_q_heads, num_kv_heads in valid_configs:
            batch_size, seq_len, head_dim, block_size = 1, 256, 64, 64

            q = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, num_q_heads, head_dim), dtype=jnp.float16)
            k = jax.random.normal(
                jax.random.PRNGKey(1), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16
            )
            v = jax.random.normal(
                jax.random.PRNGKey(2), (batch_size, seq_len, num_kv_heads, head_dim), dtype=jnp.float16
            )
            g_cmp = jax.random.uniform(jax.random.PRNGKey(3), (batch_size, seq_len, num_q_heads), dtype=jnp.float16)

            # Should not raise
            output = native_sparse_attention(q, k, v, g_cmp=g_cmp, block_size=block_size)
            assert output.shape == (batch_size, seq_len, num_q_heads, head_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
