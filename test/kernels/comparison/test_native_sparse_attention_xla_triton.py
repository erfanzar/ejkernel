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

"""Comparison tests between XLA and Triton native_sparse_attention implementations.

Note: Triton implementation requires float16/bfloat16, while XLA uses float32.
Comparisons account for precision differences between these dtypes.

Both implementations now support GQA (Grouped Query Attention).
"""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.native_sparse_attention import (
    apply_native_sparse_attention as apply_nsa_triton,
)
from ejkernel.kernels._triton.native_sparse_attention import (
    native_sparse_attention as nsa_triton,
)
from ejkernel.kernels._xla.native_sparse_attention import (
    apply_native_sparse_attention as apply_nsa_xla,
)
from ejkernel.kernels._xla.native_sparse_attention import (
    native_sparse_attention as nsa_xla,
)


class TestApplyNativeSparseMAttentionComparison:
    """Compare apply_native_sparse_attention between XLA and Triton."""

    def test_basic_sparse_attention(self):
        """Test that basic sparse attention produces valid results on both backends."""
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 16, 1, 64  # GQA ratio = 16
        block_size = 64
        block_counts = 4
        num_blocks = seq_len // block_size

        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)

        # XLA uses float32, Triton needs float16
        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)

        # Create block indices (attend to first block_counts blocks for each query block)
        block_indices = jnp.tile(
            jnp.arange(block_counts)[None, None, None, :],
            (batch, num_kv_heads, num_blocks, 1),
        ).astype(jnp.int32)

        # Create block_counts array
        block_counts_array = jnp.full((batch, num_kv_heads, num_blocks), block_counts, dtype=jnp.int32)

        scale = float(1.0 / jnp.sqrt(head_dim))

        out_xla = apply_nsa_xla(
            q_f32,
            k_f32,
            v_f32,
            block_indices,
            block_counts_array,
            block_size,
            scale,
        )

        out_triton = apply_nsa_triton(
            q_f16,
            k_f16,
            v_f16,
            block_indices,
            block_counts_array,
            block_size,
            scale,
        )

        # Check both produce valid outputs
        assert out_xla.shape == (batch, seq_len, num_q_heads, head_dim)
        assert out_triton.shape == (batch, seq_len, num_q_heads, head_dim)
        assert not jnp.any(jnp.isnan(out_xla))
        assert not jnp.any(jnp.isnan(out_triton))

        # Should be reasonably close (accounting for float16 precision)
        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=1e-1, atol=1e-2)

    def test_uniform_block_counts(self):
        """Test with uniform block counts (int instead of array)."""
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 16, 1, 32  # GQA ratio = 16
        block_size = 64
        block_counts = 2
        num_blocks = seq_len // block_size

        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        q_f16 = q_f32.astype(jnp.float16)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        k_f16 = k_f32.astype(jnp.float16)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f16 = v_f32.astype(jnp.float16)

        # Create causal block indices
        block_indices = jnp.zeros((batch, num_kv_heads, num_blocks, block_counts), dtype=jnp.int32)
        for qb in range(num_blocks):
            # Each query block attends to itself and the previous block (or 0 if first)
            indices = jnp.arange(max(0, qb - block_counts + 1), qb + 1)
            # Pad if needed
            if len(indices) < block_counts:
                indices = jnp.pad(indices, (0, block_counts - len(indices)), constant_values=0)
            block_indices = block_indices.at[:, :, qb, :].set(indices)

        scale = float(1.0 / jnp.sqrt(head_dim))

        out_xla = apply_nsa_xla(
            q_f32,
            k_f32,
            v_f32,
            block_indices,
            block_counts,  # Use int
            block_size,
            scale,
        )

        out_triton = apply_nsa_triton(
            q_f16,
            k_f16,
            v_f16,
            block_indices,
            block_counts,  # Use int
            block_size,
            scale,
        )

        # Should be reasonably close
        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=5e-2, atol=5e-3)


class TestNativeSparseAttentionComparison:
    """Compare full native_sparse_attention with compression between XLA and Triton."""

    def test_with_precomputed_indices(self):
        """Test NSA with pre-computed block indices (no compression)."""
        batch, seq_len, num_heads, head_dim = 2, 256, 16, 64
        block_size = 64
        block_counts = 4
        num_blocks = seq_len // block_size

        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        q_f16 = q_f32.astype(jnp.float16)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k_f16 = k_f32.astype(jnp.float16)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v_f16 = v_f32.astype(jnp.float16)

        # Create block indices
        block_indices = jnp.tile(
            jnp.arange(block_counts)[None, None, None, :],
            (batch, num_heads, num_blocks, 1),
        ).astype(jnp.int32)

        out_xla = nsa_xla(
            q_f32,
            k_f32,
            v_f32,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
        )

        out_triton = nsa_triton(
            q_f16,
            k_f16,
            v_f16,
            block_indices=block_indices,
            block_counts=block_counts,
            block_size=block_size,
        )

        # Should be reasonably close (accounting for float16 precision)
        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=5e-2, atol=5e-3)

    def test_with_compression_and_selection(self):
        """Test NSA with compression and automatic block selection."""
        batch, seq_len, num_heads, head_dim = 2, 256, 16, 64
        block_size = 64
        block_counts = 4

        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 4)

        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        q_f16 = q_f32.astype(jnp.float16)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k_f16 = k_f32.astype(jnp.float16)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v_f16 = v_f32.astype(jnp.float16)

        # Create compression gate (note: type annotation says hidden_dim but actually num_heads)
        g_cmp_f32 = jax.random.uniform(keys[3], (batch, seq_len, num_heads), dtype=jnp.float32)
        g_cmp_f16 = g_cmp_f32.astype(jnp.float16)

        out_xla = nsa_xla(
            q_f32,
            k_f32,
            v_f32,
            g_cmp=g_cmp_f32,
            block_counts=block_counts,
            block_size=block_size,
        )

        out_triton = nsa_triton(
            q_f16,
            k_f16,
            v_f16,
            g_cmp=g_cmp_f16,
            block_counts=block_counts,
            block_size=block_size,
        )

        # With compression, results might differ more due to top-k selection
        # and float16 accumulation, so use more relaxed tolerances
        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=1e-1, atol=1e-2)

    def test_with_both_gates(self):
        """Test NSA with both compression and selection gates."""
        batch, seq_len, num_heads, head_dim = 1, 128, 16, 32
        block_size = 64
        block_counts = 2

        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 5)

        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        q_f16 = q_f32.astype(jnp.float16)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k_f16 = k_f32.astype(jnp.float16)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v_f16 = v_f32.astype(jnp.float16)

        # Create gates
        g_cmp_f32 = jax.random.uniform(keys[3], (batch, seq_len, num_heads), dtype=jnp.float32)
        g_cmp_f16 = g_cmp_f32.astype(jnp.float16)
        g_slc_f32 = jax.random.uniform(keys[4], (batch, seq_len, num_heads), dtype=jnp.float32)
        g_slc_f16 = g_slc_f32.astype(jnp.float16)

        out_xla = nsa_xla(
            q_f32,
            k_f32,
            v_f32,
            g_cmp=g_cmp_f32,
            g_slc=g_slc_f32,
            block_counts=block_counts,
            block_size=block_size,
        )

        out_triton = nsa_triton(
            q_f16,
            k_f16,
            v_f16,
            g_cmp=g_cmp_f16,
            g_slc=g_slc_f16,
            block_counts=block_counts,
            block_size=block_size,
        )

        # With both gates, results might differ more
        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=1e-1, atol=1e-2)


class TestGradientComparison:
    """Compare gradients between XLA and Triton implementations."""

    def test_gradient_agreement(self):
        """Test that gradients are similar between implementations."""
        batch, seq_len, num_heads, head_dim = 1, 128, 16, 32
        block_size = 64
        block_counts = 2
        num_blocks = seq_len // block_size

        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        q_f32 = jax.random.normal(keys[0], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        q_f16 = q_f32.astype(jnp.float16)
        k_f32 = jax.random.normal(keys[1], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        k_f16 = k_f32.astype(jnp.float16)
        v_f32 = jax.random.normal(keys[2], (batch, seq_len, num_heads, head_dim), dtype=jnp.float32)
        v_f16 = v_f32.astype(jnp.float16)

        # Create block indices
        block_indices = jnp.tile(
            jnp.arange(block_counts)[None, None, None, :],
            (batch, num_heads, num_blocks, 1),
        ).astype(jnp.int32)

        scale = float(1.0 / jnp.sqrt(head_dim))

        # Define loss function
        def loss_fn_xla(q, k, v):
            out = apply_nsa_xla(q, k, v, block_indices, block_counts, block_size, scale)
            return jnp.sum(out**2)

        def loss_fn_triton(q, k, v):
            out = apply_nsa_triton(q, k, v, block_indices, block_counts, block_size, scale)
            return jnp.sum(out**2)

        # Compute gradients
        grad_xla = jax.grad(loss_fn_xla, argnums=(0, 1, 2))(q_f32, k_f32, v_f32)
        grad_triton = jax.grad(loss_fn_triton, argnums=(0, 1, 2))(q_f16, k_f16, v_f16)

        # Check gradients are similar
        for g_xla, g_triton in zip(grad_xla, grad_triton, strict=False):
            assert jnp.allclose(g_xla, g_triton.astype(jnp.float32), rtol=1e-1, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
