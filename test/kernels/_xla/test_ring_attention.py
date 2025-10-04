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

"""Unified tests for ring_attention (XLA implementation)."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._xla.ring_attention._interface import ring_attention


class TestBasicFunctionality:
    """Test basic ring attention functionality."""

    def test_basic_ring_attention(self):
        """Test basic ring attention without special features."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_causal_masking(self):
        """Test with causal masking."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Non-causal
        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # Causal
        output_causal = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            causal_block_size=64,
        )

        assert output_causal.shape == q.shape
        diff = float(jnp.mean(jnp.abs(output - output_causal)))
        assert diff > 1e-6, "Causal masking should affect output"

    def test_different_block_sizes(self):
        """Test that different block sizes produce similar outputs."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = []
        for blocksize in [32, 64, 128]:
            output = ring_attention(
                q,
                k,
                v,
                query_chunk_size=blocksize,
                key_chunk_size=blocksize,
            )
            outputs.append(output)

        # Different block sizes should produce very similar results
        for i in range(len(outputs) - 1):
            diff = float(jnp.mean(jnp.abs(outputs[i] - outputs[i + 1])))
            assert diff < 1e-3, f"Block size configuration affects output too much: {diff}"


class TestSlidingWindow:
    """Test sliding window attention."""

    def test_symmetric_sliding_window(self):
        """Test symmetric sliding window (int)."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 8
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Full attention
        output_full = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # Windowed attention
        output_windowed = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        assert output_windowed.shape == q.shape
        assert jnp.all(jnp.isfinite(output_windowed))

        diff = float(jnp.mean(jnp.abs(output_full - output_windowed)))
        assert diff > 1e-6, "Sliding window should affect output"

    def test_asymmetric_sliding_window(self):
        """Test asymmetric sliding window (tuple)."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Symmetric window
        output_sym = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        # Asymmetric window (left=32, right=96)
        output_asym = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=(32, 96),
        )

        assert output_asym.shape == q.shape
        assert jnp.all(jnp.isfinite(output_asym))

        diff = float(jnp.mean(jnp.abs(output_sym - output_asym)))
        assert diff > 1e-6, "Asymmetric should differ from symmetric"

    def test_different_window_sizes(self):
        """Test different window sizes."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = {}
        for window_size in [32, 64, 128]:
            outputs[window_size] = ring_attention(
                q,
                k,
                v,
                query_chunk_size=64,
                key_chunk_size=64,
                sliding_window=window_size,
            )

        # Different window sizes should produce different outputs
        diff_32_64 = float(jnp.mean(jnp.abs(outputs[32] - outputs[64])))
        diff_64_128 = float(jnp.mean(jnp.abs(outputs[64] - outputs[128])))

        assert diff_32_64 > 1e-6
        assert diff_64_128 > 1e-6


class TestLogitSoftCap:
    """Test logit soft cap feature."""

    def test_soft_cap_affects_output(self):
        """Test that soft cap actually changes the output."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Without soft cap
        output_no_cap = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # With soft cap
        output_with_cap = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            logit_soft_cap=30.0,
        )

        assert output_with_cap.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_cap))

        diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap)))
        assert diff > 1e-6, "Soft cap should affect output"

    def test_soft_cap_numerical_stability(self):
        """Test that soft cap prevents overflow with large values."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        # Create large queries to test stability
        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim)) * 10.0
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            logit_soft_cap=30.0,
        )

        assert jnp.all(jnp.isfinite(output)), "Soft cap should prevent overflow"


class TestAttentionSink:
    """Test attention sink feature."""

    def test_attention_sink_affects_output(self):
        """Test that attention sink changes the output."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Without attention sink
        output_no_sink = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        # With attention sink
        output_with_sink = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
            attention_sink_size=4,
        )

        assert output_with_sink.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_sink))

        diff = float(jnp.mean(jnp.abs(output_no_sink - output_with_sink)))
        assert diff > 1e-6, "Attention sink should affect output"

    def test_different_sink_sizes(self):
        """Test different attention sink sizes."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        outputs = {}
        for sink_size in [2, 4, 8]:
            outputs[sink_size] = ring_attention(
                q,
                k,
                v,
                query_chunk_size=64,
                key_chunk_size=64,
                sliding_window=64,
                attention_sink_size=sink_size,
            )

        # Different sink sizes should produce different outputs
        diff_2_4 = float(jnp.mean(jnp.abs(outputs[2] - outputs[4])))
        diff_4_8 = float(jnp.mean(jnp.abs(outputs[4] - outputs[8])))

        assert diff_2_4 > 1e-6
        assert diff_4_8 > 1e-6


class TestCombinedFeatures:
    """Test combinations of features."""

    def test_window_and_soft_cap(self):
        """Test sliding window with soft cap."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
            logit_soft_cap=30.0,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_asymmetric_window_and_sink(self):
        """Test asymmetric window with attention sink."""
        key = jax.random.PRNGKey(222)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=(32, 96),
            attention_sink_size=8,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    def test_all_features_combined(self):
        """Test all features together."""
        key = jax.random.PRNGKey(333)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 256
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=64,
            key_chunk_size=64,
            causal_block_size=64,  # causal
            sliding_window=(32, 96),
            attention_sink_size=4,
            logit_soft_cap=30.0,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


class TestGradients:
    """Test gradient computation."""

    def test_basic_gradients(self):
        """Test that gradients work for basic case."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            output = ring_attention(
                q,
                k,
                v,
                query_chunk_size=32,
                key_chunk_size=32,
            )
            return jnp.mean(output**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)

    def test_gradients_with_features(self):
        """Test gradients with advanced features."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        def loss_fn(q, k, v):
            output = ring_attention(
                q,
                k,
                v,
                query_chunk_size=32,
                key_chunk_size=32,
                sliding_window=32,
                logit_soft_cap=30.0,
                attention_sink_size=2,
            )
            return jnp.mean(output**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)

        assert jnp.isfinite(loss)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
