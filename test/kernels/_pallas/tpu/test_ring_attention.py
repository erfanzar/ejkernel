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

"""Tests for Pallas TPU ring_attention implementation."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._pallas.tpu.ring_attention._interface import ring_attention


class TestBasicFunctionality:
    """Test basic ring attention functionality."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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

        # Single device test (no ring communication)
        output = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # Causal
        output_causal = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            causal_block_size=64,
        )

        assert output_causal.shape == q.shape
        diff = float(jnp.mean(jnp.abs(output - output_causal)))
        assert diff > 1e-6, "Causal masking should affect output"


class TestSlidingWindow:
    """Test sliding window attention."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # Windowed attention
        output_windowed = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        assert output_windowed.shape == q.shape
        assert jnp.all(jnp.isfinite(output_windowed))

        diff = float(jnp.mean(jnp.abs(output_full - output_windowed)))
        assert diff > 1e-6, "Sliding window should affect output"

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        # Asymmetric window (left=32, right=96)
        output_asym = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=(32, 96),
        )

        assert output_asym.shape == q.shape
        assert jnp.all(jnp.isfinite(output_asym))

        diff = float(jnp.mean(jnp.abs(output_sym - output_asym)))
        assert diff > 1e-6, "Asymmetric should differ from symmetric"


class TestLogitSoftCap:
    """Test logit soft cap feature."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        # With soft cap
        output_with_cap = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            logit_soft_cap=30.0,
        )

        assert output_with_cap.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_cap))

        diff = float(jnp.mean(jnp.abs(output_no_cap - output_with_cap)))
        assert diff > 1e-6, "Soft cap should affect output"


class TestAttentionSink:
    """Test attention sink feature."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
        )

        # With attention sink
        output_with_sink = ring_attention(
            q,
            k,
            v,
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            sliding_window=64,
            attention_sink_size=4,
        )

        assert output_with_sink.shape == q.shape
        assert jnp.all(jnp.isfinite(output_with_sink))

        diff = float(jnp.mean(jnp.abs(output_no_sink - output_with_sink)))
        assert diff > 1e-6, "Attention sink should affect output"


class TestSegmentIds:
    """Test segment IDs feature."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
    def test_separate_segment_ids(self):
        """Test separate Q and KV segment IDs."""
        key = jax.random.PRNGKey(999)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Create segment IDs
        q_segment_ids = jnp.array([[0] * 64 + [1] * 64, [0] * 64 + [1] * 64])
        kv_segment_ids = jnp.array([[0] * 64 + [1] * 64, [0] * 32 + [1] * 96])

        output = ring_attention(
            q,
            k,
            v,
            bias=None,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            query_chunk_size=64,
            key_chunk_size=64,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


class TestCombinedFeatures:
    """Test combinations of features."""

    @pytest.mark.skipif(
        not jax.devices()[0].platform == "tpu",
        reason="Pallas ring attention requires TPU",
    )
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
            bias=None,
            query_chunk_size=64,
            key_chunk_size=64,
            causal_block_size=64,
            sliding_window=(32, 96),
            attention_sink_size=4,
            logit_soft_cap=30.0,
        )

        assert output.shape == q.shape
        assert jnp.all(jnp.isfinite(output))


class TestAPICompatibility:
    """Test API compatibility and parameter handling."""

    def test_import_success(self):
        """Test that the module imports successfully."""
        from ejkernel.kernels._pallas.tpu.ring_attention._interface import ring_attention

        assert ring_attention is not None

    def test_parameter_defaults(self):
        """Test that parameter defaults are set correctly."""
        # This should not raise any errors
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Call with minimal parameters (only required ones)
        # Note: This will be skipped on non-TPU but tests the API
        if jax.devices()[0].platform == "tpu":
            output = ring_attention(q, k, v, bias=None)
            assert output.shape == q.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
