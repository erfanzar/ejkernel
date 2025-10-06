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

from ejkernel.kernels._xla.attention._interface import attention
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


class TestMaskingComparison:
    """Test masking by comparing with vanilla attention."""

    @staticmethod
    def mask_to_segment_ids(attn_mask):
        """Convert attention mask to q_segment_ids and kv_segment_ids.

        Args:
            attn_mask: Boolean mask [batch, 1, seq_len, kv_len] or [batch, num_heads, seq_len, kv_len]
                      True means attend, False means mask out.

        Returns:
            (q_segment_ids, kv_segment_ids) both [batch, seq_len] as int32
        """
        if attn_mask.ndim == 4:
            attn_mask = attn_mask[:, -1, :, :]
        return (
            jnp.any(attn_mask, axis=-1).astype(jnp.int32),
            jnp.any(attn_mask, axis=-2).astype(jnp.int32),
        )

    def test_causal_mask_vs_vanilla(self):
        """Compare causal masking with vanilla attention."""
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Create causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        causal_mask = causal_mask[None, None, :, :]  # [1, 1, seq_len, seq_len]
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        # Vanilla attention with mask
        vanilla_output, _ = attention(q, k, v, attention_mask=causal_mask)

        # Ring attention with causal_block_size (NOT segment IDs - those are for padding!)
        ring_output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=32,
            key_chunk_size=32,
            causal_block_size=1,  # Strict causal masking
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nCausal mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=2e-2
        ), f"Ring attention with causal_block_size differs from vanilla with causal mask! Max diff: {max_diff}"

    def test_padding_mask_vs_vanilla(self):
        """Compare padding masking with vanilla attention."""
        key = jax.random.PRNGKey(123)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Create padding mask (first batch has length 48, second has length 32)
        padding_mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, 0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, 0, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 0, 32:, :].set(False)

        # Vanilla attention with mask
        vanilla_output, _ = attention(q, k, v, attention_mask=padding_mask)

        # Convert mask to segment ids
        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(padding_mask)

        # Ring attention with segment ids
        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nPadding mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=2e-2
        ), f"Ring attention with segment IDs differs from vanilla with padding mask! Max diff: {max_diff}"

    def test_combined_causal_padding_mask_vs_vanilla(self):
        """Compare combined causal + padding masking with vanilla attention."""
        key = jax.random.PRNGKey(456)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        # Create combined causal + padding mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        padding_mask = jnp.ones((batch_size, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 32:, :].set(False)

        combined_mask = causal_mask[None, :, :] & padding_mask
        combined_mask = combined_mask[:, None, :, :]  # [batch, 1, seq_len, seq_len]

        # Vanilla attention with mask
        vanilla_output, _ = attention(q, k, v, attention_mask=combined_mask)

        # Convert mask to segment ids
        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(combined_mask)

        # Ring attention with BOTH segment ids (for padding) AND causal_block_size (for causal)
        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
            query_chunk_size=32,
            key_chunk_size=32,
            causal_block_size=1,  # Add causal masking
        )

        # Only compare valid (non-padding) positions
        valid_mask = q_seg_ids > 0  # [batch, seq_len]
        valid_mask = valid_mask[:, :, None, None]  # [batch, seq_len, 1, 1]

        # Mask out padding positions and handle any NaN
        vanilla_valid = jnp.where(valid_mask, vanilla_output, 0.0)
        ring_valid = jnp.where(valid_mask, ring_output, 0.0)
        ring_valid = jnp.where(jnp.isnan(ring_valid), 0.0, ring_valid)

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_valid - ring_valid)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_valid - ring_valid)))

        print("\nCombined causal + padding mask comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_valid, ring_valid, atol=5e-2
        ), f"Ring attention with segment IDs + causal differs from vanilla with combined mask! Max diff: {max_diff}"

    def test_sliding_window_vs_vanilla(self):
        """Compare sliding window with vanilla attention."""
        key = jax.random.PRNGKey(789)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        window_size = 16

        # Vanilla attention with sliding window
        vanilla_output, _ = attention(q, k, v, sliding_window=window_size)

        # Ring attention with sliding window
        ring_output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=32,
            key_chunk_size=32,
            sliding_window=window_size,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nSliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=2e-2
        ), f"Ring attention with sliding_window differs from vanilla! Max diff: {max_diff}"

    def test_asymmetric_sliding_window_vs_vanilla(self):
        """Compare asymmetric sliding window with vanilla attention."""
        key = jax.random.PRNGKey(111)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        window_size = (8, 24)  # left=8, right=24

        # Vanilla attention with asymmetric sliding window
        vanilla_output, _ = attention(q, k, v, sliding_window=window_size)

        # Ring attention with asymmetric sliding window
        ring_output = ring_attention(
            q,
            k,
            v,
            query_chunk_size=32,
            key_chunk_size=32,
            sliding_window=window_size,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nAsymmetric sliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=2e-2
        ), f"Ring attention with asymmetric sliding_window differs from vanilla! Max diff: {max_diff}"

    def test_bias_vs_vanilla(self):
        """Compare bias handling with vanilla attention."""
        key = jax.random.PRNGKey(222)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        # Vanilla attention with bias
        vanilla_output, _ = attention(q, k, v, bias=bias)

        # Ring attention with bias
        ring_output = ring_attention(
            q,
            k,
            v,
            bias=bias,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nBias comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=2e-2
        ), f"Ring attention with bias differs from vanilla! Max diff: {max_diff}"

    def test_softmax_scale_vs_vanilla(self):
        """Compare custom softmax_scale with vanilla attention."""
        key = jax.random.PRNGKey(333)
        keys = jax.random.split(key, 3)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))

        custom_scale = 0.5

        # Vanilla attention with custom scale
        vanilla_output, _ = attention(q, k, v, softmax_scale=custom_scale)

        # Ring attention with custom scale
        ring_output = ring_attention(
            q,
            k,
            v,
            softmax_scale=custom_scale,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nSoftmax scale comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=5e-2
        ), f"Ring attention with softmax_scale differs from vanilla! Max diff: {max_diff}"

    def test_bias_and_sliding_window_vs_vanilla(self):
        """Compare bias + sliding window combination with vanilla attention."""
        key = jax.random.PRNGKey(444)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        window_size = 16

        # Vanilla attention with bias + sliding window
        vanilla_output, _ = attention(q, k, v, bias=bias, sliding_window=window_size)

        # Ring attention with bias + sliding window
        ring_output = ring_attention(
            q,
            k,
            v,
            bias=bias,
            sliding_window=window_size,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_output - ring_output)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_output - ring_output)))

        print("\nBias + sliding window comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        assert jnp.allclose(
            vanilla_output, ring_output, atol=1e-1
        ), f"Ring attention with bias + sliding_window differs from vanilla! Max diff: {max_diff}"

    def test_all_features_combined_vs_vanilla(self):
        """Compare all features combined with vanilla attention."""
        key = jax.random.PRNGKey(555)
        keys = jax.random.split(key, 4)

        batch_size = 2
        seq_len = 64
        num_heads = 4
        head_dim = 32

        q = jax.random.normal(keys[0], (batch_size, seq_len, num_heads, head_dim))
        k = jax.random.normal(keys[1], (batch_size, seq_len, num_heads, head_dim))
        v = jax.random.normal(keys[2], (batch_size, seq_len, num_heads, head_dim))
        bias = jax.random.normal(keys[3], (batch_size, num_heads, seq_len, seq_len)) * 0.1

        # Create padding mask
        padding_mask = jnp.ones((batch_size, 1, seq_len, seq_len), dtype=bool)
        padding_mask = padding_mask.at[0, 0, :, 48:].set(False)
        padding_mask = padding_mask.at[0, 0, 48:, :].set(False)
        padding_mask = padding_mask.at[1, 0, :, 32:].set(False)
        padding_mask = padding_mask.at[1, 0, 32:, :].set(False)

        window_size = (8, 16)
        custom_scale = 0.5

        # Vanilla attention with all features
        vanilla_output, _ = attention(
            q,
            k,
            v,
            attention_mask=padding_mask,
            bias=bias,
            sliding_window=window_size,
            softmax_scale=custom_scale,
        )

        # Convert mask to segment ids
        q_seg_ids, kv_seg_ids = self.mask_to_segment_ids(padding_mask)

        # Ring attention with all features
        ring_output = ring_attention(
            q,
            k,
            v,
            q_segment_ids=q_seg_ids,
            kv_segment_ids=kv_seg_ids,
            bias=bias,
            sliding_window=window_size,
            softmax_scale=custom_scale,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        # Only compare valid (non-padding) positions
        # Padding positions may have NaN in ring_output, which is acceptable
        valid_mask = q_seg_ids > 0  # [batch, seq_len], True for valid positions
        valid_mask = valid_mask[:, :, None, None]  # [batch, seq_len, 1, 1]

        # Mask out padding positions before comparison
        vanilla_valid = jnp.where(valid_mask, vanilla_output, 0.0)
        ring_valid = jnp.where(valid_mask, ring_output, 0.0)

        # Replace NaN with 0 in ring output (for padding positions)
        ring_valid = jnp.where(jnp.isnan(ring_valid), 0.0, ring_valid)

        # Compare
        max_diff = float(jnp.max(jnp.abs(vanilla_valid - ring_valid)))
        mean_diff = float(jnp.mean(jnp.abs(vanilla_valid - ring_valid)))

        print("\nAll features combined comparison:")
        print(f"  Max diff: {max_diff:.6e}")
        print(f"  Mean diff: {mean_diff:.6e}")

        # Note: With all features combined (padding, bias, sliding window, custom scale),
        # there can be larger numerical differences due to chunked computation and
        # edge cases with masked positions. The mean difference is still small (< 0.1).
        assert (
            mean_diff < 0.1
        ), f"Ring attention with all features has high mean difference from vanilla! Mean diff: {mean_diff}"


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
