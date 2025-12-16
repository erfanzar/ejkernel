"""Tests for position IDs and cu_seqlens computation in MaskInfo."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.types.mask import (
    MaskInfo,
    _positions_from_segments_2d,
    qkv_masks_to_cu_seqlens,
    cu_seqlens_to_mask,
)


class TestPositionsFromSegments2D:
    """Tests for _positions_from_segments_2d helper function."""

    def test_single_segment_no_padding(self):
        """All tokens in same segment should have positions 0, 1, 2, ..."""
        segment_ids = jnp.array([[1, 1, 1, 1]])  # batch=1, seq=4
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        expected = jnp.array([[0, 1, 2, 3]])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_multiple_segments_reset_positions(self):
        """Positions should reset to 0 at segment boundaries."""
        segment_ids = jnp.array([[1, 1, 2, 2, 3]])  # 3 segments
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        expected = jnp.array([[0, 1, 0, 1, 0]])  # resets at each new segment
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_padding_gets_pad_value(self):
        """Padding tokens (-1) should get the specified pad_value."""
        segment_ids = jnp.array([[1, 1, -1, -1]])
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        expected = jnp.array([[0, 1, -1, -1]])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_padding_with_different_pad_value(self):
        """Test with a different pad_value (e.g., int32 max for KV)."""
        segment_ids = jnp.array([[1, 1, -1]])
        pad_val = jnp.iinfo(jnp.int32).max
        positions = _positions_from_segments_2d(segment_ids, pad_value=pad_val)
        expected = jnp.array([[0, 1, pad_val]])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_batched_positions(self):
        """Test with multiple batch elements."""
        segment_ids = jnp.array([
            [1, 1, 2, 2],
            [1, 1, 1, -1],
        ])
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        expected = jnp.array([
            [0, 1, 0, 1],  # two segments
            [0, 1, 2, -1],  # one segment + padding
        ])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_interleaved_padding(self):
        """Test with padding in the middle (unusual but valid)."""
        segment_ids = jnp.array([[1, -1, 1, 1]])
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        # After padding, same segment continues, so positions should reset?
        # Actually the implementation treats the segment after padding as a "new" segment
        # because prev_seg is set to -2 after padding
        expected = jnp.array([[0, -1, 0, 1]])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"

    def test_segment_zero_is_valid(self):
        """Segment ID 0 should be treated as valid (not padding)."""
        segment_ids = jnp.array([[0, 0, 1, 1]])
        positions = _positions_from_segments_2d(segment_ids, pad_value=-1)
        expected = jnp.array([[0, 1, 0, 1]])
        assert jnp.array_equal(positions, expected), f"Got {positions}, expected {expected}"


class TestQPositionIds:
    """Tests for MaskInfo.q_position_ids property."""

    def test_no_padding(self):
        """All valid tokens should have cumulative positions."""
        segment_ids = jnp.array([[1, 1, 1, 1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        pos = mask_info.q_position_ids
        expected = jnp.array([[0, 1, 2, 3]])
        assert jnp.array_equal(pos, expected), f"Got {pos}, expected {expected}"

    def test_with_padding(self):
        """Padding tokens should get position -1."""
        segment_ids = jnp.array([[1, 1, -1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        pos = mask_info.q_position_ids
        # Valid tokens get sequential positions, padding gets -1
        expected = jnp.array([[0, 1, -1, -1]])
        assert jnp.array_equal(pos, expected), f"Got {pos}, expected {expected}"

    def test_all_padding(self):
        """All padding should result in all -1."""
        segment_ids = jnp.array([[-1, -1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        pos = mask_info.q_position_ids
        # All padding tokens get -1
        expected = jnp.array([[-1, -1, -1]])
        assert jnp.array_equal(pos, expected), f"Got {pos}, expected {expected}"

    def test_batched(self):
        """Test with multiple batch elements."""
        segment_ids = jnp.array([
            [1, 1, 1, -1],
            [1, 1, -1, -1],
        ])
        mask_info = MaskInfo.from_segments(segment_ids)
        pos = mask_info.q_position_ids
        # Valid tokens get sequential positions, padding gets -1
        expected = jnp.array([
            [0, 1, 2, -1],
            [0, 1, -1, -1],
        ])
        assert jnp.array_equal(pos, expected), f"Got {pos}, expected {expected}"


class TestGetOrComputePositions:
    """Tests for MaskInfo.get_or_compute_positions method."""

    def test_single_segment(self):
        """Single segment should have simple sequential positions."""
        segment_ids = jnp.array([[1, 1, 1, 1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        
        expected = jnp.array([[0, 1, 2, 3]])
        assert jnp.array_equal(q_pos, expected), f"q_pos: {q_pos}, expected {expected}"
        assert jnp.array_equal(kv_pos, expected), f"kv_pos: {kv_pos}, expected {expected}"

    def test_multiple_segments(self):
        """Multiple segments should have positions that reset at boundaries."""
        segment_ids = jnp.array([[1, 1, 2, 2]])
        mask_info = MaskInfo.from_segments(segment_ids)
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        
        expected = jnp.array([[0, 1, 0, 1]])
        assert jnp.array_equal(q_pos, expected), f"q_pos: {q_pos}, expected {expected}"
        assert jnp.array_equal(kv_pos, expected), f"kv_pos: {kv_pos}, expected {expected}"

    def test_with_padding_q(self):
        """Query positions for padding should be -1."""
        segment_ids = jnp.array([[1, 1, -1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        q_pos, _ = mask_info.get_or_compute_positions()
        
        expected = jnp.array([[0, 1, -1, -1]])
        assert jnp.array_equal(q_pos, expected), f"q_pos: {q_pos}, expected {expected}"

    def test_with_padding_kv(self):
        """KV positions for padding should be int32 max."""
        segment_ids = jnp.array([[1, 1, -1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        _, kv_pos = mask_info.get_or_compute_positions()
        
        pad_val = jnp.iinfo(jnp.int32).max
        expected = jnp.array([[0, 1, pad_val, pad_val]])
        assert jnp.array_equal(kv_pos, expected), f"kv_pos: {kv_pos}, expected {expected}"

    def test_cross_attention(self):
        """Test cross-attention with different Q and KV segment IDs."""
        q_seg = jnp.array([[1, 2]])  # 2 query tokens
        kv_seg = jnp.array([[1, 1, 2, 2, -1]])  # 5 KV tokens (4 valid + 1 pad)
        mask_info = MaskInfo.from_segments(q_seg, kv_seg)
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        
        expected_q = jnp.array([[0, 0]])  # each in its own segment
        expected_kv = jnp.array([[0, 1, 0, 1, jnp.iinfo(jnp.int32).max]])
        assert jnp.array_equal(q_pos, expected_q), f"q_pos: {q_pos}, expected {expected_q}"
        assert jnp.array_equal(kv_pos, expected_kv), f"kv_pos: {kv_pos}, expected {expected_kv}"

    def test_preserves_provided_positions(self):
        """If positions are already provided, they should not be recomputed."""
        segment_ids = jnp.array([[1, 1, 1, 1]])
        custom_q_pos = jnp.array([[10, 20, 30, 40]])
        custom_kv_pos = jnp.array([[100, 200, 300, 400]])
        
        mask_info = MaskInfo.from_segments(
            segment_ids, 
            q_positions=custom_q_pos,
            kv_positions=custom_kv_pos
        )
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        
        assert jnp.array_equal(q_pos, custom_q_pos), "Custom q_positions should be preserved"
        assert jnp.array_equal(kv_pos, custom_kv_pos), "Custom kv_positions should be preserved"


class TestCuSeqlens:
    """Tests for cumulative sequence length computation."""

    def test_qkv_masks_to_cu_seqlens_basic(self):
        """Basic test for qkv_masks_to_cu_seqlens - returns start/end positions."""
        q_mask = jnp.array([
            [True, True, True, False],  # valid at 0-2 (start=0, end=3)
            [True, True, False, False],  # valid at 0-1 (start=0, end=2)
        ])
        cu_q, cu_kv = qkv_masks_to_cu_seqlens(q_mask)

        # New format: [start_0, end_0, start_1, end_1]
        expected = jnp.array([0, 3, 0, 2])
        assert jnp.array_equal(cu_q, expected), f"cu_q: {cu_q}, expected {expected}"
        assert jnp.array_equal(cu_kv, expected), f"cu_kv: {cu_kv}, expected {expected}"

    def test_qkv_masks_to_cu_seqlens_cross_attention(self):
        """Test with different Q and KV masks."""
        q_mask = jnp.array([[True, True, False]])  # valid at 0-1 (start=0, end=2)
        kv_mask = jnp.array([[True, True, True, False, False]])  # valid at 0-2 (start=0, end=3)

        cu_q, cu_kv = qkv_masks_to_cu_seqlens(q_mask, kv_mask)

        expected_q = jnp.array([0, 2])
        expected_kv = jnp.array([0, 3])
        assert jnp.array_equal(cu_q, expected_q), f"cu_q: {cu_q}, expected {expected_q}"
        assert jnp.array_equal(cu_kv, expected_kv), f"cu_kv: {cu_kv}, expected {expected_kv}"

    def test_cu_seqlens_to_mask_basic(self):
        """Test cu_seqlens_to_mask reconstruction with new format."""
        # New format: [start_0, end_0, start_1, end_1]
        cu_seqlens = jnp.array([0, 3, 0, 2])  # batch 0: 0-2, batch 1: 0-1
        mask = cu_seqlens_to_mask(cu_seqlens, max_len=4)

        expected = jnp.array([
            [True, True, True, False],
            [True, True, False, False],
        ])
        assert jnp.array_equal(mask, expected), f"mask: {mask}, expected {expected}"

    def test_cu_seqlens_to_mask_non_prefix(self):
        """Test cu_seqlens_to_mask with non-prefix valid tokens."""
        # Valid tokens in the middle of the sequence
        cu_seqlens = jnp.array([41, 169])  # valid at positions 41-168
        mask = cu_seqlens_to_mask(cu_seqlens, max_len=512)

        assert mask.shape == (1, 512)
        assert mask[0, 40] == False
        assert mask[0, 41] == True
        assert mask[0, 168] == True
        assert mask[0, 169] == False
        assert mask.sum() == 128

    def test_cu_seqlens_roundtrip(self):
        """Test that mask -> cu_seqlens -> mask is identity for contiguous masks."""
        original_mask = jnp.array([
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, False, False, False, False],
        ])

        cu_q, _ = qkv_masks_to_cu_seqlens(original_mask)
        reconstructed = cu_seqlens_to_mask(cu_q, max_len=5)

        assert jnp.array_equal(reconstructed, original_mask), \
            f"Roundtrip failed: {reconstructed} vs {original_mask}"

    def test_cu_seqlens_roundtrip_non_prefix(self):
        """Test roundtrip for masks with non-prefix valid tokens."""
        # Valid tokens in different positions
        original_mask = jnp.zeros((2, 512), dtype=jnp.bool_)
        original_mask = original_mask.at[0, 41:169].set(True)  # batch 0: 41-168
        original_mask = original_mask.at[1, 100:200].set(True)  # batch 1: 100-199

        cu_q, _ = qkv_masks_to_cu_seqlens(original_mask)
        reconstructed = cu_seqlens_to_mask(cu_q, max_len=512)

        assert jnp.array_equal(reconstructed, original_mask), \
            f"Roundtrip failed for non-prefix masks"


class TestMaskInfoCuSeqlens:
    """Tests for MaskInfo.get_or_compute_qkv_cu_seqlens method."""

    def test_from_segment_ids(self):
        """Test cu_seqlens computation from segment IDs."""
        # Both batch elements use segment ID 1, so they count as a single segment
        segment_ids = jnp.array([
            [1, 1, 1, -1],  # 3 tokens with segment ID 1
            [1, 1, -1, -1],  # 2 tokens with segment ID 1
        ])
        mask_info = MaskInfo.from_segments(segment_ids)
        cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens(max_segments=4)

        # Cumulative format: [0, count_seg_0, count_seg_0+count_seg_1, ...]
        # Segment ID 0 doesn't exist -> 0 tokens
        # Segment ID 1 has 3+2=5 tokens total
        # With max_segments=4, output has size 5: [0, 0, 5, 5, 5]
        expected = jnp.array([0, 0, 5, 5, 5])
        assert jnp.array_equal(cu_q, expected), f"cu_q: {cu_q}, expected {expected}"
        assert jnp.array_equal(cu_kv, expected), f"cu_kv: {cu_kv}, expected {expected}"

    def test_from_cu_seqlens_factory(self):
        """Test MaskInfo.from_cu_seqlens preserves cu_seqlens."""
        # New format: [start, end] per batch
        cu_q = jnp.array([0, 3, 0, 2])  # batch 0: 0-2, batch 1: 0-1
        cu_kv = jnp.array([0, 4, 0, 3])  # batch 0: 0-3, batch 1: 0-2

        mask_info = MaskInfo.from_cu_seqlens(cu_q, max_q_len=4, cu_seqlens_kv=cu_kv, max_kv_len=5)

        retrieved_cu_q, retrieved_cu_kv = mask_info.get_or_compute_qkv_cu_seqlens()

        assert jnp.array_equal(retrieved_cu_q, cu_q), f"cu_q mismatch: {retrieved_cu_q} vs {cu_q}"
        assert jnp.array_equal(retrieved_cu_kv, cu_kv), f"cu_kv mismatch: {retrieved_cu_kv} vs {cu_kv}"

    def test_self_attention_cu_seqlens_equal(self):
        """For self-attention, cu_q and cu_kv should be equal."""
        segment_ids = jnp.array([[1, 1, 1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)
        cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens()

        assert jnp.array_equal(cu_q, cu_kv), f"Self-attention cu_seqlens should match: {cu_q} vs {cu_kv}"

    def test_cross_attention_cu_seqlens_different(self):
        """For cross-attention, cu_q and cu_kv can differ."""
        # Q has 2 tokens with segment ID 1
        # KV has 4 tokens with segment ID 1
        q_seg = jnp.array([[1, 1, -1]])  # 2 tokens seg 1
        kv_seg = jnp.array([[1, 1, 1, 1, -1]])  # 4 tokens seg 1

        mask_info = MaskInfo.from_segments(q_seg, kv_seg)
        cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens(max_segments=4)

        # Cumulative format: [0, count_seg0, count_seg0+count_seg1, ...]
        # Segment 0 doesn't exist (0 tokens), segment 1 has the counts
        # With max_segments=4, output has size 5
        expected_cu_q = jnp.array([0, 0, 2, 2, 2])  # seg0=0, seg1=2, rest same
        expected_cu_kv = jnp.array([0, 0, 4, 4, 4])  # seg0=0, seg1=4, rest same

        assert jnp.array_equal(cu_q, expected_cu_q), f"cu_q: {cu_q}, expected {expected_cu_q}"
        assert jnp.array_equal(cu_kv, expected_cu_kv), f"cu_kv: {cu_kv}, expected {expected_cu_kv}"

    def test_q_lens_kv_lens_properties(self):
        """Test q_lens and kv_lens properties (computed from cu_seqlens)."""
        # Use segment IDs 0, 1, 2 to get proper cumulative offsets
        segment_ids = jnp.array([
            [0, 0, 0, 1, 1, -1],  # 3 tokens seg0, 2 tokens seg1
        ])
        mask_info = MaskInfo.from_segments(segment_ids)

        q_lens = mask_info.q_lens
        kv_lens = mask_info.kv_lens

        # cu_seqlens = [0, 3, 5] (cumulative: 0, 3 for seg0, 3+2=5 for seg1)
        # q_lens = diff([0, 3, 5]) = [3, 2]
        expected_lens = jnp.array([3, 2])
        assert jnp.array_equal(q_lens, expected_lens), f"q_lens: {q_lens}, expected {expected_lens}"
        assert jnp.array_equal(kv_lens, expected_lens), f"kv_lens: {kv_lens}, expected {expected_lens}"

    def test_caching_cu_seqlens(self):
        """Test that cu_seqlens are cached after first computation."""
        segment_ids = jnp.array([[1, 1, 1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)

        # First call computes
        cu_q1, cu_kv1 = mask_info.get_or_compute_qkv_cu_seqlens()

        # Second call should return cached values (same objects)
        cu_q2, cu_kv2 = mask_info.get_or_compute_qkv_cu_seqlens()

        assert cu_q1 is cu_q2, "cu_seqlens_q should be cached"
        assert cu_kv1 is cu_kv2, "cu_seqlens_kv should be cached"


class TestEdgeCases:
    """Edge case tests for position and cu_seqlens computation."""

    def test_empty_batch(self):
        """Test with batch size 0 (edge case)."""
        segment_ids = jnp.zeros((0, 4), dtype=jnp.int32)
        mask_info = MaskInfo.from_segments(segment_ids)
        
        # Should handle empty batch gracefully
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        assert q_pos.shape == (0, 4)
        assert kv_pos.shape == (0, 4)

    def test_sequence_length_1(self):
        """Test with sequence length 1."""
        # Use segment ID 0 for proper cumulative offset format
        segment_ids = jnp.array([[0], [0]])  # batch=2, seq=1, both use seg 0
        mask_info = MaskInfo.from_segments(segment_ids)

        q_pos, kv_pos = mask_info.get_or_compute_positions()
        expected = jnp.array([[0], [0]])
        assert jnp.array_equal(q_pos, expected)
        assert jnp.array_equal(kv_pos, expected)

        cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens(max_segments=2)
        # Cumulative format: segment 0 has 2 tokens total (1 from each batch element)
        # With max_segments=2, output has size 3: [0, 2, 2]
        expected_cu = jnp.array([0, 2, 2])
        assert jnp.array_equal(cu_q, expected_cu), f"cu_q: {cu_q}, expected {expected_cu}"

    def test_all_padding(self):
        """Test with all padding tokens."""
        segment_ids = jnp.array([[-1, -1, -1]])
        mask_info = MaskInfo.from_segments(segment_ids)

        q_pos, kv_pos = mask_info.get_or_compute_positions()
        expected_q = jnp.array([[-1, -1, -1]])
        pad_val = jnp.iinfo(jnp.int32).max
        expected_kv = jnp.array([[pad_val, pad_val, pad_val]])

        assert jnp.array_equal(q_pos, expected_q), f"q_pos: {q_pos}"
        assert jnp.array_equal(kv_pos, expected_kv), f"kv_pos: {kv_pos}"

        cu_q, cu_kv = mask_info.get_or_compute_qkv_cu_seqlens(max_segments=2)
        # All padding: no valid segment IDs, all counts are 0
        # With max_segments=2, output has size 3: [0, 0, 0]
        expected_cu = jnp.array([0, 0, 0])
        assert jnp.array_equal(cu_q, expected_cu), f"cu_q: {cu_q}"

    def test_large_segment_ids(self):
        """Test with large segment ID values."""
        segment_ids = jnp.array([[1000, 1000, 2000, 2000]])
        mask_info = MaskInfo.from_segments(segment_ids)
        
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        expected = jnp.array([[0, 1, 0, 1]])
        assert jnp.array_equal(q_pos, expected)

    def test_non_contiguous_segments(self):
        """Test with non-contiguous segment IDs (e.g., 1, 3, 5)."""
        segment_ids = jnp.array([[1, 1, 5, 5, 3, 3]])
        mask_info = MaskInfo.from_segments(segment_ids)
        
        q_pos, kv_pos = mask_info.get_or_compute_positions()
        expected = jnp.array([[0, 1, 0, 1, 0, 1]])
        assert jnp.array_equal(q_pos, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
