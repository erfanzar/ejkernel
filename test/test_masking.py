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


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.types.mask import (
    MaskInfo,
    mask_to_segment_ids,
    segment_ids_to_mask,
    segment_ids_to_qkv_masks,
)


def _row_equal(m: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(m[:, None, :] == m[None, :, :], axis=-1)


def _col_equal(m: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(m.T[:, None, :] == m.T[None, :, :], axis=-1)


def _assert_contiguous_and_ordered(ids: np.ndarray):
    valid = ids >= 0
    if not np.any(valid):
        return
    vals = np.unique(ids[valid])
    assert np.array_equal(vals, np.arange(vals.size)), f"IDs not contiguous from 0..G-1: {vals}"

    first_idx = []
    for g in range(vals.size):
        idxs = np.where(ids == g)[0]
        assert idxs.size > 0, f"Missing gid {g}"
        first_idx.append(idxs.min())
    assert np.all(np.diff(first_idx) > 0), f"First-occurrence indices not strictly increasing: {first_idx}"


def _verify_segmentation(mask_2d: jnp.ndarray, q_ids: jnp.ndarray, kv_ids: jnp.ndarray):
    """
    Verify for a single 2D mask:
    - Equal rows <=> equal q_ids for non-padded rows
    - Equal cols <=> equal kv_ids for non-padded cols
    - Padded rows/cols (all-zero) have id -1
    - IDs are contiguous and ordered by first occurrence
    """
    m = mask_2d.astype(bool)
    q_pad = ~jnp.any(m, axis=-1)
    kv_pad = ~jnp.any(m, axis=0)

    assert jnp.all(jnp.where(q_pad, q_ids == -1, True)), "Non -1 q_id found on padded row"
    assert jnp.all(jnp.where(~q_pad, q_ids >= 0, True)), "Negative q_id on non-padded row"
    assert jnp.all(jnp.where(kv_pad, kv_ids == -1, True)), "Non -1 kv_id found on padded col"
    assert jnp.all(jnp.where(~kv_pad, kv_ids >= 0, True)), "Negative kv_id on non-padded col"

    row_eq = _row_equal(m)
    qid_eq = q_ids[:, None] == q_ids[None, :]
    valid_pair_q = (~q_pad[:, None]) & (~q_pad[None, :])
    assert jnp.all(row_eq[valid_pair_q] == qid_eq[valid_pair_q]), "Row grouping mismatch"

    col_eq = _col_equal(m)
    kvid_eq = kv_ids[:, None] == kv_ids[None, :]
    valid_pair_k = (~kv_pad[:, None]) & (~kv_pad[None, :])
    assert jnp.all(col_eq[valid_pair_k] == kvid_eq[valid_pair_k]), "Column grouping mismatch"

    _assert_contiguous_and_ordered(np.array(q_ids))
    _assert_contiguous_and_ordered(np.array(kv_ids))


def _run_and_verify_2d(name: str, mask_2d: jnp.ndarray):
    q_ids, kv_ids = mask_to_segment_ids(mask_2d, per_head=False)
    _verify_segmentation(mask_2d, q_ids, kv_ids)
    print(f"  ✓ {name}: OK")


def _run_and_verify_3d(name: str, mask_3d: jnp.ndarray):
    q_ids, kv_ids = mask_to_segment_ids(mask_3d, per_head=False)
    B, Q, K = mask_3d.shape
    assert q_ids.shape == (B, Q) and kv_ids.shape == (B, K)
    for b in range(B):
        _verify_segmentation(mask_3d[b], q_ids[b], kv_ids[b])
    print(f"  ✓ {name}: OK")


def _run_and_verify_4d_per_head_false(name: str, mask_4d: jnp.ndarray):
    q_ids, kv_ids = mask_to_segment_ids(mask_4d, per_head=False)
    B, _, Q, K = mask_4d.shape
    assert q_ids.shape == (B, Q) and kv_ids.shape == (B, K)
    q_ids_ref, kv_ids_ref = mask_to_segment_ids(mask_4d[:, 0, :, :], per_head=False)
    assert jnp.array_equal(q_ids, q_ids_ref)
    assert jnp.array_equal(kv_ids, kv_ids_ref)
    for b in range(B):
        _verify_segmentation(mask_4d[b, 0], q_ids[b], kv_ids[b])
    print(f"  ✓ {name} (per_head=False uses head 0): OK")


def _run_and_verify_4d_per_head_true(name: str, mask_4d: jnp.ndarray):
    q_ids, kv_ids = mask_to_segment_ids(mask_4d, per_head=True)
    B, H, Q, K = mask_4d.shape
    assert q_ids.shape == (B, H, Q) and kv_ids.shape == (B, H, K)
    for b in range(B):
        for h in range(H):
            _verify_segmentation(mask_4d[b, h], q_ids[b, h], kv_ids[b, h])
    print(f"  ✓ {name} (per_head=True): OK")


def test_segment_ids_to_mask_conversions():
    """Test segment_ids_to_mask for self- and cross-attention inputs."""
    print("\nTesting segment_ids_to_mask() conversions...")

    segment_ids = jnp.array([[1, 1, 2, -1]], dtype=jnp.int32)
    seg_arr = np.array(jax.device_get(segment_ids))
    attn = jax.device_get(segment_ids_to_mask(segment_ids))
    expected_self = (seg_arr[:, :, None] == seg_arr[:, None, :]) & (
        (seg_arr >= 0)[:, :, None] & (seg_arr >= 0)[:, None, :]
    )
    assert attn.shape == (1, 1, 4, 4)
    assert np.array_equal(attn[:, 0], expected_self)

    q_mask, kv_mask, attn_separate = segment_ids_to_mask(segment_ids, return_separate_masks=True)
    assert np.array_equal(jax.device_get(q_mask), seg_arr >= 0)
    assert np.array_equal(jax.device_get(kv_mask), seg_arr >= 0)
    assert np.array_equal(jax.device_get(attn_separate)[:, 0], expected_self)

    q_segments = jnp.array([[1, 2, 2, -1]], dtype=jnp.int32)
    kv_segments = jnp.array([[1, 1, 2, 2, 3]], dtype=jnp.int32)
    q_arr = np.array(jax.device_get(q_segments))
    kv_arr = np.array(jax.device_get(kv_segments))

    mask_float = segment_ids_to_mask((q_segments, kv_segments), dtype=jnp.float32)
    mask_cross = jax.device_get(mask_float[:, 0] > 0.5)
    expected_cross = (q_arr[:, :, None] == kv_arr[:, None, :]) & (
        (q_arr >= 0)[:, :, None] & (kv_arr >= 0)[:, None, :]
    )
    assert mask_float.dtype == jnp.float32
    assert mask_float.shape == (1, 1, 4, 5)
    assert np.array_equal(mask_cross, expected_cross)

    print("  ✓ segment_ids_to_mask: OK")


def test_segment_ids_to_qkv_masks_cross_attention():
    """Test segment_ids_to_qkv_masks returns consistent masks for cross-attention."""
    print("\nTesting segment_ids_to_qkv_masks() for cross-attention inputs...")

    q_segments = jnp.array([[1, 2, 2, -1]], dtype=jnp.int32)
    kv_segments = jnp.array([[1, 1, 2, 2, 3]], dtype=jnp.int32)
    q_arr = np.array(jax.device_get(q_segments))
    kv_arr = np.array(jax.device_get(kv_segments))

    q_mask, kv_mask, attn = segment_ids_to_qkv_masks(q_segments, kv_segments)
    expected_q = q_arr >= 0
    expected_kv = kv_arr >= 0
    expected_attn = (q_arr[:, :, None] == kv_arr[:, None, :]) & (
        expected_q[:, :, None] & expected_kv[:, None, :]
    )

    assert np.array_equal(jax.device_get(q_mask), expected_q)
    assert np.array_equal(jax.device_get(kv_mask), expected_kv)
    assert np.array_equal(jax.device_get(attn)[:, 0], expected_attn)

    print("  ✓ segment_ids_to_qkv_masks: OK")


def test_maskinfo_apply_kv_lengths_updates_masks_and_segments():
    """Test apply_kv_lengths masking, slicing, and segment updates."""
    print("\nTesting MaskInfo.apply_kv_lengths() slicing and masking...")

    q_segments = jnp.array(
        [[1, 1, 2, 2, 3], [1, 2, 2, 3, 3]],
        dtype=jnp.int32,
    )
    mask_info = MaskInfo.from_segments(q_segments)

    kv_lengths = jnp.array([4, 3], dtype=jnp.int32)
    end_index = jnp.array([5, 4], dtype=jnp.int32)

    updated = mask_info.apply_kv_lengths(kv_lengths=kv_lengths, q_len=2, end_index=end_index)

    assert updated.attention_mask.shape == (2, 1, 2, 5)
    attn = jax.device_get(updated.attention_mask.astype(bool))
    for batch, length in enumerate(kv_lengths):
        assert not attn[batch, 0, :, length:].any(), "KV positions beyond length must be masked"

    expected_q_segments = np.array([[2, 3], [2, 3]], dtype=np.int32)
    expected_kv_segments = np.array([[1, 1, 2, 2, -1], [1, 2, 2, -1, -1]], dtype=np.int32)

    assert np.array_equal(jax.device_get(updated.q_segment_ids), expected_q_segments)
    assert np.array_equal(jax.device_get(updated.kv_segment_ids), expected_kv_segments)

    print("  ✓ apply_kv_lengths masking: OK")


def test_maskinfo_apply_kv_lengths_requires_indices():
    """Test apply_kv_lengths enforces start/end indices when slicing queries."""
    print("\nTesting MaskInfo.apply_kv_lengths() input validation...")

    q_segments = jnp.array([[1, 1, 1, 1]], dtype=jnp.int32)
    mask_info = MaskInfo.from_segments(q_segments)

    with pytest.raises(ValueError):
        mask_info.apply_kv_lengths(kv_lengths=jnp.array([3]), q_len=2)

    print("  ✓ apply_kv_lengths validation: OK")


def test_maskinfo_from_segments():
    """Test MaskInfo.from_segments() factory method."""
    print("\nTesting MaskInfo.from_segments()...")

    q_seg = jnp.array([[1, 1, 2, 2, -1]])
    mask_info = MaskInfo.from_segments(q_seg)

    assert mask_info.q_segment_ids.shape == (1, 5)
    assert mask_info.kv_segment_ids.shape == (1, 5)
    assert mask_info.attention_mask.shape == (1, 1, 5, 5)
    assert jnp.array_equal(mask_info.q_segment_ids, q_seg)
    assert jnp.array_equal(mask_info.kv_segment_ids, q_seg)

    attn = mask_info.attention_mask[0, 0]

    assert jnp.all(attn[0:2, 0:2])

    assert jnp.all(attn[2:4, 2:4])

    assert not jnp.any(attn[0:2, 2:4])
    assert not jnp.any(attn[2:4, 0:2])

    assert not jnp.any(attn[4, :])
    assert not jnp.any(attn[:, 4])

    print("  ✓ from_segments: OK")


def _mk_base_mask(B=1, Q=16, K=16, left=4, right=2):
    """Full mask -> sliding window to get a deterministic base."""
    m = MaskInfo.from_random(batch_size=B, q_len=Q, kv_len=K, sparsity=0.0, seed=0)
    m = m.apply_sliding_window((left, right))
    base = jax.device_get(m.get_or_compute_attention_mask(dtype=jnp.bool_))
    return m, base  # m holds the stored attention_mask


def test_maskinfo_from_attention_mask_3d():
    """Test MaskInfo.from_attention_mask() with 3D pairwise mask."""
    print("\nTesting MaskInfo.from_attention_mask() with 3D mask...")

    # 3D input gets converted to 4D internally
    mask = jnp.array([[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]])
    mask_info = MaskInfo.from_attention_mask(mask)

    assert mask_info.q_segment_ids.shape == (1, 4)
    assert mask_info.kv_segment_ids.shape == (1, 4)
    # After conversion, the mask is 4D (batch, 1, q, kv)
    assert mask_info.attention_mask.shape == (1, 1, 4, 4)

    assert mask_info.q_segment_ids[0, 0] == mask_info.q_segment_ids[0, 1]
    assert mask_info.q_segment_ids[0, 2] == mask_info.q_segment_ids[0, 3]
    assert mask_info.q_segment_ids[0, 0] != mask_info.q_segment_ids[0, 2]

    print("  ✓ from_attention_mask (3D): OK")


def test_maskinfo_properties():
    """Test MaskInfo properties (batch_size, q_len, kv_len, shape)."""
    print("\nTesting MaskInfo properties...")

    segment_ids = jnp.array([[1, 1, 2, -1], [1, 2, 2, -1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    assert mask_info.batch_size == 2
    assert mask_info.q_len == 4
    assert mask_info.kv_len == 4
    assert mask_info.shape == (2, 4, 4)

    print("  ✓ Properties: OK")


def test_maskinfo_get_qkv_masks():
    """Test MaskInfo.get_qkv_masks()."""
    print("\nTesting MaskInfo.get_qkv_masks()...")

    segment_ids = jnp.array([[1, 1, 2, -1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    q_mask, kv_mask, attn_mask = mask_info.get_qkv_masks()

    assert q_mask.shape == (1, 4)
    assert kv_mask.shape == (1, 4)
    assert attn_mask.shape == (1, 1, 4, 4)

    expected_valid = jnp.array([[True, True, True, False]])
    assert jnp.array_equal(q_mask, expected_valid)
    assert jnp.array_equal(kv_mask, expected_valid)

    print("  ✓ get_qkv_masks: OK")


def test_maskinfo_is_self_attention():
    """Test MaskInfo.is_self_attention()."""
    print("\nTesting MaskInfo.is_self_attention()...")

    q_seg = jnp.array([[1, 1, 2]])
    mask_info = MaskInfo.from_segments(q_seg)
    assert mask_info.is_self_attention()

    q_seg = jnp.array([[1, 2]])
    kv_seg = jnp.array([[1, 1, 2, 2]])
    mask_info = MaskInfo.from_segments(q_seg, kv_seg)
    assert not mask_info.is_self_attention()

    print("  ✓ is_self_attention: OK")


def test_maskinfo_to_dtype():
    """Test MaskInfo.to_dtype()."""
    print("\nTesting MaskInfo.to_dtype()...")

    segment_ids = jnp.array([[1, 1, 2]])
    mask_info = MaskInfo.from_segments(segment_ids)

    mask_info_float = mask_info.to_dtype(jnp.float32)
    assert mask_info_float.attention_mask.dtype == jnp.float32

    assert mask_info.attention_mask.dtype == jnp.bool_

    print("  ✓ to_dtype: OK")


def test_maskinfo_apply_causal():
    """Test MaskInfo.apply_causal()."""
    print("\nTesting MaskInfo.apply_causal()...")

    segment_ids = jnp.array([[1, 1, 1, 1, 1]])
    mask_info = MaskInfo.from_segments(segment_ids)
    causal_mask_info = mask_info.apply_causal()

    attn = causal_mask_info.attention_mask[0, 0]
    for i in range(5):
        for j in range(5):
            if j <= i:
                assert attn[i, j], f"Expected causal mask at [{i},{j}]"
            else:
                assert not attn[i, j], f"Expected no attention at [{i},{j}]"

    causal_mask_info_offset = mask_info.apply_causal(offset=1)
    attn_offset = causal_mask_info_offset.attention_mask[0, 0]

    assert attn_offset[0, 1]

    print("  ✓ apply_causal: OK")


def test_maskinfo_apply_sliding_window():
    """Test MaskInfo.apply_sliding_window()."""
    print("\nTesting MaskInfo.apply_sliding_window()...")

    segment_ids = jnp.array([[1, 1, 1, 1, 1, 1, 1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    windowed = mask_info.apply_sliding_window(window_size=(1, 1))
    attn = windowed.attention_mask[0, 0]

    assert not attn[3, 0]
    assert not attn[3, 1]
    assert attn[3, 2]
    assert attn[3, 3]
    assert attn[3, 4]
    assert not attn[3, 5]
    assert not attn[3, 6]

    windowed_onesided = mask_info.apply_sliding_window(window_size=(None, 2))
    attn_onesided = windowed_onesided.attention_mask[0, 0]

    assert attn_onesided[3, 0]
    assert attn_onesided[3, 5]
    assert not attn_onesided[3, 6]

    print("  ✓ apply_sliding_window: OK")


def test_maskinfo_apply_chunked():
    """Test MaskInfo.apply_chunked()."""
    print("\nTesting MaskInfo.apply_chunked()...")

    segment_ids = jnp.array([[1, 1, 1, 1, 1, 1]])
    mask_info = MaskInfo.from_segments(segment_ids)
    chunked = mask_info.apply_chunked(chunk_size=3)

    q_seg = chunked.q_segment_ids[0]
    assert q_seg[0] == q_seg[1] == q_seg[2]
    assert q_seg[3] == q_seg[4] == q_seg[5]
    assert q_seg[0] != q_seg[3]

    attn = chunked.attention_mask[0, 0]

    assert attn[0, 0]
    assert not attn[0, 1]
    assert attn[1, 0]
    assert attn[1, 1]

    assert not attn[0, 3]
    assert not attn[3, 0]

    assert attn[4, 3]
    assert not attn[3, 4]

    print("  ✓ apply_chunked: OK")


def test_maskinfo_composable():
    """Test that mask operations are composable."""
    print("\nTesting composable mask operations...")

    segment_ids = jnp.array([[1, 1, 1, 1, 1, 1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    composed = mask_info.apply_causal().apply_sliding_window(window_size=2)

    attn = composed.attention_mask[0, 0]

    assert not attn[4, 0]
    assert not attn[4, 1]
    assert attn[4, 2]
    assert attn[4, 3]
    assert attn[4, 4]
    assert not attn[4, 5]

    print("  ✓ Composable operations: OK")


def test_maskinfo_with_padding():
    """Test that mask operations preserve padding (-1)."""
    print("\nTesting mask operations with padding...")

    segment_ids = jnp.array([[1, 1, 2, 2, -1, -1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    causal = mask_info.apply_causal()
    assert jnp.all(causal.q_segment_ids[0, 4:] == -1)
    assert jnp.all(causal.kv_segment_ids[0, 4:] == -1)

    windowed = mask_info.apply_sliding_window(window_size=1)
    assert jnp.all(windowed.q_segment_ids[0, 4:] == -1)
    assert jnp.all(windowed.kv_segment_ids[0, 4:] == -1)

    chunked = mask_info.apply_chunked(chunk_size=2)
    assert jnp.all(chunked.q_segment_ids[0, 4:] == -1)
    assert jnp.all(chunked.kv_segment_ids[0, 4:] == -1)

    print("  ✓ Padding preservation: OK")


def test_maskinfo_from_random():
    """Test MaskInfo.from_random()."""
    print("\nTesting MaskInfo.from_random()...")

    # Random masks don't compute segment IDs (to avoid O(n^2) memory usage)
    mask_info = MaskInfo.from_random(batch_size=2, q_len=10, seed=42)
    assert mask_info.attention_mask.shape == (2, 1, 10, 10)
    assert mask_info._q_segment_ids is None
    assert mask_info._kv_segment_ids is None
    q_ids, kv_ids = mask_info.get_or_compute_segment_ids()
    assert q_ids.shape == (2, 10)
    assert kv_ids.shape == (2, 10)
    assert mask_info._q_segment_ids is not None
    assert mask_info._kv_segment_ids is not None
    assert mask_info.is_self_attention()

    mask_info_cross = MaskInfo.from_random(batch_size=1, q_len=8, kv_len=12, seed=0)
    assert mask_info_cross.attention_mask.shape == (1, 1, 8, 12)
    assert not mask_info_cross.is_self_attention()

    # Sparsity edge cases
    mask_full = MaskInfo.from_random(batch_size=1, q_len=5, sparsity=0.0, seed=0)
    assert jnp.all(mask_full.attention_mask)

    mask_empty = MaskInfo.from_random(batch_size=1, q_len=5, sparsity=1.0, seed=0)
    assert not jnp.any(mask_empty.attention_mask)

    # Seed reproducibility
    mask1 = MaskInfo.from_random(batch_size=1, q_len=20, sparsity=0.5, seed=123)
    mask2 = MaskInfo.from_random(batch_size=1, q_len=20, sparsity=0.5, seed=123)
    assert jnp.array_equal(mask1.attention_mask, mask2.attention_mask)

    mask3 = MaskInfo.from_random(batch_size=1, q_len=20, sparsity=0.5, seed=456)
    assert not jnp.array_equal(mask1.attention_mask, mask3.attention_mask)

    print("  ✓ from_random: OK")


def test_maskinfo_get_or_compute_positions():
    """Test MaskInfo.get_or_compute_positions()."""
    print("\nTesting MaskInfo.get_or_compute_positions()...")

    segment_ids = jnp.array([[1, 1, 2, 2, -1]])
    mask_info = MaskInfo.from_segments(segment_ids)

    q_pos, kv_pos = mask_info.get_or_compute_positions()

    assert q_pos is not None
    assert kv_pos is not None
    assert q_pos.shape == (1, 5)
    assert kv_pos.shape == (1, 5)

    expected_q = jnp.array([[0, 1, 0, 1, -1]], dtype=jnp.int32)
    expected_kv = jnp.array([[0, 1, 0, 1, jnp.iinfo(jnp.int32).max]], dtype=jnp.int32)
    assert jnp.array_equal(q_pos, expected_q)
    assert jnp.array_equal(kv_pos, expected_kv)

    q_seg = jnp.array([[1, 2]])
    kv_seg = jnp.array([[1, 1, 2, 2]])
    mask_info_cross = MaskInfo.from_segments(q_seg, kv_seg)

    q_pos_cross, kv_pos_cross = mask_info_cross.get_or_compute_positions()
    assert q_pos_cross.shape == (1, 2)
    assert kv_pos_cross.shape == (1, 4)
    assert jnp.array_equal(q_pos_cross, jnp.array([[0, 0]], dtype=jnp.int32))
    assert jnp.array_equal(kv_pos_cross, jnp.array([[0, 1, 0, 1]], dtype=jnp.int32))

    print("  ✓ get_or_compute_positions: OK")


def test_maskinfo_positions_from_factory():
    """Test that positions can be provided to factory methods."""
    print("\nTesting MaskInfo factory methods with positions...")

    segment_ids = jnp.array([[1, 1, 2]])
    custom_q_pos = jnp.array([[0, 1, 2]], dtype=jnp.int32)
    custom_kv_pos = jnp.array([[0, 1, 2]], dtype=jnp.int32)

    mask_info = MaskInfo.from_segments(segment_ids, q_positions=custom_q_pos, kv_positions=custom_kv_pos)

    assert mask_info.q_positions is not None
    assert mask_info.kv_positions is not None
    assert jnp.array_equal(mask_info.q_positions, custom_q_pos)
    assert jnp.array_equal(mask_info.kv_positions, custom_kv_pos)

    attn_mask = jnp.ones((1, 3, 3), dtype=jnp.bool_)
    mask_info_attn = MaskInfo.from_attention_mask(attn_mask, q_positions=custom_q_pos, kv_positions=custom_kv_pos)

    assert jnp.array_equal(mask_info_attn.q_positions, custom_q_pos)
    assert jnp.array_equal(mask_info_attn.kv_positions, custom_kv_pos)

    mask_info_random = MaskInfo.from_random(
        batch_size=1, q_len=3, seed=0, q_positions=custom_q_pos, kv_positions=custom_kv_pos
    )

    assert jnp.array_equal(mask_info_random.q_positions, custom_q_pos)
    assert jnp.array_equal(mask_info_random.kv_positions, custom_kv_pos)

    print("  ✓ positions from factory methods: OK")


def test_maskinfo_positions_preservation():
    """Test that positions are preserved through transformations."""
    print("\nTesting position preservation through transformations...")

    segment_ids = jnp.array([[1, 1, 1, 1, 1]])
    custom_q_pos = jnp.array([[10, 11, 12, 13, 14]], dtype=jnp.int32)
    custom_kv_pos = jnp.array([[10, 11, 12, 13, 14]], dtype=jnp.int32)

    mask_info = MaskInfo.from_segments(segment_ids, q_positions=custom_q_pos, kv_positions=custom_kv_pos)

    causal = mask_info.apply_causal()
    assert causal.q_positions is not None
    assert causal.kv_positions is not None
    assert jnp.array_equal(causal.q_positions, custom_q_pos)
    assert jnp.array_equal(causal.kv_positions, custom_kv_pos)

    windowed = mask_info.apply_sliding_window(window_size=2)
    assert windowed.q_positions is not None
    assert windowed.kv_positions is not None
    assert jnp.array_equal(windowed.q_positions, custom_q_pos)
    assert jnp.array_equal(windowed.kv_positions, custom_kv_pos)

    chunked = mask_info.apply_chunked(chunk_size=2)
    assert chunked.q_positions is not None
    assert chunked.kv_positions is not None
    assert jnp.array_equal(chunked.q_positions, custom_q_pos)
    assert jnp.array_equal(chunked.kv_positions, custom_kv_pos)

    as_float = mask_info.to_dtype(jnp.float32)
    assert as_float.q_positions is not None
    assert as_float.kv_positions is not None
    assert jnp.array_equal(as_float.q_positions, custom_q_pos)
    assert jnp.array_equal(as_float.kv_positions, custom_kv_pos)

    print("  ✓ positions preservation: OK")


def test_maskinfo_positions_cross_attention():
    """Test positions with cross-attention (different q and kv lengths)."""
    print("\nTesting positions with cross-attention...")

    q_seg = jnp.array([[1, 2, 2]])
    kv_seg = jnp.array([[1, 1, 2, 2, 3]])

    q_pos = jnp.array([[0, 0, 1]], dtype=jnp.int32)
    kv_pos = jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32)

    mask_info = MaskInfo.from_segments(q_seg, kv_seg, q_positions=q_pos, kv_positions=kv_pos)

    assert mask_info.q_len == 3
    assert mask_info.kv_len == 5
    assert jnp.array_equal(mask_info.q_positions, q_pos)
    assert jnp.array_equal(mask_info.kv_positions, kv_pos)

    causal = mask_info.apply_causal()
    assert jnp.array_equal(causal.q_positions, q_pos)
    assert jnp.array_equal(causal.kv_positions, kv_pos)

    print("  ✓ cross-attention positions: OK")


def test_maskinfo_positions_with_batches():
    """Test positions with multiple batches."""
    print("\nTesting positions with batched data...")

    segment_ids = jnp.array([[1, 1, 2, 2], [1, 2, 2, -1]])

    mask_info = MaskInfo.from_segments(segment_ids)
    q_pos, kv_pos = mask_info.get_or_compute_positions()

    assert q_pos.shape == (2, 4)
    assert kv_pos.shape == (2, 4)

    expected_q = jnp.array([[0, 1, 0, 1], [0, 0, 1, -1]], dtype=jnp.int32)
    expected_kv = jnp.array(
        [[0, 1, 0, 1], [0, 0, 1, jnp.iinfo(jnp.int32).max]],
        dtype=jnp.int32,
    )
    assert jnp.array_equal(q_pos, expected_q)
    assert jnp.array_equal(kv_pos, expected_kv)

    custom_pos = jnp.array([[0, 1, 2, 3], [10, 11, 12, 13]], dtype=jnp.int32)

    mask_info_custom = MaskInfo.from_segments(segment_ids, q_positions=custom_pos, kv_positions=custom_pos)

    retrieved_q, retrieved_kv = mask_info_custom.get_or_compute_positions()
    assert jnp.array_equal(retrieved_q, custom_pos)
    assert jnp.array_equal(retrieved_kv, custom_pos)

    print("  ✓ batched positions: OK")


def test_maskinfo_segment_ids_as_source_of_truth():
    """Test that segment IDs are always used as source of truth for attention mask."""
    print("\nTesting segment IDs as source of truth...")

    # Create MaskInfo with both segment IDs and attention mask
    q_seg = jnp.array([[1, 1, 2, 2]])
    kv_seg = jnp.array([[1, 1, 2, 2]])

    # Create a "wrong" attention mask that doesn't match the segment IDs (4D)
    wrong_mask = jnp.ones((1, 1, 4, 4), dtype=jnp.bool_)  # All True

    mask_info = MaskInfo.from_segments(q_seg, kv_seg)
    mask_with_wrong = mask_info.replace(attention_mask=wrong_mask)
    assert jnp.array_equal(mask_with_wrong.get_or_compute_attention_mask(), wrong_mask)

    mask_with_wrong._attention_mask = None
    computed_mask = mask_with_wrong.get_or_compute_attention_mask()
    expected_mask = MaskInfo.from_segments(q_seg, kv_seg).attention_mask

    # The computed mask should reflect segment structure, not the "wrong" mask
    assert not computed_mask[0, 0, 0, 2], "Token 0 (seg 1) should not attend to token 2 (seg 2)"
    assert not computed_mask[0, 0, 1, 3], "Token 1 (seg 1) should not attend to token 3 (seg 2)"
    assert computed_mask[0, 0, 0, 1], "Token 0 should attend to token 1 (same segment)"
    assert computed_mask[0, 0, 2, 3], "Token 2 should attend to token 3 (same segment)"
    assert jnp.array_equal(computed_mask, expected_mask)

    print("  ✓ segment IDs as source of truth: OK")


def _mk_base_mask(B=1, Q=16, K=16, left=4, right=2):
    """Full mask -> sliding window to get a deterministic base."""
    m = MaskInfo.from_random(batch_size=B, q_len=Q, kv_len=K, sparsity=0.0, seed=0)
    m = m.apply_sliding_window((left, right))
    base = jax.device_get(m.get_or_compute_attention_mask(dtype=jnp.bool_))
    return m, base  # m holds the stored attention_mask


def _eq_mask(q_types, kv_types, zero_policy="q"):
    """Build expected equality mask (B,1,Q,K) honoring zero_policy."""
    B, _Q = q_types.shape
    B2, _K = kv_types.shape
    assert B == B2
    eq2d = q_types[:, :, None] == kv_types[:, None, :]
    if zero_policy in ("q", "both"):
        eq2d = eq2d & (q_types != 0)[:, :, None]
    if zero_policy in ("kv", "both"):
        eq2d = eq2d & (kv_types != 0)[:, None, :]
    return eq2d[:, None, :, :].astype(bool)


def test_union_adds_edges_outside_window():
    B, Q, K = 1, 16, 16
    m, base = _mk_base_mask(B, Q, K, left=4, right=2)

    # Make equality true at a location outside the window (row 0, col 10)
    q_types = jnp.ones((B, Q), dtype=jnp.int32)
    kv_types = jnp.full((B, K), 2, dtype=jnp.int32)
    q_types = q_types.at[0, 0].set(5)
    kv_types = kv_types.at[0, 10].set(5)

    eq4d = _eq_mask(q_types, kv_types, zero_policy="none")
    expected = np.logical_or(base, jax.device_get(eq4d))

    m2 = m.apply_token_type_ids((q_types, kv_types), combine="union", zero_policy="none", update_segment_ids=False)
    actual = jax.device_get(m2.get_or_compute_attention_mask(dtype=jnp.bool_))

    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected)
    # Spot-check the outside-window addition
    assert expected[0, 0, 0, 10] and not base[0, 0, 0, 10]


@pytest.mark.parametrize("zero_policy", ["q", "kv", "both", "none"])
def test_zero_policy_effect(zero_policy):
    B, Q, K = 1, 12, 12
    m, base = _mk_base_mask(B, Q, K, left=3, right=2)

    # Create multiple matches; include zeros based on policy to verify disabling.
    q_types = jnp.array([[1, 0, 2, 2, 0, 3, 3, 1, 0, 4, 4, 5]], dtype=jnp.int32)
    kv_types = jnp.array([[1, 2, 2, 0, 3, 3, 1, 4, 0, 4, 5, 5]], dtype=jnp.int32)

    eq4d = _eq_mask(q_types, kv_types, zero_policy=zero_policy)

    # Use union to make it easy to observe effects
    m2 = m.apply_token_type_ids((q_types, kv_types), combine="union", zero_policy=zero_policy, update_segment_ids=False)
    actual = jax.device_get(m2.get_or_compute_attention_mask(dtype=jnp.bool_))
    expected = np.logical_or(base, jax.device_get(eq4d))

    assert np.array_equal(actual, expected)

    # Sanity: under zero_policy='q', row with q==0 should not gain any new edges
    if zero_policy in ("q", "both"):
        rows_with_zero = np.where(jax.device_get(q_types[0]) == 0)[0]
        for r in rows_with_zero:
            # Check any outside-window gains are absent
            new_edges = np.logical_and(actual[0, 0, r], np.logical_not(base[0, 0, r]))
            assert not new_edges.any()


def test_cross_attention_union_shapes_and_effect():
    B, Q, K = 1, 12, 20
    m, base = _mk_base_mask(B, Q, K, left=3, right=2)

    q_types = jnp.ones((B, Q), dtype=jnp.int32)
    kv_types = jnp.full((B, K), 2, dtype=jnp.int32)
    # One match outside the window for row 0
    q_types = q_types.at[0, 0].set(9)
    kv_types = kv_types.at[0, 15].set(9)

    eq4d = _eq_mask(q_types, kv_types, zero_policy="none")
    expected = np.logical_or(base, jax.device_get(eq4d))

    m2 = m.apply_token_type_ids((q_types, kv_types), combine="union", zero_policy="none", update_segment_ids=False)
    actual = jax.device_get(m2.get_or_compute_attention_mask(dtype=jnp.bool_))
    assert actual.shape == (B, 1, Q, K)
    assert np.array_equal(actual, expected)
    assert actual[0, 0, 0, 15] and not base[0, 0, 0, 15]


def test_visualize_smoke():
    # Just ensure visualize returns a string and doesn't crash.
    B, Q, K = 1, 32, 32
    m, _ = _mk_base_mask(B, Q, K, left=8, right=4)
    s = m.visualize(block_size=4, show_segments=True, return_str=True)
    assert isinstance(s, str) and "Attention mask" in s


if __name__ == "__main__":
    pytest.main([__file__])
