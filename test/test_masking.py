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

from ejkernel.types.mask import MaskInfo, mask_to_segment_ids


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


def test_suite():
    print("\nRunning tests for mask_to_segment_ids...\n")

    N = 8
    m_full = jnp.ones((N, N), dtype=bool)
    _run_and_verify_2d("Full attention (square)", m_full)

    m_block = jnp.zeros((N, N), dtype=bool)
    m_block = m_block.at[: N // 2, : N // 2].set(True)
    m_block = m_block.at[N // 2 :, N // 2 :].set(True)
    _run_and_verify_2d("Block diagonal (square)", m_block)

    Q, K = 7, 10
    m_causal = jnp.tril(jnp.ones((Q, K), dtype=bool), k=0)
    _run_and_verify_2d("Causal (rectangular Q<K)", m_causal)

    Q, K, w = 8, 9, 3
    m_win = jnp.zeros((Q, K), dtype=bool)
    for i in range(Q):
        start = max(0, i - w // 2)
        end = min(K, i + w // 2 + 1)
        m_win = m_win.at[i, start:end].set(True)
    _run_and_verify_2d(f"Local window w={w} (rectangular)", m_win)

    N = 8
    i = jnp.arange(N)[:, None]
    j = jnp.arange(N)[None, :]
    m_parity = (i % 2) == (j % 2)
    _run_and_verify_2d("Strided parity (square)", m_parity)

    Q, K = 6, 7
    m_pad = jnp.zeros((Q, K), dtype=bool)

    m_pad = m_pad.at[0, 1:4].set(True)
    m_pad = m_pad.at[2, 2:6].set(True)

    m_pad = m_pad.at[3, 5].set(True)

    _run_and_verify_2d("Padding rows/cols (rectangular)", m_pad)

    B, Q, K = 3, 6, 7
    key = jax.random.PRNGKey(0)
    m3 = jax.random.bernoulli(key, p=0.5, shape=(B, Q, K))

    m3 = m3.at[0, 1, :].set(False)
    m3 = m3.at[0, :, 0].set(False)
    _run_and_verify_3d("Batched (B,Q,K)", m3)

    B, H, Q, K = 2, 3, 6, 7
    key = jax.random.PRNGKey(42)
    m4 = jax.random.bernoulli(key, p=0.5, shape=(B, H, Q, K))

    m4 = m4.at[0, 0, 2, :].set(False)
    m4 = m4.at[0, 0, :, 1].set(False)
    _run_and_verify_4d_per_head_false("4D (B,H,Q,K)", m4)

    _run_and_verify_4d_per_head_true("4D (B,H,Q,K)", m4)

    N = 8
    m_bool = jnp.arange(N)[:, None] <= jnp.arange(N)[None, :]
    m_int = m_bool.astype(jnp.int32)
    m_float = m_bool.astype(jnp.float32)
    _run_and_verify_2d("Dtype int", m_int)
    _run_and_verify_2d("Dtype float", m_float)


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


def test_maskinfo_from_attention_mask_2d():
    """Test MaskInfo.from_attention_mask() with 2D padding mask."""
    print("\nTesting MaskInfo.from_attention_mask() with 2D padding mask...")

    padding_mask = jnp.array([[1, 1, 1, 0, 0]])
    mask_info = MaskInfo.from_attention_mask(padding_mask)

    expected_seg = jnp.array([[1, 1, 1, -1, -1]])
    assert jnp.array_equal(mask_info.q_segment_ids, expected_seg)
    assert jnp.array_equal(mask_info.kv_segment_ids, expected_seg)

    assert mask_info.attention_mask.shape == (1, 1, 5, 5)

    attn = mask_info.attention_mask[0, 0]
    assert jnp.all(attn[0:3, 0:3])

    assert not jnp.any(attn[3:, :])
    assert not jnp.any(attn[:, 3:])

    print("  ✓ from_attention_mask (2D): OK")


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
    assert mask_info.q_segment_ids is None, "Random masks should not have segment IDs"
    assert mask_info.kv_segment_ids is None, "Random masks should not have segment IDs"
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

    expected_pos = jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32)
    assert jnp.array_equal(q_pos, expected_pos)
    assert jnp.array_equal(kv_pos, expected_pos)

    q_seg = jnp.array([[1, 2]])
    kv_seg = jnp.array([[1, 1, 2, 2]])
    mask_info_cross = MaskInfo.from_segments(q_seg, kv_seg)

    q_pos_cross, kv_pos_cross = mask_info_cross.get_or_compute_positions()
    assert q_pos_cross.shape == (1, 2)
    assert kv_pos_cross.shape == (1, 4)
    assert jnp.array_equal(q_pos_cross, jnp.array([[0, 1]], dtype=jnp.int32))
    assert jnp.array_equal(kv_pos_cross, jnp.array([[0, 1, 2, 3]], dtype=jnp.int32))

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

    expected = jnp.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=jnp.int32)
    assert jnp.array_equal(q_pos, expected)
    assert jnp.array_equal(kv_pos, expected)

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

    mask_info = MaskInfo(
        attention_mask=wrong_mask,
        q_segment_ids=q_seg,
        kv_segment_ids=kv_seg
    )

    # get_or_compute_attention_mask should ALWAYS regenerate from segment IDs
    computed_mask = mask_info.get_or_compute_attention_mask()

    # The computed mask should reflect segment structure, not the "wrong" mask
    # Tokens [0,1] are in segment 1, tokens [2,3] are in segment 2
    # So [0,1] should NOT attend to [2,3]
    assert not computed_mask[0, 0, 0, 2], "Token 0 (seg 1) should not attend to token 2 (seg 2)"
    assert not computed_mask[0, 0, 1, 3], "Token 1 (seg 1) should not attend to token 3 (seg 2)"
    assert computed_mask[0, 0, 0, 1], "Token 0 should attend to token 1 (same segment)"
    assert computed_mask[0, 0, 2, 3], "Token 2 should attend to token 3 (same segment)"

    # Verify it's not just using the stored wrong_mask
    assert not jnp.array_equal(computed_mask, wrong_mask), \
        "Should regenerate mask from segment IDs, not use stored mask"

    print("  ✓ segment IDs as source of truth: OK")


if __name__ == "__main__":
    pytest.main([__file__])
