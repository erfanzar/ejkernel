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

from ejkernel.xla_utils import mask_to_segment_ids


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
    B, _H, Q, K = mask_4d.shape
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


if __name__ == "__main__":
    test_suite()
