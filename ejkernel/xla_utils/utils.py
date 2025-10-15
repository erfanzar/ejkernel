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
from jax import Array
from jaxtyping import Bool, DTypeLike, Int


def cdiv(a: Int[Array, "..."], b: int) -> Int[Array, "..."]:
    """Computes ceiling division for integers in a JAX-compatible way."""
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "num_seqs"]:
    """
    Calculates the lengths of individual sequences from cumulative sequence lengths.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths (e.g., [0, len1, len1+len2, ...]).

    Returns:
        A 1D array of sequence lengths (e.g., [len1, len2, ...]).
    """
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_lens_from_mask(mask: Bool[Array, "batch seq_len"]) -> Int[Array, "batch"]:
    """
    Calculates the length of each sequence from a boolean attention mask.

    Args:
        mask: A 2D boolean attention mask (batch_size, seq_len).

    Returns:
        A 1D array of sequence lengths with dtype int32.
    """
    return mask.sum(axis=-1, dtype=jnp.int32)


def prepare_cu_seqlens_from_mask(
    mask: Bool[Array, "batch seq_len"], out_dtype: DTypeLike = jnp.int32
) -> Int[Array, "batch_plus_one"]:
    """
    Creates cumulative sequence lengths from a boolean attention mask.

    Args:
        mask: A 2D boolean attention mask (batch_size, seq_len).
        out_dtype: The desired dtype for the output array.

    Returns:
        A 1D array of cumulative sequence lengths (e.g., [0, len1, len1+len2, ...]).
    """
    cumsum_lens = prepare_lens_from_mask(mask).cumsum(axis=0, dtype=out_dtype)
    return jnp.pad(cumsum_lens, (1, 0))


def prepare_position_ids(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens"]:
    """
    Generates position IDs for a batch of packed sequences.

    This creates a single 1D array like [0, 1, 2, 0, 1, 0, 1, 2, 3] for sequences
    of lengths [3, 2, 4].

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths.

    Returns:
        A 1D array of position IDs for the packed sequences.
    """
    lens = prepare_lens(cu_seqlens)
    total_length = cu_seqlens[-1]

    indices = jnp.arange(total_length, dtype=cu_seqlens.dtype)

    start_offsets = jnp.repeat(cu_seqlens[:-1], repeats=lens)

    return indices - start_offsets


def prepare_sequence_ids(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens"]:
    """
    Generates sequence IDs (0-indexed) for a batch of packed sequences.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths.

    Returns:
        A 1D array of sequence IDs, e.g., [0, 0, 0, 1, 1, 2, 2, 2, 2].
    """
    position_ids = prepare_position_ids(cu_seqlens)
    return (position_ids == 0).cumsum(axis=0) - 1


def prepare_token_indices(cu_seqlens: Int[Array, "num_seqs_plus_one"]) -> Int[Array, "total_tokens 2"]:
    """
    Generates (sequence_id, position_id) pairs for each token in the packed batch.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths.

    Returns:
        A 2D array of shape (total_tokens, 2) where each row is [sequence_id, position_id].
    """
    position_ids = prepare_position_ids(cu_seqlens)

    sequence_ids = (position_ids == 0).cumsum(axis=0) - 1

    stacked = jnp.stack([sequence_ids, position_ids], axis=1)
    return stacked.astype(cu_seqlens.dtype)


def prepare_chunk_indices(cu_seqlens: Int[Array, "num_seqs_plus_one"], chunk_size: int) -> Int[Array, "total_chunks 2"]:
    """
    Generates (sequence_id, chunk_id) pairs for each chunk in the packed batch.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths.
        chunk_size: The size of each chunk.

    Returns:
        A 2D array of shape (total_chunks, 2) where each row is [sequence_id, chunk_id_in_sequence].
    """
    lens = prepare_lens(cu_seqlens)
    num_chunks_per_seq = cdiv(lens, chunk_size)

    total_chunks = num_chunks_per_seq.sum()
    cu_chunks = jnp.pad(num_chunks_per_seq.cumsum(), (1, 0))
    start_offsets = jnp.repeat(cu_chunks[:-1], repeats=num_chunks_per_seq)

    indices = jnp.arange(total_chunks) - start_offsets

    sequence_ids_for_chunks = (indices == 0).cumsum(axis=0) - 1

    stacked = jnp.stack([sequence_ids_for_chunks, indices], axis=1)
    return stacked.astype(cu_seqlens.dtype)


def prepare_chunk_offsets(
    cu_seqlens: Int[Array, "num_seqs_plus_one"], chunk_size: int
) -> Int[Array, "num_seqs_plus_one"]:
    """
    Computes the cumulative offsets of chunks in the packed batch.

    Args:
        cu_seqlens: A 1D array of cumulative sequence lengths.
        chunk_size: The size of each chunk.

    Returns:
        A 1D array of cumulative chunk counts (e.g., [0, num_chunks_seq1, num_chunks_seq1 + num_chunks_seq2, ...]).
    """
    num_chunks_per_seq = cdiv(prepare_lens(cu_seqlens), chunk_size)
    zero = jnp.array([0], dtype=cu_seqlens.dtype)

    concatenated = jnp.concatenate([zero, num_chunks_per_seq])
    return concatenated.cumsum(axis=-1)


def segment_ids_to_mask(
    segment_ids: Int[Array, "batch seq_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]],
    dtype: DTypeLike = jnp.bool_,
    return_separate_masks: bool = False,
) -> Array | tuple[Array, Array, Array]:
    """
    Converts segment IDs to an attention mask.

    This function creates a 2D or 4D attention mask from segment IDs, where tokens
    in the same segment can attend to each other. It properly handles the padding
    conventions:
    - Segment IDs: -1 or 0 indicates padding
    - Attention mask: 0 indicates padding (masked out), 1 indicates valid attention

    The function works with both query and key-value segment IDs:
    - If only query segment IDs are provided: creates a square mask where tokens
      with the same segment ID can attend to each other
    - If both query and key-value segment IDs are provided: creates a rectangular
      mask allowing cross-attention between matching segments

    Args:
        segment_ids: Segment IDs array. Can be:
            - 2D: (batch_size, seq_len) for query segment IDs only
            - Tuple of two 2D arrays: (q_segment_ids, kv_segment_ids)
        dtype: The output dtype for the mask. Common choices:
            - jnp.bool_: Boolean mask (True=attend, False=masked)
            - jnp.float32: Float mask (1.0=attend, 0.0=masked)
        return_separate_masks: If True, returns (q_mask, kv_mask, attention_mask) tuple
            where q_mask and kv_mask are 2D masks indicating valid (non-padding) tokens.
            Default is False, which returns only the attention_mask.

    Returns:
        If return_separate_masks=False (default):
            Attention mask array with shape:
            - (batch_size, seq_len, seq_len) if segment_ids is 2D
            - (batch_size, q_len, kv_len) if segment_ids is a tuple

        If return_separate_masks=True:
            Tuple of (q_mask, kv_mask, attention_mask) where:
            - q_mask: (batch_size, q_len) - query mask (True for valid tokens)
            - kv_mask: (batch_size, kv_len) - key-value mask (True for valid tokens)
            - attention_mask: (batch_size, q_len, kv_len) - pairwise attention mask

        The mask will be broadcasted to (batch_size, num_heads, seq_len, seq_len)
        when used in attention operations.

    Examples:
        >>>
        >>> segment_ids = jnp.array([
        ...     [1, 1, 2, 2, -1],
        ...     [1, 1, 1, -1, -1],
        ... ])
        >>> mask = segment_ids_to_mask(segment_ids)
        >>> mask.shape
        (2, 5, 5)
        >>>
        >>>
        >>>

        >>>
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_mask(segment_ids, return_separate_masks=True)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((2, 5), (2, 5), (2, 5, 5))
        >>> q_mask[0]
        >>> kv_mask[0]

        >>>
        >>> q_segment_ids = jnp.array([[1, 2, 3]])
        >>> kv_segment_ids = jnp.array([[1, 1, 2, 2, 3]])
        >>> mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids))
        >>> mask.shape
        (1, 3, 5)
        >>>
        >>>
        >>>

        >>>
        >>> mask = segment_ids_to_mask(segment_ids, dtype=jnp.float32)
        >>>

    Notes:
        - Segment IDs of -1 or 0 are treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - The output mask is suitable for use with most attention implementations
        - For additive attention bias, convert: bias = (1.0 - mask) * large_negative_value
    """
    if isinstance(segment_ids, tuple):
        q_segment_ids, kv_segment_ids = segment_ids
        q_mask = (q_segment_ids > 0).astype(dtype)
        kv_mask = (kv_segment_ids > 0).astype(dtype)
        q_seg = q_segment_ids[:, :, None]
        kv_seg = kv_segment_ids[:, None, :]
        attention_mask = (q_seg == kv_seg) & (q_seg > 0) & (kv_seg > 0)
    else:
        q_mask = (segment_ids > 0).astype(dtype)
        kv_mask = q_mask
        seg_q = segment_ids[:, :, None]
        seg_kv = segment_ids[:, None, :]
        attention_mask = (seg_q == seg_kv) & (seg_q > 0) & (seg_kv > 0)

    attention_mask = attention_mask.astype(dtype)

    if return_separate_masks:
        return q_mask, kv_mask, attention_mask
    else:
        return attention_mask


def segment_ids_to_qkv_masks(
    q_segment_ids: Int[Array, "batch q_len"],
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    dtype: DTypeLike = jnp.bool_,
) -> tuple[Array, Array, Array]:
    """
    Converts query and key-value segment IDs to separate Q mask, KV mask, and attention mask.

    This is a convenience function that always returns the three masks separately,
    useful when you need individual control over query and key-value masking.

    Args:
        q_segment_ids: Query segment IDs of shape (batch_size, q_len).
            Values of -1 or 0 indicate padding.
        kv_segment_ids: Key-value segment IDs of shape (batch_size, kv_len).
            If None, uses q_segment_ids (self-attention case).
            Values of -1 or 0 indicate padding.
        dtype: The output dtype for masks. Common choices:
            - jnp.bool_: Boolean mask (True=attend, False=masked)
            - jnp.float32: Float mask (1.0=attend, 0.0=masked)

    Returns:
        Tuple of (q_mask, kv_mask, attention_mask):
        - q_mask: (batch_size, q_len) - Query mask indicating valid (non-padding) query tokens
        - kv_mask: (batch_size, kv_len) - Key-value mask indicating valid (non-padding) KV tokens
        - attention_mask: (batch_size, q_len, kv_len) - Pairwise attention mask where tokens
          in matching segments can attend to each other

    Examples:
        >>>
        >>> segment_ids = jnp.array([[1, 1, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(segment_ids)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 4), (1, 4), (1, 4, 4))
        >>> q_mask[0]
        >>> attn_mask[0, 0, 2]

        >>>
        >>> q_seg = jnp.array([[1, 2]])
        >>> kv_seg = jnp.array([[1, 1, 2, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(q_seg, kv_seg)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 2), (1, 5), (1, 2, 5))
        >>> kv_mask[0]
        >>> attn_mask[0, 0, :2]

        >>>
        >>>
        >>>
        >>>
        >>>

    Notes:
        - This function always returns three separate masks for maximum flexibility
        - Segment IDs of -1 or 0 are treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - For self-attention, q_mask and kv_mask will be identical
    """
    if kv_segment_ids is None:
        kv_segment_ids = q_segment_ids

    return segment_ids_to_mask((q_segment_ids, kv_segment_ids), dtype=dtype, return_separate_masks=True)


def identity_dtype_convert(dtype: jnp.dtype):
    @jax.custom_vjp
    def identity_fn(x):
        return x

    def identity_fn_fwd(x):
        return x, None

    def identity_fn_bwd(res, g):
        return (g.astype(dtype),)

    identity_fn.defvjp(identity_fn_fwd, identity_fn_bwd)

    return identity_fn


def _compress_ids_from_anchors(anchors: jnp.ndarray, pad_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Convert 'anchors' (min representative index per element) into contiguous segment IDs [0..G-1],
    with -1 for padded entries indicated by pad_mask.
    """
    n = anchors.shape[0]
    sentinel = n + 1
    vals = jnp.where(pad_mask, sentinel, anchors)
    idx_sorted = jnp.argsort(vals)
    vals_sorted = vals[idx_sorted]
    valid_sorted = vals_sorted != sentinel

    head = valid_sorted[:1]
    rest_new = (vals_sorted[1:] != vals_sorted[:-1]) & valid_sorted[1:]
    is_new_sorted = jnp.concatenate([head, rest_new], axis=0).astype(jnp.int32)

    gid_sorted = jnp.cumsum(is_new_sorted) - 1
    gid_sorted = jnp.where(valid_sorted, gid_sorted, -1)

    gid = jnp.zeros_like(gid_sorted)
    gid = gid.at[idx_sorted].set(gid_sorted)
    return gid


def _mask_to_segments_single(m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    m: (Q, K) boolean mask
    Returns:
      q_segment_ids: (Q,) int32 in [0..Gq-1], -1 for all-zero rows
      kv_segment_ids: (K,) int32 in [0..Gk-1], -1 for all-zero cols
    """
    m = m.astype(jnp.bool_)
    Q, K = m.shape

    q_pad = ~jnp.any(m, axis=-1)
    kv_pad = ~jnp.any(m, axis=0)

    row_bytes = jnp.packbits(m, axis=-1)
    row_equal = jnp.all(row_bytes[:, None, :] == row_bytes[None, :, :], axis=-1)
    idxs_q = jnp.arange(Q, dtype=jnp.int32)[None, :]
    q_anchors = jnp.min(jnp.where(row_equal, idxs_q, Q), axis=-1)
    q_segment_ids = _compress_ids_from_anchors(q_anchors, q_pad)

    col_bytes = jnp.packbits(m.T, axis=-1)
    col_equal = jnp.all(col_bytes[:, None, :] == col_bytes[None, :, :], axis=-1)
    idxs_k = jnp.arange(K, dtype=jnp.int32)[None, :]
    kv_anchors = jnp.min(jnp.where(col_equal, idxs_k, K), axis=-1)
    kv_segment_ids = _compress_ids_from_anchors(kv_anchors, kv_pad)

    return q_segment_ids, kv_segment_ids


def mask_to_segment_ids(mask: jnp.ndarray, per_head: bool = False):
    """
    JIT-friendly mask → (q_segment_ids, kv_segment_ids) for rectangular masks.

    Input shapes:
      - (Q, K)
      - (B, Q, K)
      - (B, H, Q, K)

    Returns:
      - If (Q, K): (Q,), (K,)
      - If (B, Q, K): (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=False: (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=True:  (B, H, Q), (B, H, K)

    Padded rows/cols (all-zero) receive segment id -1.
    """
    m = mask.astype(jnp.bool_)

    if m.ndim == 2:
        q_ids, kv_ids = _mask_to_segments_single(m)
        return q_ids, kv_ids

    if m.ndim == 3:
        q_ids, kv_ids = jax.vmap(_mask_to_segments_single, in_axes=0)(m)
        return q_ids, kv_ids

    if m.ndim == 4:
        if per_head:
            q_ids, kv_ids = jax.vmap(jax.vmap(_mask_to_segments_single, in_axes=0), in_axes=0)(m)
            return q_ids, kv_ids
        else:
            q_ids, kv_ids = jax.vmap(_mask_to_segments_single, in_axes=0)(m[:, 0, :, :])
            return q_ids, kv_ids

    raise ValueError(f"mask must be (Q,K), (B,Q,K), or (B,H,Q,K); got {m.shape}")
