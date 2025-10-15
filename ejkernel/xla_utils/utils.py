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


def mask_to_segment_ids(attention_mask: Array) -> tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]]:
    """
    Converts an attention mask to query and key-value segment IDs.

    This function analyzes the attention mask pattern to infer segment boundaries.
    It uses a simplified approach that identifies contiguous blocks of valid attention
    and assigns segment IDs accordingly. This is JAX-compatible and works with JIT.

    Args:
        attention_mask: Attention mask of shape:
            - (batch_size, seq_len, seq_len) for self-attention
            - (batch_size, q_len, kv_len) for cross-attention
            - (batch_size, num_heads, q_len, kv_len) - will use first head
            Values: True/1 for valid attention, False/0 for masked

    Returns:
        Tuple of (q_segment_ids, kv_segment_ids):
        - q_segment_ids: (batch_size, q_len) - Segment IDs for queries (-1 for padding)
        - kv_segment_ids: (batch_size, kv_len) - Segment IDs for keys/values (-1 for padding)

    Examples:
        >>>
        >>> mask = jnp.array([
        ...     [[True, True, False, False],
        ...      [True, True, False, False],
        ...      [False, False, True, True],
        ...      [False, False, True, True]]
        ... ])
        >>> q_seg, kv_seg = mask_to_segment_ids(mask)
        >>>
        >>>

        >>>
        >>> mask_with_pad = jnp.array([
        ...     [[True, True, False, False],
        ...      [True, True, False, False],
        ...      [False, False, True, False],
        ...      [False, False, False, False]]
        ... ])
        >>> q_seg, kv_seg = mask_to_segment_ids(mask_with_pad)
        >>>

    Notes:
        - This is a JAX-compatible, JIT-safe implementation
        - Works well for simple patterns (padding, contiguous segments)
        - Padding tokens (no valid attention) are assigned segment ID -1
        - Segment IDs are 1-indexed (1, 2, 3, ...) to distinguish from padding
        - For complex patterns, consider providing explicit segment IDs directly
        - Uses cumulative sum approach to identify segment boundaries
    """

    if attention_mask.ndim == 4:
        mask_bool = attention_mask[:, 0, :, :].astype(jnp.bool_)
    else:
        mask_bool = attention_mask.astype(jnp.bool_)

    q_is_valid = jnp.any(mask_bool, axis=-1)
    kv_is_valid = jnp.any(mask_bool, axis=-2)

    q_pattern_changes = jnp.concatenate(
        [jnp.ones_like(q_is_valid[:, :1]), jnp.any(mask_bool[:, 1:, :] != mask_bool[:, :-1, :], axis=-1)], axis=1
    )

    q_segment_ids = jnp.cumsum(q_pattern_changes, axis=1, dtype=jnp.int32)

    q_segment_ids = jnp.where(q_is_valid, q_segment_ids, -1)

    kv_pattern_changes = jnp.concatenate(
        [jnp.ones_like(kv_is_valid[:, :1]), jnp.any(mask_bool[:, :, 1:] != mask_bool[:, :, :-1], axis=-2)], axis=1
    )

    kv_segment_ids = jnp.cumsum(kv_pattern_changes, axis=1, dtype=jnp.int32)
    kv_segment_ids = jnp.where(kv_is_valid, kv_segment_ids, -1)

    return q_segment_ids, kv_segment_ids


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
