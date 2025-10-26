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


"""
Attention Mask Management for JAX/Flax Models.

This module provides comprehensive tools for creating, manipulating, and converting
attention masks in transformer models. It supports various attention patterns and
provides efficient conversions between different mask representations.

Key Components:
    - MaskInfo: Main dataclass for managing attention masks and segment IDs
    - Conversion functions: Convert between masks and segment IDs
    - Attention patterns: Causal, sliding window, chunked, token-type-based
    - Distributed support: Sharding specifications for multi-device training
    - Visualization: Debug and understand attention patterns

Common Usage:
    >>>
    >>> mask_info = MaskInfo.from_segments(segment_ids)
    >>>
    >>>
    >>> mask_info = MaskInfo.from_attention_mask(attention_mask)
    >>>
    >>>
    >>> causal_mask_info = mask_info.apply_causal()
    >>>
    >>>
    >>> bias = mask_info.bias

Mask Representations:
    1. Attention Mask: 4D boolean/int array (batch, heads, q_len, kv_len)
       - True/1 = valid attention, False/0 = masked
    2. Segment IDs: 2D int32 arrays (batch, seq_len)
       - Non-negative = segment membership
       - -1 = padding tokens

See Also:
    - mask_to_segment_ids(): Convert masks to segment IDs
    - segment_ids_to_mask(): Convert segment IDs to masks
    - MaskInfo: Main class for mask management
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.xla_utils import get_corrected_named_sharding

mdim_t = "batch nheads_or_1 qlen kvlen"


def _compress_ids_from_anchors(anchors: jnp.ndarray, pad_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Convert anchors (minimum representative index per element) into contiguous segment IDs.

    Takes an array where each element points to a representative "anchor" element (the
    minimum index in its group) and converts it to contiguous segment IDs [0, 1, 2, ...].
    Padded entries (indicated by pad_mask) are assigned segment ID -1.

    This is a helper function for mask-to-segment-ID conversion that ensures segment
    IDs are compact and contiguous.

    Args:
        anchors: Array where each element contains the minimum index of its group.
            Elements in the same group point to the same anchor.
        pad_mask: Boolean array indicating which positions are padding (True = padded)

    Returns:
        Array of contiguous segment IDs [0..G-1] where G is the number of groups,
        with -1 for padded entries

    Example:
        >>> anchors = jnp.array([0, 0, 2, 2, 4])
        >>> pad_mask = jnp.array([False, False, False, False, True])
        >>> _compress_ids_from_anchors(anchors, pad_mask)
        Array([0, 0, 1, 1, -1], dtype=int32)
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
    Convert a single 2D attention mask to query and key-value segment IDs.

    Analyzes the attention pattern to group queries and keys/values into segments.
    Rows (queries) with identical attention patterns are assigned the same segment ID.
    Columns (keys/values) with identical patterns are also grouped.

    This is a helper function for mask-to-segment-ID conversion that processes a
    single 2D mask. Use mask_to_segment_ids() for batched processing.

    Args:
        m: 2D boolean attention mask of shape (Q, K) where True indicates valid attention

    Returns:
        Tuple of (q_segment_ids, kv_segment_ids):
        - q_segment_ids: (Q,) int32 array with segment IDs in [0..Gq-1], -1 for all-zero rows
        - kv_segment_ids: (K,) int32 array with segment IDs in [0..Gk-1], -1 for all-zero cols

    Example:
        >>> mask = jnp.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
        >>> q_ids, kv_ids = _mask_to_segments_single(mask)
        >>> q_ids
        Array([0, 0, 1], dtype=int32)
        >>> kv_ids
        Array([0, 0, 1], dtype=int32)
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


def mask_to_segment_ids(mask: jnp.ndarray, per_head: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert attention mask to segment IDs (JIT-friendly).

    Analyzes the attention mask structure to extract query and key-value segment IDs.
    Queries with identical attention patterns are grouped into the same segment,
    and similarly for keys/values. This conversion is useful for optimized attention
    implementations that can leverage segment structure.

    Input shapes:
      - (Q, K): Single 2D mask
      - (B, Q, K): Batched 2D masks
      - (B, H, Q, K): Batched multi-head masks

    Args:
        mask: Boolean or integer attention mask array
        per_head: If True and mask is 4D, compute segment IDs separately per head.
            If False, merge across heads (default behavior). Default: False

    Returns:
      Tuple of (q_segment_ids, kv_segment_ids) with shapes:
      - If (Q, K): (Q,), (K,)
      - If (B, Q, K): (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=False: (B, Q), (B, K)
      - If (B, H, Q, K) and per_head=True:  (B, H, Q), (B, H, K)

    Notes:
        - Padded rows/cols (all-zero) receive segment ID -1
        - Queries/keys with identical attention patterns share the same segment ID
        - This function is JIT-compatible for use in compiled JAX programs

    Raises:
        ValueError: If mask shape is not 2D, 3D, or 4D

    Example:
        >>> mask = jnp.array([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]])
        >>> q_ids, kv_ids = mask_to_segment_ids(mask)
        >>> q_ids.shape, kv_ids.shape
        ((1, 3), (1, 3))
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
            merged = jnp.all(m, axis=1)
            q_ids, kv_ids = jax.vmap(_mask_to_segments_single, in_axes=0)(merged)
            return q_ids, kv_ids
    raise ValueError(
        f"Invalid mask shape. Expected 2D (Q,K), 3D (batch,Q,K), or 4D (batch,heads,Q,K) mask, "
        f"but got {m.ndim}D mask with shape {m.shape}. "
        f"Ensure your attention mask has the correct dimensionality."
    )


def _positions_from_segments_2d(segment_ids: jnp.ndarray, *, pad_value: int) -> jnp.ndarray:
    """
    Compute 0-based positions per segment (reset at boundaries).
    Padding (-1) gets pad_value.

    Args:
        segment_ids: int32 array (batch, seqlen), -1 = padding, non-negative = segment id
        pad_value: value for padding positions (e.g. -1 for Q, int32_max for KV)

    Returns:
        positions: int32 array (batch, seqlen)
    """
    segment_ids = jnp.asarray(segment_ids, jnp.int32)

    def _scan_1d(ids_1d):
        def step(carry, seg_i):
            prev_seg, cnt = carry
            is_pad = seg_i < 0
            is_new = (~is_pad) & (seg_i != prev_seg)
            cnt_candidate = jnp.where(is_new, jnp.int32(0), cnt + 1)

            pos_i = jnp.where(is_pad, jnp.int32(pad_value), cnt_candidate)
            next_prev = jnp.where(is_pad, jnp.int32(-2), seg_i)
            next_cnt = jnp.where(is_pad, jnp.int32(-1), cnt_candidate)
            return (next_prev, next_cnt), pos_i

        (_, _), pos = jax.lax.scan(step, (jnp.int32(-2), jnp.int32(-1)), ids_1d)
        return pos

    return jax.vmap(_scan_1d, in_axes=0, out_axes=0)(segment_ids)


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
    - Segment IDs: -1 indicates padding
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
            - (batch_size, 1, seq_len, seq_len) if segment_ids is 2D
            - (batch_size, 1, q_len, kv_len) if segment_ids is a tuple

            The mask is always 4D with shape (batch, 1, q, kv) where the second
            dimension is 1 to allow broadcasting across attention heads.

        If return_separate_masks=True:
            Tuple of (q_mask, kv_mask, attention_mask) where:
            - q_mask: (batch_size, q_len) - query mask (True for valid tokens)
            - kv_mask: (batch_size, kv_len) - key-value mask (True for valid tokens)
            - attention_mask: (batch_size, 1, q_len, kv_len) - 4D pairwise attention mask

    Examples:
        >>>
        >>> segment_ids = jnp.array([
        ...     [1, 1, 2, 2, -1],
        ...     [1, 1, 1, -1, -1],
        ... ])
        >>> mask = segment_ids_to_mask(segment_ids)
        >>> mask.shape
        (2, 1, 5, 5)
        >>>
        >>>
        >>>

        >>>
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_mask(segment_ids, return_separate_masks=True)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((2, 5), (2, 5), (2, 1, 5, 5))
        >>> q_mask[0]
        >>> kv_mask[0]

        >>>
        >>> q_segment_ids = jnp.array([[1, 2, 3]])
        >>> kv_segment_ids = jnp.array([[1, 1, 2, 2, 3]])
        >>> mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids))
        >>> mask.shape
        (1, 1, 3, 5)
        >>>
        >>>
        >>>

        >>>
        >>> mask = segment_ids_to_mask(segment_ids, dtype=jnp.float32)
        >>>

    Notes:
        - Segment IDs of -1 are treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - The output mask is suitable for use with most attention implementations
        - For additive attention bias, convert: bias = (1.0 - mask) * large_negative_value
    """
    if isinstance(segment_ids, tuple):
        q_segment_ids, kv_segment_ids = segment_ids
        q_valid = q_segment_ids >= 0
        kv_valid = kv_segment_ids >= 0
        q_mask = q_valid.astype(dtype)
        kv_mask = kv_valid.astype(dtype)
        attention_mask = (
            (q_segment_ids[:, :, None] == kv_segment_ids[:, None, :]) & q_valid[:, :, None] & kv_valid[:, None, :]
        )
    else:
        q_valid = segment_ids >= 0
        q_mask = q_valid.astype(dtype)
        kv_mask = q_mask
        attention_mask = (segment_ids[:, :, None] == segment_ids[:, None, :]) & q_valid[:, :, None] & q_valid[:, None, :]

    attention_mask = attention_mask.astype(dtype)
    attention_mask = attention_mask[:, None, :, :]

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
            Values of -1 indicate padding.
        kv_segment_ids: Key-value segment IDs of shape (batch_size, kv_len).
            If None, uses q_segment_ids (self-attention case).
            Values of -1 indicate padding.
        dtype: The output dtype for masks. Common choices:
            - jnp.bool_: Boolean mask (True=attend, False=masked)
            - jnp.float32: Float mask (1.0=attend, 0.0=masked)

    Returns:
        Tuple of (q_mask, kv_mask, attention_mask):
        - q_mask: (batch_size, q_len) - Query mask indicating valid (non-padding) query tokens
        - kv_mask: (batch_size, kv_len) - Key-value mask indicating valid (non-padding) KV tokens
        - attention_mask: (batch_size, 1, q_len, kv_len) - 4D pairwise attention mask where tokens
          in matching segments can attend to each other

    Examples:
        >>>
        >>> segment_ids = jnp.array([[1, 1, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(segment_ids)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 4), (1, 4), (1, 1, 4, 4))
        >>> q_mask[0]
        >>> attn_mask[0, 0, 0, 2]

        >>>
        >>> q_seg = jnp.array([[1, 2]])
        >>> kv_seg = jnp.array([[1, 1, 2, 2, -1]])
        >>> q_mask, kv_mask, attn_mask = segment_ids_to_qkv_masks(q_seg, kv_seg)
        >>> q_mask.shape, kv_mask.shape, attn_mask.shape
        ((1, 2), (1, 5), (1, 1, 2, 5))
        >>> kv_mask[0]
        >>> attn_mask[0, 0, 0, :2]

        >>>
        >>>
        >>>
        >>>
        >>>

    Notes:
        - This function always returns three separate masks for maximum flexibility
        - Segment IDs of -1 is treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - For self-attention, q_mask and kv_mask will be identical
    """
    if kv_segment_ids is None:
        kv_segment_ids = q_segment_ids
    return segment_ids_to_mask((q_segment_ids, kv_segment_ids), dtype=dtype, return_separate_masks=True)


class MaskSharding(NamedTuple):
    """
    Container for sharding specifications of attention mask components.

    Used to specify how different parts of the mask should be partitioned
    across devices in distributed training scenarios.

    Attributes:
        attention_mask: Sharding spec for the 4D attention mask (batch, heads, q, kv)
        q_segment_ids: Sharding spec for query segment IDs (batch, qlen)
        kv_segment_ids: Sharding spec for key-value segment IDs (batch, kvlen)
        q_positions: Sharding spec for query positions (batch, qlen)
        kv_positions: Sharding spec for key-value positions (batch, kvlen)
    """

    attention_mask: PartitionSpec | None
    q_segment_ids: PartitionSpec | None
    kv_segment_ids: PartitionSpec | None
    q_positions: PartitionSpec | None
    kv_positions: PartitionSpec | None


@dataclass
class MaskInfo:
    """
    Container for attention mask information with utilities for conversion and manipulation.

    This dataclass holds both attention masks and their corresponding segment IDs,
    along with optional position indices for queries and keys/values.
    It provides convenient methods for conversion between representations and extracting
    derived information.

    Attributes:
        attention_mask: The 2D/3D/4D boolean or integer attention mask
        q_segment_ids: Query segment IDs (batch, qlen) where -1 indicates padding
        kv_segment_ids: Key-value segment IDs (batch, kvlen) where -1 indicates padding
        q_positions: Query position indices (batch, qlen) for positional embeddings
        kv_positions: Key-value position indices (batch, kvlen) for positional embeddings
    """

    _attention_mask: Bool[Array, "batch nheads_or_1 q k"] | Int[Array, "batch nheads_or_1 q k"] | None = None
    _q_segment_ids: Int[Array, "batch q"] | None = None
    _kv_segment_ids: Int[Array, "batch k"] | None = None

    q_positions: Int[Array, "batch qlen"] | None = None
    kv_positions: Int[Array, "batch kvlen"] | None = None

    batch_axis_name: tuple[str] | str | None = field(default=("dp", "fsdp"))
    qheads_axis_name: tuple[str] | str | None = field(default="tp")
    kvheads_axis_name: tuple[str] | str | None = field(default="tp")
    sequence_axis_name: tuple[str] | str | None = field(default="sp")

    @property
    def attention_mask(self) -> Array | None:
        if self._attention_mask is None:
            self._attention_mask = self.get_or_compute_attention_mask()
        return self._attention_mask

    @property
    def q_segment_ids(self) -> Array | None:
        if self._q_segment_ids is None:
            self._q_segment_ids, self._kv_segment_ids = self.get_or_compute_segment_ids()
        return self._q_segment_ids

    @property
    def kv_segment_ids(self) -> Array | None:
        if self._kv_segment_ids is None:
            self._q_segment_ids, self._kv_segment_ids = self.get_or_compute_segment_ids()
        return self._kv_segment_ids

    def materialize_attention_mask(self, dtype: DTypeLike = jnp.bool_) -> "MaskInfo":
        if self._attention_mask is not None:
            return (
                self
                if self._attention_mask.dtype == dtype
                else self.replace(attention_mask=self._attention_mask.astype(dtype))
            )

        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            m = segment_ids_to_mask((self._q_segment_ids, self._kv_segment_ids), dtype=dtype)
            return self.replace(attention_mask=m)
        raise ValueError(
            "Cannot materialize attention_mask: no source data available. "
            "Either provide an attention_mask directly, or provide both q_segment_ids and kv_segment_ids "
            "so the mask can be computed from segment information."
        )

    def materialize_segment_ids(self, per_head: bool = False) -> "MaskInfo":
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            return self
        if self._attention_mask is None:
            raise ValueError(
                "Cannot materialize segment IDs: no attention_mask available. "
                "Provide an attention_mask to compute segment IDs from, or initialize with segment IDs directly."
            )
        q_ids, kv_ids = mask_to_segment_ids(self._attention_mask, per_head=per_head)

        q_ids = jnp.asarray(q_ids, jnp.int32)
        kv_ids = jnp.asarray(kv_ids, jnp.int32)
        return self.replace(q_segment_ids=q_ids, kv_segment_ids=kv_ids)

    @classmethod
    def from_segments(
        cls,
        q_segment_ids: Int[Array, "batch qlen"],
        kv_segment_ids: Int[Array, "batch kvlen"] | None = None,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        apply_padding: bool = False,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """
        Create MaskInfo from segment IDs.

        Constructs a MaskInfo instance from query and key-value segment IDs, automatically
        generating the corresponding attention mask. Segment IDs group tokens that can
        attend to each other (same segment ID = can attend).

        Args:
            q_segment_ids: Query segment IDs of shape (batch, qlen). Values should be:
                - Non-negative integers: segment membership (0, 1, 2, ...)
                - -1: padding tokens
            kv_segment_ids: Key-value segment IDs of shape (batch, kvlen). If None, uses
                q_segment_ids (self-attention case). Values follow same convention as q_segment_ids.
            q_positions: Optional query position indices (batch, qlen) for positional embeddings
            kv_positions: Optional key-value position indices (batch, kvlen) for positional embeddings
            apply_padding: If True, converts binary segment IDs (0/1) to padding convention (-1/0).
                This is useful when segment IDs use 0 to indicate padding instead of -1.
                Default: False
            batch_axis_name: Axis name(s) for batch dimension in distributed sharding.
                Default: ("dp", "fsdp")
            qheads_axis_name: Axis name(s) for query heads dimension in distributed sharding.
                Default: "tp"
            kvheads_axis_name: Axis name(s) for key-value heads dimension in distributed sharding.
                Default: "tp"
            sequence_axis_name: Axis name(s) for sequence dimension in distributed sharding.
                Default: "sp"

        Returns:
            MaskInfo with segment IDs, computed attention mask, optional positions, and sharding configuration

        Example:
            >>> q_seg = jnp.array([[1, 1, 2, 2, -1]])
            >>> mask_info = MaskInfo.from_segments(q_seg)
            >>> mask_info.attention_mask.shape
            (1, 1, 5, 5)
        """

        def _canon_01_to_neg1_0(ids):
            ids = jnp.asarray(ids, dtype=jnp.int32)
            is_01 = jnp.all((ids == 0) | (ids == 1))
            return jax.lax.cond(
                is_01,
                lambda x: jnp.where(x == 0, jnp.int32(-1), jnp.int32(0)),
                lambda x: x,
                ids,
            )

        q_segment_ids = jnp.asarray(q_segment_ids, dtype=jnp.int32)
        if kv_segment_ids is not None:
            kv_segment_ids = jnp.asarray(kv_segment_ids, dtype=jnp.int32)

        if apply_padding:
            q_segment_ids = _canon_01_to_neg1_0(q_segment_ids)

        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids
        elif apply_padding:
            kv_segment_ids = _canon_01_to_neg1_0(kv_segment_ids)

        return cls(
            _attention_mask=None,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @classmethod
    def from_attention_mask(
        cls,
        attention_mask: Bool[Array, mdim_t] | Int[Array, mdim_t],
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """
        Create MaskInfo from an existing attention mask.

        Analyzes the attention mask structure to extract segment IDs and create
        a complete MaskInfo representation. Useful when you have a mask and need
        to derive the segment structure.

        Args:
            attention_mask: Attention mask array. Supported shapes:
                - (qlen, kvlen): 2D padding mask
                - (batch, qlen, kvlen): 3D batched mask
                - (batch, heads, qlen, kvlen): 4D multi-head mask
                Values: True/1 = valid attention, False/0 = masked
            per_head: If True and mask is 4D, compute separate segment IDs per head.
                If False, merge mask across heads before computing segments. Default: False
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)

        Returns:
            MaskInfo with derived segment IDs, original attention mask, and optional positions

        Raises:
            ValueError: If attention_mask is not 2D, 3D, or 4D

        Example:
            >>> mask = jnp.array([[[1, 1, 0], [1, 1, 0], [0, 0, 1]]])
            >>> mask_info = MaskInfo.from_attention_mask(mask)
            >>> mask_info.q_segment_ids.shape
            (1, 3)
        """
        m = attention_mask.astype(jnp.bool_)
        q_segment_ids = kv_segment_ids = None
        if m.ndim == 2:
            q_segment_ids = jnp.where(m, 0, -1)
            kv_segment_ids = q_segment_ids
            pairwise_mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids))
            return cls(
                _attention_mask=pairwise_mask,
                _q_segment_ids=q_segment_ids,
                _kv_segment_ids=kv_segment_ids,
                q_positions=q_positions,
                kv_positions=kv_positions,
                batch_axis_name=batch_axis_name,
                qheads_axis_name=qheads_axis_name,
                kvheads_axis_name=kvheads_axis_name,
                sequence_axis_name=sequence_axis_name,
            )
        if m.ndim == 3:
            m = m[:, None, :, :]
        elif m.ndim != 4:
            raise ValueError(
                f"Invalid attention_mask dimensionality. Expected 2D (Q,K), 3D (batch,Q,K), or 4D (batch,heads,Q,K), "
                f"but got {m.ndim}D with shape {m.shape}. "
                f"Check that your mask has the proper dimensions for the attention operation."
            )

        return cls(
            _attention_mask=m,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @classmethod
    def from_random(
        cls,
        batch_size: int,
        q_len: int,
        kv_len: int | None = None,
        sparsity: float = 0.5,
        seed: int = 0,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
        batch_axis_name: tuple[str] | str | None = ("dp", "fsdp"),
        qheads_axis_name: tuple[str] | str | None = "tp",
        kvheads_axis_name: tuple[str] | str | None = "tp",
        sequence_axis_name: tuple[str] | str | None = "sp",
    ) -> "MaskInfo":
        """
        Create MaskInfo with random attention pattern.

        Generates a random binary attention mask with specified sparsity level.
        Useful for testing, experimentation, and studying sparse attention patterns.

        Args:
            batch_size: Batch size
            q_len: Query sequence length
            kv_len: Key-value sequence length. If None, uses q_len (self-attention)
            sparsity: Fraction of attention positions to mask out (0.0 = full attention,
                1.0 = fully masked). Default: 0.5 (50% masked)
            seed: Random seed for reproducibility. Default: 0
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)

        Returns:
            MaskInfo with random attention pattern and optional positions

        Example:
            >>>
            >>> mask_info = MaskInfo.from_random(
            ...     batch_size=2,
            ...     q_len=128,
            ...     sparsity=0.7,
            ...     seed=42
            ... )
            >>> mask_info.attention_mask.shape
            (2, 1, 128, 128)

            >>>
            >>> mask_info = MaskInfo.from_random(
            ...     batch_size=1,
            ...     q_len=64,
            ...     kv_len=128,
            ...     sparsity=0.5,
            ...     seed=0
            ... )
            >>> mask_info.attention_mask.shape
            (1, 1, 64, 128)
        """
        if kv_len is None:
            kv_len = q_len

        if not 0.0 <= sparsity <= 1.0:
            raise ValueError(
                f"Invalid sparsity value. Expected a float in the range [0.0, 1.0] "
                f"(where 0.0 = full attention, 1.0 = fully masked), but got {sparsity}. "
                f"Please provide a valid sparsity level between 0 and 1."
            )

        key = jax.random.PRNGKey(seed)

        random_mask = jax.random.bernoulli(key, p=1.0 - sparsity, shape=(batch_size, 1, q_len, kv_len))
        return cls(
            _attention_mask=random_mask,
            _q_segment_ids=None,
            _kv_segment_ids=None,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    @property
    def bias(self):
        """
        Create attention bias from the mask (convenience property).

        Returns an attention bias tensor where valid attention positions are 0.0
        and masked positions are set to the minimum float value for the dtype.

        Returns:
            Attention bias array with dtype float32
        """
        return self.create_bias()

    def create_bias(self, dtype: jnp.dtype = jnp.float32) -> Array:
        """
        Create attention bias from the mask.

        Converts the boolean attention mask into an additive bias tensor suitable
        for attention score computation. Valid positions (mask=True) get 0.0,
        while masked positions (mask=False) get a large negative value (dtype.min).

        Args:
            dtype: Output dtype for the bias tensor. Default: jnp.float32

        Returns:
            Attention bias array where:
            - Valid attention positions: 0.0
            - Masked positions: jnp.finfo(dtype).min

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> bias = mask_info.create_bias(dtype=jnp.float32)
            >>>
        """
        mask = self.get_or_compute_attention_mask()
        return jnp.where(
            mask,
            jnp.full(mask.shape, 0.0).astype(dtype),
            jnp.full(mask.shape, jnp.finfo(dtype).min).astype(dtype),
        )

    @staticmethod
    def get_empty_sharding() -> MaskSharding:
        """
        Create an empty MaskSharding with all specs set to None.

        Useful as a default or placeholder when no sharding is needed.

        Returns:
            MaskSharding with all fields set to None
        """
        return MaskSharding(
            attention_mask=None,
            q_segment_ids=None,
            kv_segment_ids=None,
            q_positions=None,
            kv_positions=None,
        )

    def get_shardings(
        self,
        sequence_parallel: bool = False,
        *,
        mesh: Mesh,
    ) -> MaskSharding:
        """
        Generate sharding specifications for all mask components.

        Creates PartitionSpec objects that define how to distribute the mask tensors
        across devices in a multi-device setup. Uses the axis names configured in
        the MaskInfo instance.

        Args:
            sequence_parallel: Whether to shard along the sequence dimension.
                If True, sequences are split across devices. Default: False
            mesh: JAX mesh defining the device grid and axis names

        Returns:
            MaskSharding containing partition specs for all mask components

        Raises:
            ValueError: If configured axis names are not present in the mesh,
                or if attention_mask is not 4D

        Example:
            >>> from jax.sharding import Mesh
            >>> devices = jax.devices()
            >>> mesh = Mesh(devices, axis_names=('dp', 'tp'))
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> shardings = mask_info.get_shardings(mesh=mesh)
        """
        batch_axis_name = self.batch_axis_name
        qheads_axis_name = self.qheads_axis_name
        sequence_axis_name = self.sequence_axis_name if sequence_parallel else None

        axis_names = set(mesh.axis_names)

        def _check(name_like, label):
            if name_like is None:
                return
            names = name_like if isinstance(name_like, tuple) else (name_like,)
            for e in names:
                if e not in axis_names:
                    raise ValueError(
                        f"Invalid axis name configuration. Axis '{e}' (from {label}) is not present in the mesh. "
                        f"Available mesh axes: {sorted(mesh.axis_names)}. "
                        f"Please ensure all configured axis names match the mesh definition."
                    )

        _check(qheads_axis_name, "qheads_axis_name")
        _check(batch_axis_name, "batch_axis_name")
        _check(sequence_axis_name, "sequence_axis_name")

        att_seq_spec = sequence_axis_name if sequence_parallel else None

        attention_mask = None
        if self._attention_mask is not None:
            if self._attention_mask.ndim != 4:
                raise ValueError(
                    f"Attention mask must be a 4D array with shape (batch, num_heads_or_1, q_len, kv_len) "
                    f"for sharding computation, but got "
                    f"{self._attention_mask.ndim}D array with shape {self._attention_mask.shape}. "
                    f"Use from_attention_mask() to construct MaskInfo from lower-dimensional masks."
                )
            att_spec = PartitionSpec(batch_axis_name, qheads_axis_name, att_seq_spec, att_seq_spec)
            attention_mask = get_corrected_named_sharding(self._attention_mask.shape, att_spec, mesh=mesh).spec

        q_segment_ids = (
            get_corrected_named_sharding(
                self._q_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self._q_segment_ids is not None
            else None
        )
        kv_segment_ids = (
            get_corrected_named_sharding(
                self._kv_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self._kv_segment_ids is not None
            else None
        )
        q_positions = (
            get_corrected_named_sharding(
                self.q_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self.q_positions is not None
            else None
        )
        kv_positions = (
            get_corrected_named_sharding(
                self.kv_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
            if self.kv_positions is not None
            else None
        )
        return MaskSharding(attention_mask, q_segment_ids, kv_segment_ids, q_positions, kv_positions)

    def get_or_compute_positions(self) -> tuple[Int[Array, "batch qlen"] | None, Int[Array, "batch kvlen"] | None]:
        """
        Get position arrays, computing them if not already available.

        Generates position indices for queries and keys/values when not explicitly provided.
        Position arrays are useful for positional embeddings and rotary position embeddings (RoPE).

        Returns:
            Tuple of (q_positions, kv_positions) where:
            - q_positions: (batch, qlen) position indices for queries, or None if dimensions unknown
            - kv_positions: (batch, kvlen) position indices for keys/values, or None if dimensions unknown

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> q_pos, kv_pos = mask_info.get_or_compute_positions()
            >>> q_pos.shape
            (1, 4)
            >>> kv_pos[0]
            Array([0, 1, 2, 3], dtype=int32)
        """

        q_positions = self.q_positions
        kv_positions = self.kv_positions

        need_q = q_positions is None
        need_kv = kv_positions is None

        if not (need_q or need_kv):
            return q_positions, kv_positions

        if self._q_segment_ids is None or self._kv_segment_ids is None:
            if self._attention_mask is None:
                return q_positions, kv_positions
            self.get_or_compute_segment_ids(per_head=False)

        if need_q and self._q_segment_ids is not None:
            q_positions = _positions_from_segments_2d(self._q_segment_ids, pad_value=-1)

        if need_kv and self._kv_segment_ids is not None:
            kv_pad = jnp.iinfo(jnp.int32).max
            kv_positions = _positions_from_segments_2d(self._kv_segment_ids, pad_value=kv_pad)

        self.q_positions = q_positions
        self.kv_positions = kv_positions
        return self.q_positions, self.kv_positions

    def get_or_compute_attention_mask(self, dtype: DTypeLike = jnp.bool_) -> Array:
        """
        Get attention mask, always computing from segment IDs when available.

        Prioritizes segment IDs as the source of truth - if segment IDs are available,
        the attention mask is always generated from them rather than using a cached version.
        This ensures consistency and avoids stale mask data.

        Args:
            dtype: Desired output dtype (default: bool)

        Returns:
            Attention mask array

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """

        if self._attention_mask is not None:
            return self._attention_mask.astype(dtype)
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            self._attention_mask = segment_ids_to_mask((self._q_segment_ids, self._kv_segment_ids), dtype=dtype)
            return self._attention_mask
        raise ValueError(
            "Cannot compute attention mask: MaskInfo is empty (both attention_mask and segment_ids are None). "
            "Initialize MaskInfo with either an attention_mask or segment "
            "IDs using from_attention_mask() or from_segments()."
        )

    def get_or_compute_segment_ids(self, per_head: bool = False) -> tuple[Int[Array, "..."], Int[Array, "..."]]:
        """
        Get segment IDs, computing from attention mask if not available.

        Args:
            per_head: If True and attention mask is 4D, compute segment IDs per head

        Returns:
            Tuple of (q_segment_ids, kv_segment_ids)

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """
        if self._q_segment_ids is not None and self._kv_segment_ids is not None:
            return self._q_segment_ids, self._kv_segment_ids
        if self._attention_mask is not None:
            self._q_segment_ids, self._kv_segment_ids = mask_to_segment_ids(self._attention_mask, per_head=per_head)
            return self._q_segment_ids, self._kv_segment_ids
        raise ValueError(
            "Cannot compute segment IDs: MaskInfo is empty (both attention_mask and segment_ids are None). "
            "Initialize MaskInfo with either an attention_mask or segment IDs using "
            "from_attention_mask() or from_segments()."
        )

    def get_qkv_masks(
        self, dtype: DTypeLike = jnp.bool_
    ) -> tuple[
        Array,
        Array,
        Bool[Array, "batch nheads_or_1 qlen kvlen"] | Int[Array, "batch nheads_or_1 qlen kvlen"],
    ]:
        """
        Get separate query mask, key-value mask, and attention mask.

        Args:
            dtype: Desired output dtype (default: bool)

        Returns:
            Tuple of (q_mask, kv_mask, attention_mask) where:
            - q_mask: (batch, qlen) boolean mask for valid query positions
            - kv_mask: (batch, kvlen) boolean mask for valid key-value positions
            - attention_mask: (batch, 1, qlen, kvlen) 4D pairwise attention mask

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """
        q_ids, kv_ids = self.get_or_compute_segment_ids()
        return segment_ids_to_qkv_masks(q_ids, kv_ids, dtype=dtype)

    def is_self_attention(self) -> bool:
        """
        Check if this represents self-attention (same query and key-value sequences).

        Returns:
            True if query and key-value sequences are identical, False otherwise
        """
        if self.q_segment_ids is not None and self.kv_segment_ids is not None:
            return self.q_segment_ids.shape == self.kv_segment_ids.shape and jnp.array_equal(
                self.q_segment_ids, self.kv_segment_ids
            )

        if self.attention_mask is not None:
            shape = self.attention_mask.shape
            return shape[-2] == shape[-1]

        return False

    def to_dtype(self, dtype: DTypeLike) -> "MaskInfo":
        """
        Convert attention mask to specified dtype, returning a new MaskInfo.

        Args:
            dtype: Target dtype (e.g., jnp.float32, jnp.bool_)

        Returns:
            New MaskInfo with converted attention mask
        """
        if self.attention_mask is None:
            return self

        return MaskInfo(
            _attention_mask=self.attention_mask.astype(dtype),
            _q_segment_ids=self._q_segment_ids,
            _kv_segment_ids=self._kv_segment_ids,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    @property
    def batch_size(self) -> int | None:
        """
        Get batch size from available data.

        Infers the batch dimension from either segment IDs or attention mask.

        Returns:
            Batch size if available, None otherwise
        """
        if self.q_segment_ids is not None:
            return self.q_segment_ids.shape[0]
        if self.attention_mask is not None:
            return self.attention_mask.shape[0]
        return None

    @property
    def q_len(self) -> int | None:
        """
        Get query sequence length.

        Infers the query sequence dimension from either segment IDs or attention mask.

        Returns:
            Query sequence length if available, None otherwise
        """
        if self.q_segment_ids is not None:
            return self.q_segment_ids.shape[-1]
        if self.attention_mask is not None:
            return self.attention_mask.shape[-2]
        return None

    @property
    def kv_len(self) -> int | None:
        """
        Get key-value sequence length.

        Infers the key-value sequence dimension from either segment IDs or attention mask.

        Returns:
            Key-value sequence length if available, None otherwise
        """
        if self.kv_segment_ids is not None:
            return self.kv_segment_ids.shape[-1]
        if self.attention_mask is not None:
            return self.attention_mask.shape[-1]
        return None

    @property
    def shape(self) -> tuple[int | None, int | None, int | None]:
        """
        Get (batch_size, q_len, kv_len) shape tuple.

        Convenience property that returns all three dimensions at once.

        Returns:
            Tuple of (batch_size, query_length, key_value_length)
        """
        return (self.batch_size, self.q_len, self.kv_len)

    def apply_kv_lengths(
        self,
        kv_lengths: Int[Array, "batch"],
        *,
        q_len: int | None = None,
        end_index: Int[Array, "batch"] | None = None,
        start_index: Int[Array, "batch"] | None = None,
        clamp: bool = True,
        update_segment_ids: bool = True,
    ) -> "MaskInfo":
        """
        Apply per-batch KV sequence lengths and optionally slice query dimension.

        This method is useful for incremental decoding and variable-length sequences where
        different batch elements have different numbers of valid KV tokens. It masks out
        KV positions beyond each batch's valid length and can optionally slice the query
        dimension to a window for efficient incremental attention.

        Args:
            kv_lengths: Integer array of shape (batch,) specifying the number of valid KV tokens
                per batch element. KV positions at indices [0, kv_lengths[b]) are kept valid,
                while positions >= kv_lengths[b] are masked out.
            q_len: If specified, slices the query dimension to this length. Requires either
                end_index or start_index to determine which query rows to keep.
            end_index: Array of shape (batch,) specifying the end position for query slicing.
                When provided with q_len, keeps query rows [end_index[b] - q_len, end_index[b])
                for each batch element. Useful for incremental decoding where you want the
                last q_len query positions.
            start_index: Array of shape (batch,) specifying the start position for query slicing.
                When provided with q_len, keeps query rows [start_index[b], start_index[b] + q_len)
                for each batch element. Alternative to end_index.
            clamp: If True, clamps all indices and lengths to valid ranges to prevent out-of-bounds
                errors. Recommended for safety. Default: True
            update_segment_ids: If True, updates segment IDs to reflect masking:
                - Sets kv_segment_ids[b, t] = -1 for t >= kv_lengths[b]
                - Slices q_segment_ids if q_len is provided
                - Creates q_segment_ids (filled with 0s) if missing and q slicing occurs
                Default: True

        Returns:
            New MaskInfo with:
            - attention_mask: Shape (batch, 1, q_len or Q_total, K_total) with KV positions
              beyond kv_lengths masked out and optional query slicing applied
            - kv_segment_ids: Updated with -1 for positions >= kv_lengths[b]
            - q_segment_ids: Sliced if q_len provided, or created if missing

        Raises:
            ValueError: If attention mask is not 4D or if q_len is provided without position indices

        Example:
            >>>
            >>> mask_info = MaskInfo.from_segments(jnp.ones((2, 128), dtype=jnp.int32))
            >>> kv_lengths = jnp.array([100, 80])
            >>> current_pos = jnp.array([100, 80])
            >>> new_mask = mask_info.apply_kv_lengths(
            ...     kv_lengths=kv_lengths,
            ...     q_len=1,
            ...     end_index=current_pos
            ... )
            >>> new_mask.attention_mask.shape
            (2, 1, 1, 128)

        Notes:
            - This is particularly useful for KV caching in autoregressive generation
            - The query slicing supports both "last N positions" (via end_index) and
              "positions starting at offset" (via start_index) patterns
            - Segment IDs are updated to maintain consistency with the masked regions
        """

        base_mask = (
            self.attention_mask.astype(jnp.bool_)
            if self.attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )
        if base_mask.ndim == 3:
            base_mask = base_mask[:, None, :, :]
        if base_mask.ndim != 4:
            raise ValueError(
                f"Expected 4D attention mask with shape (batch, heads, q_len, kv_len), "
                f"but got {base_mask.ndim}D mask with shape {base_mask.shape}. "
                f"Ensure the mask is properly formatted before applying KV lengths."
            )
        B, _H, Q_total, K_total = base_mask.shape

        kv_lengths = jnp.asarray(kv_lengths, jnp.int32).reshape(B)
        if clamp:
            kv_lengths = jnp.clip(kv_lengths, 0, K_total)

        sliced_mask = base_mask
        q_seg = self._q_segment_ids
        if q_len is not None:
            if end_index is None and start_index is None:
                raise ValueError(
                    "Query slicing requires position information. When q_len is specified, "
                    "you must provide either 'end_index' or 'start_index' to determine which query rows to slice. "
                    "For example: end_index=current_position or start_index=offset."
                )

            if end_index is not None:
                end_index = jnp.asarray(end_index, jnp.int32).reshape(B)
                q_start = end_index - jnp.int32(q_len)
            else:
                start_index = jnp.asarray(start_index, jnp.int32).reshape(B)
                q_start = start_index

            if clamp:
                q_start = jnp.clip(q_start, 0, jnp.maximum(0, Q_total - jnp.int32(q_len)))

            def _slice_q_per_batch(m_b, s_b):
                return jax.lax.dynamic_slice(m_b, (0, s_b, 0), (m_b.shape[0], q_len, m_b.shape[2]))

            sliced_mask = jax.vmap(_slice_q_per_batch, in_axes=(0, 0), out_axes=0)(base_mask, q_start)

            if q_seg is not None:
                q_seg = jnp.asarray(q_seg, jnp.int32)

                def _slice_ids_row(ids_b, s_b):
                    return jax.lax.dynamic_slice_in_dim(ids_b, s_b, q_len, axis=0)

                q_seg = jax.vmap(_slice_ids_row, in_axes=(0, 0), out_axes=0)(q_seg, q_start)
            elif update_segment_ids:
                q_seg = jnp.zeros((B, q_len), jnp.int32)

        kv_idx = jnp.arange(K_total, dtype=jnp.int32)
        kv_valid = kv_idx[None, :] < kv_lengths[:, None]
        kv_mask4d = kv_valid[:, None, None, :]
        attention_mask = sliced_mask & kv_mask4d

        if update_segment_ids:
            if self._kv_segment_ids is not None:
                kv_seg = jnp.asarray(self._kv_segment_ids, jnp.int32)
                kv_segment_ids = jnp.where(kv_valid, kv_seg, jnp.int32(-1))
            else:
                kv_segment_ids = jnp.where(kv_valid, jnp.int32(0), jnp.int32(-1))
        else:
            kv_segment_ids = self._kv_segment_ids

        return MaskInfo(
            _attention_mask=attention_mask,
            _q_segment_ids=q_seg if q_len is not None else self._q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    def apply_causal(self, offset: int = 0) -> "MaskInfo":
        """
        Apply causal (autoregressive) masking to the attention pattern.

        Restricts attention so that each query position can only attend to
        key positions at or before its own position (plus an optional offset).
        The segment IDs are preserved to maintain grouping structure.

        Args:
            offset: Position offset for causal masking. Default: 0
                - offset=0: Standard causal (q_i attends to kv_j where j <= i)
                - offset>0: Allows attending to future tokens (j <= i + offset)
                - offset<0: More restrictive causal (j <= i + offset)

        Returns:
            New MaskInfo with causal constraint applied while preserving segment IDs

        Raises:
            ValueError: If mask dimensions are unknown

        Example:
            >>> segment_ids = jnp.array([[1, 1, 1, 1]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>> causal_mask = mask_info.apply_causal()
            >>>
        """
        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply causal mask: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        q_seg, kv_seg = self._q_segment_ids, self._kv_segment_ids

        base_mask = (
            self._attention_mask.astype(jnp.bool_)
            if self._attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )
        Q, K = base_mask.shape[-2], base_mask.shape[-1]
        q_idx = jnp.arange(Q, dtype=jnp.int32)
        kv_idx = jnp.arange(K, dtype=jnp.int32)
        causal = (q_idx[:, None] + offset >= kv_idx[None, :])[None, None, :, :]
        return MaskInfo(
            _attention_mask=base_mask & causal,
            _q_segment_ids=q_seg,
            _kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    def apply_sliding_window(
        self,
        window_size: int | tuple[int, int] | tuple[int | None, int | None],
        offset: int = 0,
    ) -> "MaskInfo":
        """
        Apply sliding window (local) attention by preserving segment IDs and applying window constraint.

        Restricts attention to a local window around each query position. The segment IDs
        are preserved to maintain the original grouping structure, while the attention mask
        encodes both the segment constraint AND the sliding window constraint.

        Args:
            window_size: Size of the attention window. Can be:
                - int: Symmetric window of size (window_size, window_size)
                - tuple[int, int]: Asymmetric window (left_size, right_size)
                - tuple[int|None, int|None]: One-sided window (None means unlimited)
            offset: Offset of q start wrt kv (same as causal mask offset)

        Returns:
            New MaskInfo with sliding window constraint applied

        Example:
            >>> segment_ids = jnp.array([[1, 1, 1, 1, 1]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>>
            >>> windowed = mask_info.apply_sliding_window(window_size=(1, 1))
        """
        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply sliding window: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        if isinstance(window_size, int):
            left, right = window_size, window_size
        else:
            left, right = window_size

        q_seg, kv_seg = self._q_segment_ids, self._kv_segment_ids
        base_mask = (
            self._attention_mask.astype(jnp.bool_)
            if self._attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )

        Q, K = base_mask.shape[-2], base_mask.shape[-1]
        q_idx = jnp.arange(Q, dtype=jnp.int32)
        kv_idx = jnp.arange(K, dtype=jnp.int32)

        local_mask = jnp.ones((Q, K), dtype=jnp.bool_)
        if left is not None:
            local_mask = local_mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
        if right is not None:
            local_mask = local_mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])

        attention_mask = base_mask & local_mask[None, None, :, :]

        return MaskInfo(
            _attention_mask=attention_mask,
            _q_segment_ids=q_seg,
            _kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    def apply_chunked(self, chunk_size: int, offset: int = 0) -> "MaskInfo":
        """
        Apply chunked causal attention and ALWAYS update q/kv segment IDs to chunk IDs.

        - New segment IDs are the chunk indices + 1 (padding stays -1)
        - Attention mask becomes: existing_mask AND (same_chunk AND causal)
        - This makes segment IDs the canonical representation of chunk structure.
        Note: segment IDs encode chunk grouping; causal direction still requires positions/rule.

        Args:
            chunk_size: Positive chunk size.
            offset: Optional causal offset (default 0).

        Returns:
            New MaskInfo with updated attention_mask and updated segment IDs.
        """
        if chunk_size <= 0:
            raise ValueError(
                f"Invalid chunk_size: expected a positive integer, but got {chunk_size}. "
                f"Chunk size must be greater than 0 to define valid chunks."
            )
        if self.q_len is None or self.kv_len is None:
            raise ValueError(
                "Cannot apply chunked attention: mask dimensions are unknown. "
                "The MaskInfo instance must have defined q_len and kv_len "
                "(available via attention_mask or segment_ids). "
                f"Current state: q_len={self.q_len}, kv_len={self.kv_len}"
            )

        base_mask = (
            self.attention_mask.astype(jnp.bool_)
            if self.attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )

        q_idx = jnp.arange(self.q_len, dtype=jnp.int32)
        kv_idx = jnp.arange(self.kv_len, dtype=jnp.int32)

        same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
        causal = kv_idx[None, :] <= (q_idx[:, None] + offset)
        chunked_4d = (same_chunk & causal)[None, None, :, :]

        attention_mask = base_mask & chunked_4d

        try:
            q_seg_cur, kv_seg_cur = self.get_or_compute_segment_ids()
            q_pad = q_seg_cur < 0
            kv_pad = kv_seg_cur < 0
        except Exception:
            q_valid = jnp.any(base_mask, axis=(1, 3))
            kv_valid = jnp.any(base_mask, axis=(1, 2))
            q_pad = ~q_valid
            kv_pad = ~kv_valid

        q_chunk_ids = (q_idx // chunk_size).astype(jnp.int32)[None, :]
        kv_chunk_ids = (kv_idx // chunk_size).astype(jnp.int32)[None, :]

        q_segment_ids = jnp.where(q_pad, -1, q_chunk_ids)
        kv_segment_ids = jnp.where(kv_pad, -1, kv_chunk_ids)

        return MaskInfo(
            _attention_mask=attention_mask,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    def apply_token_type_ids(
        self,
        token_type_ids: Int[Array, "batch q_len"] | tuple[Int[Array, "batch q_len"], Int[Array, "batch kv_len"]],
        *,
        combine: Literal["union", "intersect", "replace"] = "union",
        zero_policy: Literal["q", "kv", "both", "none"] = "q",
        update_segment_ids: bool | None = None,
    ) -> "MaskInfo":
        """
        Integrate token_type_ids into the attention pattern.

        - Builds an equality mask between q and kv token types.
        - Optionally treats token_type_id == 0 as "disabled" (no token-type matching)
        on the query side, kv side, both, or neither (zero_policy).
        - Combines with the current attention mask by union/intersect/replace.
        - Optionally updates segment IDs to reflect token types (0 -> -1 padding).

        Args:
            token_type_ids:
                - self-attn: (batch, q_len)
                - cross-attn: (q_token_type_ids, kv_token_type_ids)
            combine: How to combine with existing mask:
                - "union": base_mask OR token_type_mask   (matches your old snippet)
                - "intersect": base_mask AND token_type_mask
                - "replace": token_type_mask only
            zero_policy:
                - "q": treat q==0 as disabled (no token-type matching for those queries) [matches old code]
                - "kv": treat kv==0 as disabled (no matching into those keys/values)
                - "both": treat 0 as disabled on both sides
                - "none": don't treat 0 specially
            update_segment_ids:
                - If None: defaults to False for "union" (cannot encode union in seg-ids),
                and True for "intersect"/"replace".
                - If True: set q/kv segment IDs from token types with 0 -> -1.
                - If False: keep existing segment IDs.

        Returns:
            New MaskInfo with updated attention_mask (and optionally updated segment IDs).
        """
        if isinstance(token_type_ids, tuple):
            q_types, kv_types = token_type_ids
        else:
            q_types = token_type_ids
            kv_types = token_type_ids
        if q_types.ndim != 2 or kv_types.ndim != 2:
            raise ValueError(
                f"Invalid token_type_ids shape. Expected 2D arrays with shape (batch, seq_len), "
                f"but got q_types.shape={q_types.shape} ({q_types.ndim}D) and "
                f"kv_types.shape={kv_types.shape} ({kv_types.ndim}D). "
                f"For self-attention, pass a single (batch, seq_len) array. "
                f"For cross-attention, pass a tuple of ((batch, q_len), (batch, kv_len))."
            )

        q_types = jnp.asarray(q_types, jnp.int32)
        kv_types = jnp.asarray(kv_types, jnp.int32)

        Bq, _Q = q_types.shape
        Bk, _K = kv_types.shape
        if Bq != Bk:
            raise ValueError(
                f"Batch size mismatch in token_type_ids. Query token types have batch size {Bq}, "
                f"but key-value token types have batch size {Bk}. "
                f"Both must have the same batch dimension. "
                f"Shapes: q_types={q_types.shape}, kv_types={kv_types.shape}"
            )

        base_mask = (
            self.attention_mask.astype(jnp.bool_)
            if self.attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )

        eq2d = q_types[:, :, None] == kv_types[:, None, :]
        if zero_policy not in ("q", "kv", "both", "none"):
            raise ValueError(
                f"Invalid zero_policy value. Expected one of ['q', 'kv', 'both', 'none'], "
                f"but got '{zero_policy}'. The zero_policy determines how token_type_id=0 is treated: "
                f"'q'=disable on queries, 'kv'=disable on keys/values, "
                f"'both'=disable on both sides, 'none'=no special treatment."
            )

        if zero_policy in ("q", "both"):
            q_valid = (q_types != 0)[:, :, None]
            eq2d = eq2d & q_valid
        if zero_policy in ("kv", "both"):
            kv_valid = (kv_types != 0)[:, None, :]
            eq2d = eq2d & kv_valid

        eq4d = eq2d[:, None, :, :].astype(jnp.bool_)

        if combine == "union":
            new_mask = base_mask | eq4d
        elif combine == "intersect":
            new_mask = base_mask & eq4d
        elif combine == "replace":
            new_mask = eq4d
        else:
            raise ValueError(
                f"Invalid combine mode. Expected one of ['union', 'intersect', 'replace'], "
                f"but got '{combine}'. The combine mode determines how token types interact with the existing mask: "
                f"'union'=base_mask OR token_type_mask, 'intersect'=base_mask AND token_type_mask, "
                f"'replace'=token_type_mask only."
            )

        if update_segment_ids is None:
            update_segment_ids = combine != "union"

        if update_segment_ids:
            q_seg = jnp.where(q_types == 0, jnp.array(-1, q_types.dtype), q_types)
            kv_seg = jnp.where(kv_types == 0, jnp.array(-1, kv_types.dtype), kv_types)
            q_seg = q_seg.astype(jnp.int32)
            kv_seg = kv_seg.astype(jnp.int32)
        else:
            q_seg = self._q_segment_ids
            kv_seg = self._kv_segment_id

        return MaskInfo(
            _attention_mask=new_mask,
            _q_segment_ids=q_seg,
            _kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
            batch_axis_name=self.batch_axis_name,
            qheads_axis_name=self.qheads_axis_name,
            kvheads_axis_name=self.kvheads_axis_name,
            sequence_axis_name=self.sequence_axis_name,
        )

    @staticmethod
    def create_chunked_attention_mask(
        chunk_size: int,
        q_len: int,
        kv_len: int | None = None,
        offset: int = 0,
        dtype=jnp.bool_,
    ) -> jnp.ndarray:
        """
        Create a chunked causal attention mask (static method).

        Generates a 2D attention mask where attention is restricted to tokens
        within the same chunk, with causal ordering enforced within chunks.

        Args:
            chunk_size: Size of each chunk (must be positive)
            q_len: Query sequence length
            kv_len: Key-value sequence length. If None, uses q_len
            offset: Causal offset. Default: 0
            dtype: Output dtype. Default: jnp.bool_

        Returns:
            2D attention mask of shape (q_len, kv_len) with chunked causal pattern

        Raises:
            ValueError: If chunk_size is not positive

        Example:
            >>> mask = MaskInfo.create_chunked_attention_mask(
            ...     chunk_size=4, q_len=8, kv_len=8
            ... )
            >>> mask.shape
            (8, 8)
        """
        if chunk_size <= 0:
            raise ValueError(
                f"Invalid chunk_size: expected a positive integer, but got {chunk_size}. "
                f"Chunk size must be greater than 0."
            )
        if kv_len is None:
            kv_len = q_len
        q_idx = jnp.arange(q_len, dtype=jnp.int32)
        kv_idx = jnp.arange(kv_len, dtype=jnp.int32)
        same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
        causal = kv_idx[None, :] <= (q_idx[:, None] + offset)
        return (same_chunk & causal).astype(dtype)

    def __repr__(self) -> str:
        """
        Enhanced string representation with shape information.

        Returns:
            Human-readable string describing the MaskInfo contents and dimensions
        """
        parts = ["MaskInfo("]
        if self.attention_mask is not None:
            parts.append(f"attention_mask.shape={self.attention_mask.shape}")
        if self.q_segment_ids is not None:
            parts.append(f"q_segment_ids.shape={self.q_segment_ids.shape}")
        if self.kv_segment_ids is not None:
            parts.append(f"kv_segment_ids.shape={self.kv_segment_ids.shape}")
        parts.append(f"self_attn={self.is_self_attention()})")
        return ", ".join(parts)

    def tree_flatten(self):
        """
        Flatten MaskInfo for JAX pytree registration.

        This method is required for JAX pytree support, enabling MaskInfo instances to be used
        seamlessly in JAX transformations (jit, vmap, grad, etc.). It separates the instance
        into two parts:
        - Children: Array fields that should be traced and transformed by JAX
        - Aux data: Static metadata that remains constant across transformations

        Returns:
            Tuple of (children, aux_data) where:
            - children: Tuple of (attention_mask, q_segment_ids, kv_segment_ids, q_positions, kv_positions)
            - aux_data: Tuple of (batch_axis_name, qheads_axis_name, kvheads_axis_name, sequence_axis_name)

        Notes:
            - This method is automatically called by JAX during pytree operations
            - Users typically don't need to call this directly
            - The counterpart tree_unflatten() reconstructs the MaskInfo from flattened form
        """

        children = (
            self._attention_mask,
            self._q_segment_ids,
            self._kv_segment_ids,
            self.q_positions,
            self.kv_positions,
        )

        aux_data = (
            self.batch_axis_name,
            self.qheads_axis_name,
            self.kvheads_axis_name,
            self.sequence_axis_name,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstruct MaskInfo from flattened pytree representation.

        This method is the inverse of tree_flatten() and is required for JAX pytree support.
        It reconstructs a MaskInfo instance from its flattened components after JAX
        transformations have been applied.

        Args:
            aux_data: Static metadata tuple containing
                (batch_axis_name, qheads_axis_name, kvheads_axis_name, sequence_axis_name)
            children: Traced array tuple containing
                (attention_mask, q_segment_ids, kv_segment_ids, q_positions, kv_positions)

        Returns:
            Reconstructed MaskInfo instance with the provided arrays and metadata

        Notes:
            - This method is automatically called by JAX during pytree operations
            - Users typically don't need to call this directly
            - The method signature must match the output format of tree_flatten()
        """
        attention_mask, q_segment_ids, kv_segment_ids, q_positions, kv_positions = children
        batch_axis_name, qheads_axis_name, kvheads_axis_name, sequence_axis_name = aux_data
        return cls(
            _attention_mask=attention_mask,
            _q_segment_ids=q_segment_ids,
            _kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )

    def replace(self, *, attention_mask=None, q_segment_ids=None, kv_segment_ids=None, **kw) -> "MaskInfo":
        """
        Create a new MaskInfo with specified fields replaced.

        This is a convenience method for creating modified copies of MaskInfo instances,
        similar to dataclasses.replace(). Only specified fields are updated; others are
        preserved from the original instance.

        Args:
            attention_mask: New attention mask array, or None to keep existing
            q_segment_ids: New query segment IDs, or None to keep existing
            kv_segment_ids: New key-value segment IDs, or None to keep existing
            **kw: Additional keyword arguments for other fields:
                - q_positions: New query positions
                - kv_positions: New key-value positions
                - batch_axis_name: New batch axis name(s)
                - qheads_axis_name: New query heads axis name(s)
                - kvheads_axis_name: New key-value heads axis name(s)
                - sequence_axis_name: New sequence axis name(s)

        Returns:
            New MaskInfo instance with specified fields replaced

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.array([[1, 1, 2, 2]]))
            >>> new_mask_info = mask_info.replace(batch_axis_name="data")
            >>> new_mask_info.batch_axis_name
            'data'
        """
        return MaskInfo(
            _attention_mask=attention_mask if attention_mask is not None else self._attention_mask,
            _q_segment_ids=q_segment_ids if q_segment_ids is not None else self._q_segment_ids,
            _kv_segment_ids=kv_segment_ids if kv_segment_ids is not None else self._kv_segment_ids,
            q_positions=kw.get("q_positions", self.q_positions),
            kv_positions=kw.get("kv_positions", self.kv_positions),
            batch_axis_name=kw.get("batch_axis_name", self.batch_axis_name),
            qheads_axis_name=kw.get("qheads_axis_name", self.qheads_axis_name),
            kvheads_axis_name=kw.get("kvheads_axis_name", self.kvheads_axis_name),
            sequence_axis_name=kw.get("sequence_axis_name", self.sequence_axis_name),
        )

    @classmethod
    def dynamic_init(
        cls,
        *,
        mask_info: MaskInfo | None = None,
        input_ids: Float[Array, "batch seglen"] | None = None,
        inputs_embeds: Float[Array, "batch seqlen dim"] | None = None,
        attention_mask: Float[Array, "batch seglen"] | Bool[Array, "batch seglen"] | None = None,
    ) -> MaskInfo:
        """
        Dynamically initialize a MaskInfo from various input sources.

        This is a convenience factory method that creates a MaskInfo instance from different
        types of inputs commonly available in transformer models. It prioritizes existing
        mask_info, then constructs one from attention_mask or input shapes.

        Args:
            mask_info: Pre-existing MaskInfo to return as-is. If provided, other arguments are ignored.
            input_ids: Token IDs array with shape (batch, seq_len). Used to infer shape if mask_info
                and attention_mask are not provided.
            inputs_embeds: Token embeddings array with shape (batch, seq_len, dim). Used to infer shape
                if mask_info, attention_mask, and input_ids are not provided.
            attention_mask: Attention mask array with shape (batch, seq_len). Values should be:
                - 1/True for valid (non-padding) tokens
                - 0/False for padding tokens
                If not provided, creates an all-ones mask (no padding).

        Returns:
            MaskInfo instance constructed from the provided inputs

        Raises:
            ValueError: If insufficient information is provided (no valid inputs)

        Example:
            >>> input_ids = jnp.array([[1, 2, 3, 0], [4, 5, 0, 0]])
            >>> attn_mask = jnp.array([[1, 1, 1, 0], [1, 1, 0, 0]])
            >>> mask_info = MaskInfo.dynamic_init(input_ids=input_ids, attention_mask=attn_mask)
            >>> mask_info.shape
            (2, 4, 4)

        Notes:
            - This method is useful for model implementations where mask format may vary
            - Automatically converts 2D attention masks to segment-based representation
            - Higher-dimensional masks are handled via from_attention_mask()
        """
        if mask_info is not None:
            return mask_info

        if attention_mask is None:
            if input_ids is not None:
                batch_size, sequence_length = input_ids.shape
            elif inputs_embeds is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape
            else:
                raise ValueError(
                    "Cannot create MaskInfo: insufficient information provided. "
                    "You must provide at least one of: mask_info, input_ids, inputs_embeds, or attention_mask. "
                    "These are needed to determine the batch size and sequence length."
                )
            attention_mask = jnp.ones((batch_size, sequence_length), "b1")
        else:
            if attention_mask.dtype != jnp.bool:
                attention_mask = jnp.astype(attention_mask == 1, "b1")
        if attention_mask.ndim == 2:
            mask_info = MaskInfo.from_segments(attention_mask)
        else:
            mask_info = MaskInfo.from_attention_mask(attention_mask)
        return mask_info

    def visualize(
        self,
        block_size: int | tuple[int, int] = 32,
        batch: int = 0,
        head: int = 0,
        fit_in_screen: bool = True,
        max_rows: int = 32,
        max_cols: int = 64,
        charset: Literal["unicode", "ascii"] = "unicode",
        show_segments: bool = True,
        return_str: bool = False,
    ) -> str | None:
        """
        Pretty-print the attention mask as block-aggregated ASCII/Unicode visualization.

        Optionally shows aggregated query/key-value segment IDs for each block row/column.
        Useful for debugging and understanding attention patterns.

        Args:
            block_size: Size of aggregation blocks. Can be:
                - int: Square blocks of size (block_size, block_size)
                - tuple[int, int]: Rectangular blocks (q_block_size, kv_block_size)
            batch: Batch index to visualize. Default: 0
            head: Head index to visualize. Default: 0
            fit_in_screen: If True, downsample to fit within max_rows/max_cols. Default: True
            max_rows: Maximum number of block rows to display when fit_in_screen=True. Default: 32
            max_cols: Maximum number of block columns to display when fit_in_screen=True. Default: 64
            charset: Character set for visualization. Default: "unicode"
                - "unicode": Uses box-drawing characters (░░ for partial, ██ for full)
                - "ascii": Uses ASCII characters (.. for partial,
            show_segments: If True, display segment IDs alongside the mask. Default: True
            return_str: If True, return the visualization as a string instead of printing. Default: False

        Returns:
            If return_str=True, returns the visualization string. Otherwise, prints and returns None

        Block encoding:
            - Empty (no attention): "  " (spaces)
            - Partial (some attention): "░░" (unicode) or ".." (ascii)
            - Full (all attention): "██" (unicode) or "##" (ascii)

        Segment ID display:
            - If all tokens in a block share the same segment ID: shows that ID
            - Mixed segments: shown as "??" in header, "MIX" on left
            - Padding: shown as -1 or "PAD"

        Notes:
            - Not JIT-friendly; runs on host (uses numpy and prints)
            - Segment IDs are taken from self.q_segment_ids/self.kv_segment_ids if present,
              otherwise computed from the mask (may be per-head if H > 1)

        Example:
            >>> mask_info = MaskInfo.from_segments(jnp.ones((2, 128), dtype=jnp.int32))
            >>> mask_info.visualize(block_size=16, batch=0)
        """

        m = (
            self.attention_mask.astype(jnp.bool_)
            if self.attention_mask is not None
            else self.get_or_compute_attention_mask(dtype=jnp.bool_)
        )
        if m.ndim == 3:
            m = m[:, None, :, :]
        if m.ndim != 4:
            raise ValueError(
                f"Visualization requires a 4D attention mask with shape (batch, heads, q_len, kv_len), "
                f"but got {m.ndim}D mask with shape {m.shape}. "
                f"Ensure the mask is properly formatted before visualization."
            )

        B, H, _Q, _K = m.shape
        if not (0 <= batch < B):
            raise IndexError(
                f"Batch index out of range. Requested batch={batch}, but valid range is [0, {B}). "
                f"The attention mask has batch_size={B}."
            )
        if not (0 <= head < H):
            raise IndexError(
                f"Head index out of range. Requested head={head}, but valid range is [0, {H}). "
                f"The attention mask has {H} head(s)."
            )

        attn_2d = jax.device_get(m[batch, head])
        if isinstance(block_size, int):
            q_block = kv_block = block_size
        else:
            q_block, kv_block = block_size
        q_block = max(int(q_block), 1)
        kv_block = max(int(kv_block), 1)

        def _block_classify(arr_bool: np.ndarray, r_b: int, c_b: int) -> tuple[np.ndarray, np.ndarray]:
            Q, K = arr_bool.shape
            pad_q = (-Q) % r_b
            pad_k = (-K) % c_b
            if pad_q or pad_k:
                arr_bool = np.pad(arr_bool, ((0, pad_q), (0, pad_k)), constant_values=False)
            nrb = arr_bool.shape[0] // r_b
            ncb = arr_bool.shape[1] // c_b
            reshaped = arr_bool.reshape(nrb, r_b, ncb, c_b)
            counts = reshaped.sum(axis=(1, 3)).astype(np.float32)
            area = float(r_b * c_b)
            ratio = counts / area
            eps = 1e-8
            full = ratio >= 1.0 - eps
            empty = ratio <= eps
            cls = np.where(full, 2, np.where(empty, 0, 1)).astype(np.int32)
            return cls, ratio

        def _segment_block_labels(ids_1d: np.ndarray, block: int) -> np.ndarray:
            pad = (-len(ids_1d)) % block
            if pad:
                ids_1d = np.pad(ids_1d, (0, pad), constant_values=-1)
            nb = ids_1d.shape[0] // block
            blk = ids_1d.reshape(nb, block)
            same = np.all(blk == blk[:, :1], axis=1)
            labels = np.where(same, blk[:, 0], -2)
            return labels

        def _downsample_labels(labels: np.ndarray, step: int) -> np.ndarray:
            if step <= 1:
                return labels
            pad = (-len(labels)) % step
            if pad:
                labels = np.pad(labels, (0, pad), constant_values=-1)
            n = labels.shape[0] // step
            grp = labels.reshape(n, step)

            same = np.all(grp == grp[:, :1], axis=1)
            out = np.where(same, grp[:, 0], -2)
            return out

        def _two_char(label: int) -> str:
            if label == -1:
                return "  "
            if label == -2:
                return "??"
            return f"{int(label) % 100:02d}"

        def _left_label(label: int, width: int = 6) -> str:
            if label == -1:
                s = "PAD"
            elif label == -2:
                s = "MIX"
            else:
                s = str(int(label))
            return f"{s:>{width}}"

        cls, ratio = _block_classify(attn_2d, q_block, kv_block)
        block_rows, block_cols = cls.shape

        q_blk_labels = None
        kv_blk_labels = None
        if show_segments:
            if self.q_segment_ids is not None:
                q_ids_all = jax.device_get(self.q_segment_ids)
                if q_ids_all.ndim == 3:
                    q_ids = np.asarray(q_ids_all[batch, head])
                else:
                    q_ids = np.asarray(q_ids_all[batch])
            else:
                q_ids_all, _ = mask_to_segment_ids(m, per_head=(H > 1))
                q_ids_all = jax.device_get(q_ids_all)
                if q_ids_all.ndim == 3:
                    q_ids = np.asarray(q_ids_all[batch, head])
                else:
                    q_ids = np.asarray(q_ids_all[batch])

            if self.kv_segment_ids is not None:
                kv_ids_all = jax.device_get(self.kv_segment_ids)
                if kv_ids_all.ndim == 3:
                    kv_ids = np.asarray(kv_ids_all[batch, head])
                else:
                    kv_ids = np.asarray(kv_ids_all[batch])
            else:
                _, kv_ids_all = mask_to_segment_ids(m, per_head=(H > 1))
                kv_ids_all = jax.device_get(kv_ids_all)
                if kv_ids_all.ndim == 3:
                    kv_ids = np.asarray(kv_ids_all[batch, head])
                else:
                    kv_ids = np.asarray(kv_ids_all[batch])

            q_blk_labels = _segment_block_labels(q_ids, q_block)
            kv_blk_labels = _segment_block_labels(kv_ids, kv_block)

        if fit_in_screen:
            rows_step = int(np.maximum(np.ceil(block_rows / max_rows), 1))
            cols_step = int(np.maximum(np.ceil(block_cols / max_cols), 1))
            if rows_step > 1 or cols_step > 1:
                rpad = (-ratio.shape[0]) % rows_step
                cpad = (-ratio.shape[1]) % cols_step
                if rpad or cpad:
                    ratio = np.pad(ratio, ((0, rpad), (0, cpad)), constant_values=0.0)
                nr = ratio.shape[0] // rows_step
                nc = ratio.shape[1] // cols_step
                rsh = ratio.reshape(nr, rows_step, nc, cols_step)
                ratio = rsh.mean(axis=(1, 3))
                eps = 1e-6
                full = ratio >= 1.0 - eps
                empty = ratio <= eps
                cls = np.where(full, 2, np.where(empty, 0, 1)).astype(np.int32)
                block_rows, block_cols = cls.shape

                if show_segments and q_blk_labels is not None and kv_blk_labels is not None:
                    q_blk_labels = _downsample_labels(q_blk_labels, rows_step)
                    kv_blk_labels = _downsample_labels(kv_blk_labels, cols_step)

        block_chars = {
            0: "  ",
            1: (".." if charset == "ascii" else "░░"),
            2: ("##" if charset == "ascii" else "██"),
        }
        lines = ["".join(block_chars[int(v)] for v in cls[r]) for r in range(block_rows)]

        left_width = 6 if show_segments else 0
        width_cells = len(lines[0]) // 2 if lines else 0
        top_bot = "==" * width_cells
        header = f"Attention mask (batch={batch}, head={head}) block=({q_block}x{kv_block}) mask_shape={m.shape}\n"

        rendered = header

        if show_segments and kv_blk_labels is not None and block_cols > 0:
            kv_label_str = "".join(_two_char(int(lbl)) for lbl in kv_blk_labels)
            rendered += " " * (left_width + 3) + kv_label_str + "\n"

        rendered += " " * left_width + "  " + top_bot + "  \n"
        for r, line in enumerate(lines):
            if show_segments and q_blk_labels is not None:
                rendered += _left_label(int(q_blk_labels[r]), width=left_width) + " "
            else:
                rendered += " " * left_width
            rendered += "||" + line + "||\n"
        rendered += " " * left_width + "  " + top_bot + "  \n"

        legend_mask = "Legend mask: full='██'/##, partial='░░'/.., empty='  '"
        legend_seg = "Legend seg: left=Q block ID, top=KV block ID, PAD=-1, MIX=??"
        rendered += legend_mask + "\n"
        if show_segments:
            rendered += legend_seg + "\n"

        if return_str:
            return rendered
        else:
            print(rendered)
            return None


jax.tree_util.register_pytree_node(MaskInfo, MaskInfo.tree_flatten, MaskInfo.tree_unflatten)
