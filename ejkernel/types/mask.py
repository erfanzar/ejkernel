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


from dataclasses import dataclass, field
from typing import NamedTuple

import jax
from jax import numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jaxtyping import Array, Bool, DTypeLike, Int

from ejkernel.xla_utils import get_corrected_named_sharding

mdim_t = "batch nheads_or_1 qlen kvlen"


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

    # Always return 4D attention mask (batch, 1, q, kv)
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
        - Segment IDs of -1 or 0 are treated as padding
        - Positive segment IDs (1, 2, 3, ...) indicate different segments
        - Tokens can only attend within their own segment
        - For self-attention, q_mask and kv_mask will be identical
    """
    if kv_segment_ids is None:
        kv_segment_ids = q_segment_ids

    return segment_ids_to_mask((q_segment_ids, kv_segment_ids), dtype=dtype, return_separate_masks=True)


class MaskSharding(NamedTuple):
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
        q_segment_ids: Query segment IDs (batch, qlen) where -1 or 0 indicates padding
        kv_segment_ids: Key-value segment IDs (batch, kvlen) where -1 or 0 indicates padding
        q_positions: Query position indices (batch, qlen) for positional embeddings
        kv_positions: Key-value position indices (batch, kvlen) for positional embeddings
    """

    attention_mask: Bool[Array, mdim_t] | Int[Array, mdim_t] | None = None
    q_segment_ids: Int[Array, "batch qlen"] | None = None
    kv_segment_ids: Int[Array, "batch kvlen"] | None = None
    q_positions: Int[Array, "batch qlen"] | None = None
    kv_positions: Int[Array, "batch kvlen"] | None = None

    batch_axis_name: tuple[str] | str | None = field(default=("dp", "fsdp"))
    qheads_axis_name: tuple[str] | str | None = field(default="tp")
    kvheads_axis_name: tuple[str] | str | None = field(default="tp")
    sequence_axis_name: tuple[str] | str | None = field(default="sp")

    @classmethod
    def from_segments(
        cls,
        q_segment_ids: Int[Array, "batch qlen"],
        kv_segment_ids: Int[Array, "batch kvlen"] | None = None,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
    ):
        """
        Create MaskInfo from segment IDs.

        Args:
            q_segment_ids: Query segment IDs (batch, qlen)
            kv_segment_ids: Key-value segment IDs (batch, kvlen). If None, uses q_segment_ids.
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)

        Returns:
            MaskInfo with segment IDs, computed attention mask, and optional positions
        """
        if kv_segment_ids is None:
            kv_segment_ids = q_segment_ids

        return cls(
            attention_mask=segment_ids_to_mask((q_segment_ids, kv_segment_ids)),
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )

    @classmethod
    def from_attention_mask(
        cls,
        attention_mask: Bool[Array, mdim_t] | Int[Array, mdim_t],
        per_head: bool = False,
        q_positions: Int[Array, "batch qlen"] | None = None,
        kv_positions: Int[Array, "batch kvlen"] | None = None,
    ):
        """
        Create MaskInfo from an attention mask.

        Args:
            attention_mask: Attention mask. Can be:
                - 2D (B, Qlen): Padding mask indicating valid positions
                - 3D (B, Q, K): Pairwise attention mask
                - 4D (B, H, Q, K): Per-head pairwise attention mask
            per_head: If True and mask is 4D, compute segment IDs per head
            q_positions: Optional query position indices (batch, qlen)
            kv_positions: Optional key-value position indices (batch, kvlen)

        Returns:
            MaskInfo with attention mask, computed segment IDs, and optional positions
        """

        if attention_mask.ndim == 2:
            # 2D padding mask: treat as valid positions and convert to 4D attention mask
            q_segment_ids = jnp.where(attention_mask.astype(jnp.bool_), 1, -1)
            kv_segment_ids = q_segment_ids

            pairwise_mask = segment_ids_to_mask((q_segment_ids, kv_segment_ids))
            return cls(
                attention_mask=pairwise_mask,
                q_segment_ids=q_segment_ids,
                kv_segment_ids=kv_segment_ids,
                q_positions=q_positions,
                kv_positions=kv_positions,
            )

        # Ensure attention mask is 4D
        if attention_mask.ndim == 3:
            # 3D mask (batch, q, kv) -> 4D (batch, 1, q, kv)
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim != 4:
            raise ValueError(f"attention_mask must be 2D (padding), 3D, or 4D, got {attention_mask.ndim}D")

        q_segment_ids, kv_segment_ids = mask_to_segment_ids(attention_mask, per_head)
        return cls(
            attention_mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
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
    ):
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
            raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")

        key = jax.random.PRNGKey(seed)
        # Generate 4D mask (batch, 1, q, kv) for consistency
        random_mask = jax.random.bernoulli(key, p=1.0 - sparsity, shape=(batch_size, 1, q_len, kv_len))
        return cls(
            attention_mask=random_mask,
            q_segment_ids=None,
            kv_segment_ids=None,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )

    @staticmethod
    def get_empty_sharding() -> MaskSharding:
        return MaskSharding(
            attention_mask=None,
            q_segment_ids=None,
            kv_segment_ids=None,
            q_positions=None,
            kv_positions=None,
        )

    def get_shardings(self, sequence_parallel: bool = False, *, mesh: Mesh) -> MaskSharding:
        sequence_axis_name = None
        batch_axis_name = self.batch_axis_name
        qheads_axis_name = self.qheads_axis_name
        axis_names = list(mesh.axis_names)
        if qheads_axis_name is not None:
            its = qheads_axis_name if isinstance(qheads_axis_name, tuple) else (qheads_axis_name,)
            for e in its:
                if e not in axis_names:
                    raise ValueError(f"{qheads_axis_name} is not it {mesh.axis_names}")
        if batch_axis_name is not None:
            its = batch_axis_name if isinstance(batch_axis_name, tuple) else (batch_axis_name,)
            for e in its:
                if e not in axis_names:
                    raise ValueError(f"{batch_axis_name} is not it {mesh.axis_names}")
        if sequence_axis_name is not None:
            its = sequence_axis_name if isinstance(sequence_axis_name, tuple) else (sequence_axis_name,)
            for e in its:
                if e not in axis_names:
                    raise ValueError(f"{sequence_axis_name} is not it {mesh.axis_names}")
        if sequence_parallel:
            sequence_axis_name = self.sequence_axis_name

        attention_mask = None
        q_segment_ids = None
        kv_segment_ids = None
        q_positions = None
        kv_positions = None

        if self.attention_mask is not None:
            if self.attention_mask.ndim == 4:
                attention_mask = PartitionSpec(batch_axis_name, qheads_axis_name, None, None)
            else:
                raise ValueError(
                    f"attention_mask should be 4D array [batch, num_heads_or_1, q, kv], "
                    f"got {self.attention_mask.ndim}D with shape {self.attention_mask.shape}"
                )
            attention_mask = get_corrected_named_sharding(
                self.attention_mask.shape,
                attention_mask,
                mesh=mesh,
            ).spec
        if self.q_segment_ids is not None:
            q_segment_ids = get_corrected_named_sharding(
                self.q_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
        if self.kv_segment_ids is not None:
            kv_segment_ids = get_corrected_named_sharding(
                self.kv_segment_ids.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
        if self.q_positions is not None:
            q_positions = get_corrected_named_sharding(
                self.q_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec
        if self.kv_positions is not None:
            kv_positions = get_corrected_named_sharding(
                self.kv_positions.shape,
                PartitionSpec(batch_axis_name, sequence_axis_name),
                mesh=mesh,
            ).spec

        return MaskSharding(
            attention_mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
        )

    def get_or_compute_positions(self) -> tuple[Array | None, Array | None]:
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

        if q_positions is None and self.q_len is not None and self.batch_size is not None:
            q_positions = jnp.tile(jnp.arange(self.q_len, dtype=jnp.int32), (self.batch_size, 1))

        if kv_positions is None and self.kv_len is not None and self.batch_size is not None:
            kv_positions = jnp.tile(jnp.arange(self.kv_len, dtype=jnp.int32), (self.batch_size, 1))

        return q_positions, kv_positions

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
        if self.q_segment_ids is not None and self.kv_segment_ids is not None:
            return segment_ids_to_mask((self.q_segment_ids, self.kv_segment_ids), dtype=dtype)

        if self.attention_mask is not None:
            return self.attention_mask.astype(dtype)

        raise ValueError("Cannot compute attention mask: both attention_mask and segment_ids are None")

    def get_or_compute_segment_ids(self, per_head: bool = False) -> tuple[Array, Array]:
        """
        Get segment IDs, computing from attention mask if not available.

        Args:
            per_head: If True and attention mask is 4D, compute segment IDs per head

        Returns:
            Tuple of (q_segment_ids, kv_segment_ids)

        Raises:
            ValueError: If both attention_mask and segment_ids are None
        """
        if self.q_segment_ids is not None and self.kv_segment_ids is not None:
            return self.q_segment_ids, self.kv_segment_ids

        if self.attention_mask is not None:
            return mask_to_segment_ids(self.attention_mask, per_head)

        raise ValueError("Cannot compute segment IDs: both attention_mask and segment_ids are None")

    def get_qkv_masks(self, dtype: DTypeLike = jnp.bool_) -> tuple[Array, Array, Array]:
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
            attention_mask=self.attention_mask.astype(dtype),
            q_segment_ids=self.q_segment_ids,
            kv_segment_ids=self.kv_segment_ids,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
        )

    @property
    def batch_size(self) -> int | None:
        """Get batch size from available data."""
        if self.q_segment_ids is not None:
            return self.q_segment_ids.shape[0]
        if self.attention_mask is not None:
            return self.attention_mask.shape[0]
        return None

    @property
    def q_len(self) -> int | None:
        """Get query sequence length."""
        if self.q_segment_ids is not None:
            return self.q_segment_ids.shape[-1]
        if self.attention_mask is not None:
            return self.attention_mask.shape[-2]
        return None

    @property
    def kv_len(self) -> int | None:
        """Get key-value sequence length."""
        if self.kv_segment_ids is not None:
            return self.kv_segment_ids.shape[-1]
        if self.attention_mask is not None:
            return self.attention_mask.shape[-1]
        return None

    @property
    def shape(self) -> tuple[int | None, int | None, int | None]:
        """Get (batch_size, q_len, kv_len) shape tuple."""
        return (self.batch_size, self.q_len, self.kv_len)

    def apply_causal(self, offset: int = 0) -> "MaskInfo":
        """
        Apply causal masking by preserving segment IDs and applying causal constraint to attention mask.

        Causal masking prevents tokens from attending to future positions. The segment IDs
        are preserved to maintain the original grouping structure, while the attention mask
        encodes both the segment constraint (tokens in same segment) AND the causal constraint
        (no attending to future positions).

        Args:
            offset: Offset of q start wrt kv. A positive offset shifts the bottom
                triangle upward, a negative one shifts it downward.
                - offset=0: Standard causal mask (attend to self and past)
                - offset>0: Can attend slightly into the "future"
                - offset<0: Cannot attend to most recent tokens

        Returns:
            New MaskInfo with causal constraint applied

        Example:
            >>> segment_ids = jnp.array([[1, 1, 2, 2]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>> causal_mask_info = mask_info.apply_causal()
            >>>
        """
        if self.q_len is None or self.kv_len is None:
            raise ValueError("Cannot apply causal mask: mask dimensions unknown")

        q_seg, kv_seg = self.get_or_compute_segment_ids()

        # Use existing attention mask if available, otherwise compute from segment IDs
        # This allows chaining transformations without losing previous constraints
        if self.attention_mask is not None:
            attention_mask = self.attention_mask
        else:
            attention_mask = self.get_or_compute_attention_mask()

        shape = attention_mask.shape
        q_idx = jnp.arange(shape[-2], dtype=jnp.int32)
        kv_idx = jnp.arange(shape[-1], dtype=jnp.int32)
        causal = (q_idx[:, None] + offset >= kv_idx[None, :]).astype(jnp.bool_)

        # Attention mask is always 4D (batch, 1 or heads, q, kv)
        # Broadcast causal mask to 4D: (1, 1, q, kv)
        causal = causal[None, None, :, :]

        attention_mask = attention_mask & causal

        return MaskInfo(
            attention_mask=attention_mask,
            q_segment_ids=q_seg,
            kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
        )

    def apply_sliding_window(
        self, window_size: int | tuple[int, int] | tuple[int | None, int | None], offset: int = 0
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
            raise ValueError("Cannot apply sliding window: mask dimensions unknown")

        if isinstance(window_size, int):
            left, right = window_size, window_size
        else:
            left, right = window_size

        q_seg, kv_seg = self.get_or_compute_segment_ids()

        # Use existing attention mask if available, otherwise compute from segment IDs
        # This allows chaining transformations without losing previous constraints
        if self.attention_mask is not None:
            attention_mask = self.attention_mask
        else:
            attention_mask = self.get_or_compute_attention_mask()

        shape = attention_mask.shape
        q_idx = jnp.arange(shape[-2], dtype=jnp.int32)
        kv_idx = jnp.arange(shape[-1], dtype=jnp.int32)

        local_mask = jnp.ones((shape[-2], shape[-1]), dtype=jnp.bool_)
        if left is not None:
            local_mask = local_mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
        if right is not None:
            local_mask = local_mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])

        # Attention mask is always 4D (batch, 1 or heads, q, kv)
        # Broadcast local mask to 4D: (1, 1, q, kv)
        local_mask = local_mask[None, None, :, :]

        attention_mask = attention_mask & local_mask

        return MaskInfo(
            attention_mask=attention_mask,
            q_segment_ids=q_seg,
            kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
        )

    def apply_chunked(self, chunk_size: int) -> "MaskInfo":
        """
        Apply chunked causal attention by modifying segment IDs.

        Divides the sequence into chunks where attention is causal within each chunk,
        but tokens cannot attend across chunk boundaries. This is implemented by
        creating new segment IDs based on chunk membership, using -1 for padding.
        Used in models like Llama 4.

        Args:
            chunk_size: Size of each attention chunk. Must be positive.

        Returns:
            New MaskInfo with chunked causal constraint applied to segment IDs

        Raises:
            ValueError: If chunk_size <= 0

        Example:
            >>> segment_ids = jnp.array([[1, 1, 1, 1, 1, 1]])
            >>> mask_info = MaskInfo.from_segments(segment_ids)
            >>>
            >>> chunked = mask_info.apply_chunked(chunk_size=3)
            >>>
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if self.q_len is None or self.kv_len is None:
            raise ValueError("Cannot apply chunked mask: mask dimensions unknown")

        q_seg, kv_seg = self.get_or_compute_segment_ids()

        q_idx = jnp.arange(self.q_len, dtype=jnp.int32)
        kv_idx = jnp.arange(self.kv_len, dtype=jnp.int32)

        q_chunk_ids = q_idx // chunk_size
        kv_chunk_ids = kv_idx // chunk_size

        if self.is_self_attention():
            new_seg = jnp.where(q_seg > 0, q_chunk_ids[None, :] + 1, -1)
            q_seg = new_seg
            kv_seg = new_seg
        else:
            q_seg = jnp.where(q_seg > 0, q_chunk_ids[None, :] + 1, -1)
            kv_seg = jnp.where(kv_seg > 0, kv_chunk_ids[None, :] + 1, -1)

        attention_mask = segment_ids_to_mask((q_seg, kv_seg))

        q_pos = jnp.arange(self.q_len, dtype=jnp.int32)
        kv_pos = jnp.arange(self.kv_len, dtype=jnp.int32)
        causal = q_pos[:, None] >= kv_pos[None, :]

        # Attention mask is always 4D (batch, 1, q, kv)
        # Broadcast causal mask to 4D: (1, 1, q, kv)
        causal = causal[None, None, :, :]
        attention_mask = attention_mask & causal

        return MaskInfo(
            attention_mask=attention_mask,
            q_segment_ids=q_seg,
            kv_segment_ids=kv_seg,
            q_positions=self.q_positions,
            kv_positions=self.kv_positions,
        )

    def __repr__(self) -> str:
        """Enhanced string representation with shape information."""
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
        """Flatten MaskInfo for JAX pytree registration.

        Separates traced array fields (children) from static metadata fields (aux_data).
        """
        # Arrays that should be traced by JAX
        children = (
            self.attention_mask,
            self.q_segment_ids,
            self.kv_segment_ids,
            self.q_positions,
            self.kv_positions,
        )
        # Static metadata that should not be traced
        aux_data = (
            self.batch_axis_name,
            self.qheads_axis_name,
            self.kvheads_axis_name,
            self.sequence_axis_name,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct MaskInfo from flattened pytree representation."""
        attention_mask, q_segment_ids, kv_segment_ids, q_positions, kv_positions = children
        batch_axis_name, qheads_axis_name, kvheads_axis_name, sequence_axis_name = aux_data
        return cls(
            attention_mask=attention_mask,
            q_segment_ids=q_segment_ids,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            kv_positions=kv_positions,
            batch_axis_name=batch_axis_name,
            qheads_axis_name=qheads_axis_name,
            kvheads_axis_name=kvheads_axis_name,
            sequence_axis_name=sequence_axis_name,
        )


jax.tree_util.register_pytree_node(MaskInfo, MaskInfo.tree_flatten, MaskInfo.tree_unflatten)
