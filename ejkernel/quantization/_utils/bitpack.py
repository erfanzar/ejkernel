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

"""Bit-packing utilities for quantized weight storage.

This module provides functions to pack and unpack quantized values into
uint32 words in an MLX-compatible format. Values are stored LSB-first
(least significant bit first) within each 32-bit word.

The packing format stores `32 // bits` values per uint32 word, handling
the case where values may span word boundaries for non-power-of-2 bit widths.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp


def _pack_bits(values: jax.Array, bits: int) -> jax.Array:
    """Pack quantized codes into uint32 words (LSB-first).

    Each quantized value occupies `bits` bits, packed sequentially into
    32-bit words. Values that would span a word boundary are split across
    two consecutive words.

    Args:
        values: Array of quantized codes to pack. The last dimension contains
            the values to pack. Each value must fit in `bits` bits (0 to 2^bits - 1).
        bits: Number of bits per value. Common values: 2, 3, 4, 5, 6, 7, 8.

    Returns:
        Packed uint32 array with shape (*values.shape[:-1], n_words) where
        n_words = ceil(values.shape[-1] * bits / 32).

    Example:
        >>> values = jnp.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=jnp.uint32)
        >>> packed = _pack_bits(values, bits=4)
        >>> packed.shape  # (1,) - 8 values * 4 bits = 32 bits = 1 word
    """
    bits = int(bits)
    values = values.astype(jnp.uint32)
    n = values.shape[-1]
    bit_offsets = jnp.arange(n, dtype=jnp.uint32) * jnp.uint32(bits)
    word_idx = (bit_offsets // 32).astype(jnp.int32)
    shift = bit_offsets % 32
    split = shift + bits > 32
    low_bits = jnp.where(split, 32 - shift, bits)

    low_mask = (jnp.uint32(1) << low_bits) - 1
    low = jnp.left_shift(values & low_mask, shift)
    high = jnp.where(split, values >> low_bits, jnp.uint32(0))

    words = int((n * bits + 31) // 32)
    out = jnp.zeros((*values.shape[:-1], words), dtype=jnp.uint32)
    out = out.at[..., word_idx].add(low)

    high_idx = jnp.where(split, word_idx + 1, word_idx)
    out = out.at[..., high_idx].add(high)
    return out


def _unpack_bits(packed: jax.Array, n: int, bits: int) -> jax.Array:
    """Unpack quantized codes from uint32 words (LSB-first).

    Reverses the packing performed by _pack_bits(), extracting `n` values
    of `bits` bits each from the packed uint32 representation.

    Args:
        packed: Packed uint32 array with shape (*batch_dims, n_words).
        n: Number of values to extract from the last dimension.
        bits: Number of bits per value.

    Returns:
        Unpacked uint32 array with shape (*packed.shape[:-1], n) containing
        the original quantized codes.

    Example:
        >>> packed = jnp.array([0x76543210], dtype=jnp.uint32)  # 8 4-bit values
        >>> values = _unpack_bits(packed, n=8, bits=4)
        >>> values  # [0, 1, 2, 3, 4, 5, 6, 7]
    """
    bits = int(bits)
    bit_offsets = jnp.arange(n, dtype=jnp.uint32) * jnp.uint32(bits)
    word_idx = (bit_offsets // 32).astype(jnp.int32)
    shift = bit_offsets % 32
    split = shift + bits > 32
    low_bits = jnp.where(split, 32 - shift, bits)
    high_bits = bits - low_bits

    low_mask = (jnp.uint32(1) << low_bits) - 1
    low_word = jnp.take(packed, word_idx, axis=-1)
    low = (low_word >> shift) & low_mask

    max_idx = packed.shape[-1] - 1
    high_idx = jnp.minimum(word_idx + 1, max_idx)
    high_mask = (jnp.uint32(1) << high_bits) - 1
    high_word = jnp.take(packed, high_idx, axis=-1)
    high = jnp.where(split, high_word & high_mask, jnp.uint32(0))

    return low | (high << low_bits)
