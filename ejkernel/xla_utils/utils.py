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
