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


from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import DTypeLike


@partial(jit, static_argnames=["chunk_size", "reverse", "head_first"])
def chunk_local_cumsum_scalar(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size & (chunk_size - 1) == 0, "chunk_size must be a power of 2"
    output_dtype = output_dtype or g.dtype
    if not head_first:
        g = jnp.transpose(g, (0, 2, 1))
    T = g.shape[2]
    pad_length = (chunk_size - T % chunk_size) % chunk_size
    if pad_length > 0:
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_length)), mode="constant")
    T_padded = T + pad_length
    num_chunks = T_padded // chunk_size
    g_chunked = g.reshape(B, H, num_chunks, chunk_size)
    if reverse:
        g_flipped = jnp.flip(g_chunked, axis=-1)
        cumsum_flipped = jnp.cumsum(g_flipped, axis=-1)
        result_chunked = jnp.flip(cumsum_flipped, axis=-1)
    else:
        result_chunked = jnp.cumsum(g_chunked, axis=-1)
    if softmax_scale is not None:
        result_chunked *= softmax_scale
    result = result_chunked.reshape(B, H, T_padded)
    if pad_length > 0:
        result = result[:, :, :T]
    if not head_first:
        result = jnp.transpose(result, (0, 2, 1))
    return result.astype(output_dtype)


@partial(jit, static_argnames=["chunk_size", "reverse", "head_first"])
def chunk_local_cumsum_vector(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    if head_first:
        B, H, T, _S = g.shape
    else:
        B, T, H, _S = g.shape
    assert chunk_size & (chunk_size - 1) == 0, "chunk_size must be a power of 2"
    output_dtype = output_dtype or g.dtype
    if not head_first:
        g = jnp.transpose(g, (0, 2, 1, 3))
    T = g.shape[2]
    pad_length = (chunk_size - T % chunk_size) % chunk_size
    if pad_length > 0:
        g = jnp.pad(g, ((0, 0), (0, 0), (0, pad_length), (0, 0)), mode="constant")
    T_padded = T + pad_length
    num_chunks = T_padded // chunk_size
    g_chunked = g.reshape(B, H, num_chunks, chunk_size, g.shape[-1])
    if reverse:
        g_flipped = jnp.flip(g_chunked, axis=-2)
        cumsum_flipped = jnp.cumsum(g_flipped, axis=-2)
        result_chunked = jnp.flip(cumsum_flipped, axis=-2)
    else:
        result_chunked = jnp.cumsum(g_chunked, axis=-2)
    if softmax_scale is not None:
        result_chunked *= softmax_scale
    result = result_chunked.reshape(B, H, T_padded, g.shape[-1])
    if pad_length > 0:
        result = result[:, :, :T, :]
    if not head_first:
        result = jnp.transpose(result, (0, 2, 1, 3))
    return result.astype(output_dtype)


@partial(jit, static_argnames=["reverse", "head_first"])
def chunk_global_cumsum_scalar(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    output_dtype = output_dtype or s.dtype
    time_axis = 2 if head_first else 1

    if reverse:
        s_flipped = jnp.flip(s, axis=time_axis)
        result = jnp.cumsum(s_flipped, axis=time_axis)
    else:
        result = jnp.cumsum(s, axis=time_axis)

    if cu_seqlens is not None:
        boundary_indices = cu_seqlens[1:-1] - 1
        gather_indices_shape = [1] * s.ndim
        gather_indices_shape[time_axis] = len(boundary_indices)
        gather_indices = boundary_indices.reshape(gather_indices_shape)
        correction_values = jnp.take_along_axis(result, gather_indices, axis=time_axis)

        zero_pad_shape = list(correction_values.shape)
        zero_pad_shape[time_axis] = 1
        full_correction_map = jnp.concatenate(
            [jnp.zeros(zero_pad_shape, dtype=result.dtype), correction_values],
            axis=time_axis,
        )

        total_len = s.shape[time_axis]
        seq_ids = jnp.cumsum(jnp.zeros(total_len, dtype=jnp.int32).at[cu_seqlens[1:]].set(1))

        id_shape = [1] * s.ndim
        id_shape[time_axis] = total_len
        seq_ids = seq_ids.reshape(id_shape)

        correction_tensor = jnp.take_along_axis(full_correction_map, seq_ids, axis=time_axis)
        result -= correction_tensor

    if reverse:
        result = jnp.flip(result, axis=time_axis)

    if softmax_scale is not None:
        result *= softmax_scale

    return result.astype(output_dtype)


@partial(jit, static_argnames=["reverse", "head_first"])
def chunk_global_cumsum_vector(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    """
    Perform global cumulative sum for vector values, with explicit support for cu_seqlens.
    """
    output_dtype = output_dtype or s.dtype
    time_axis = 2 if head_first else 1

    if reverse:
        s_flipped = jnp.flip(s, axis=time_axis)
        result = jnp.cumsum(s_flipped, axis=time_axis)
    else:
        result = jnp.cumsum(s, axis=time_axis)

    if cu_seqlens is not None:
        boundary_indices = cu_seqlens[1:-1] - 1

        gather_indices_shape = [1] * s.ndim
        gather_indices_shape[time_axis] = len(boundary_indices)
        gather_indices = boundary_indices.reshape(gather_indices_shape)

        correction_values = jnp.take_along_axis(result, gather_indices, axis=time_axis)

        zero_pad_shape = list(correction_values.shape)
        zero_pad_shape[time_axis] = 1
        full_correction_map = jnp.concatenate(
            [jnp.zeros(zero_pad_shape, dtype=result.dtype), correction_values],
            axis=time_axis,
        )

        total_len = s.shape[time_axis]
        seq_ids = jnp.cumsum(jnp.zeros(total_len, dtype=jnp.int32).at[cu_seqlens[1:]].set(1))

        id_shape = [1] * s.ndim
        id_shape[time_axis] = total_len
        seq_ids = seq_ids.reshape(id_shape)

        correction_tensor = jnp.take_along_axis(full_correction_map, seq_ids, axis=time_axis)
        result -= correction_tensor

    if reverse:
        result = jnp.flip(result, axis=time_axis)

    if softmax_scale is not None:
        result *= softmax_scale

    return result.astype(output_dtype)


@partial(
    jit,
    static_argnames=[
        "chunk_size",
        "reverse",
        "softmax_scale",
        "head_first",
        "output_dtype",
        "is_vector",
    ],
)
def _chunk_local_cumsum_vmap_core(
    g_padded_batched: jnp.ndarray,
    mask: jnp.ndarray,
    chunk_size: int,
    reverse: bool,
    softmax_scale: float | None,
    head_first: bool,
    output_dtype: DTypeLike | None,
    is_vector: bool,
):
    base_fn = chunk_local_cumsum_vector if is_vector else chunk_local_cumsum_scalar
    vmapped_fn = vmap(base_fn, in_axes=(0, None, None, None, None, None), out_axes=0)
    result_padded = vmapped_fn(g_padded_batched, chunk_size, reverse, softmax_scale, head_first, output_dtype)
    return result_padded * mask


def chunk_local_cumsum(
    g: jnp.ndarray,
    chunk_size: int,
    reverse: bool = False,
    softmax_scale: float | None = None,
    cu_seqlens: jnp.ndarray | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
    **kwargs,
) -> jnp.ndarray:
    is_vector = g.ndim == 4
    base_fn = chunk_local_cumsum_vector if is_vector else chunk_local_cumsum_scalar
    if cu_seqlens is None:
        return base_fn(g, chunk_size, reverse, softmax_scale, None, head_first, output_dtype)
    assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    seqlens = jnp.diff(cu_seqlens)
    max_seq_len = jnp.max(seqlens)
    num_seqs = len(seqlens)
    mask_indices = jnp.arange(max_seq_len) < seqlens[:, None]
    squeezed_g = g.squeeze(0)
    other_dims = squeezed_g.shape[1:]
    padded_g = jnp.zeros((num_seqs, max_seq_len, *other_dims), dtype=g.dtype)

    def create_padded_batch(i, _):
        start, length = cu_seqlens[i], seqlens[i]
        seq_slice = jax.lax.dynamic_slice(squeezed_g, (start,) + (0,) * len(other_dims), (length, *other_dims))
        return jax.lax.dynamic_update_slice(padded_g[i], seq_slice, (0,) * (len(other_dims) + 1))

    g_padded_batched = jnp.stack([create_padded_batch(i, None) for i in range(num_seqs)], axis=0)
    g_padded_batched = jnp.expand_dims(g_padded_batched, axis=0)
    mask_shape = (num_seqs, max_seq_len) + (1,) * len(other_dims)
    mask = mask_indices.reshape(mask_shape)
    result_padded = _chunk_local_cumsum_vmap_core(
        g_padded_batched,
        mask,
        chunk_size,
        reverse,
        softmax_scale,
        head_first,
        output_dtype,
        is_vector,
    )
    result_flat = result_padded.reshape(-1, *result_padded.shape[-(len(other_dims)) :])
    final_result = result_flat[mask_indices.flatten()]
    return final_result[None, ...]


def chunk_global_cumsum(
    s: jnp.ndarray,
    reverse: bool = False,
    cu_seqlens: jnp.ndarray | None = None,
    softmax_scale: float | None = None,
    head_first: bool = False,
    output_dtype: DTypeLike | None = None,
) -> jnp.ndarray:
    is_vector = s.ndim == 4
    if is_vector:
        return chunk_global_cumsum_vector(s, reverse, cu_seqlens, softmax_scale, head_first, output_dtype)
    else:
        return chunk_global_cumsum_scalar(s, reverse, cu_seqlens, softmax_scale, head_first, output_dtype)
