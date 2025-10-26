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
from jaxtyping import Array, Float, Int32

DEFAULT_MASK_VALUE: float = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a: int, b: int) -> int:
    assert b != 0
    return (a + b - 1) // b


def align_to(x: int, a: int) -> int:
    return cdiv(x, a) * a


def get_dtype_packing(dtype) -> int:
    bits = int(np.dtype(dtype).itemsize) * 8
    return 32 // bits


def merge_kv(
    k: Float[Array, "total_tokens num_kv_heads head_dim"], v: Float[Array, "total_tokens num_kv_heads head_dim"]
) -> Float[Array, "total_tokens num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"]:
    """Merge K and V into paged KV storage [T, Hk*2/pack, pack, head_dim_padded]."""
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    actual_num_kv_heads_x2 = actual_num_kv_heads * 2
    num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
    head_dim = align_to(actual_head_dim, 128)

    kv = jnp.concatenate([k, v], axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2, actual_head_dim)
    kv = jnp.pad(
        kv,
        (
            (0, 0),
            (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    )
    kv = kv.reshape(max_num_tokens, num_kv_heads_x2 // kv_packing, kv_packing, head_dim)
    return kv


def static_validate_inputs(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    *,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    **kwargs,
) -> None:
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to equal {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(f"Expected {q.shape[0]=} to equal {k.shape[0]=} and {v.shape[0]=}")
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(f"Expected {q.shape[2]=} to equal {k.shape[2]=} and {v.shape[2]=}")

    actual_head_dim = q.shape[2]
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]
    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(f"{actual_num_q_heads=} must be divisible by {actual_num_kv_heads=}")

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = kv_cache.shape
    if head_dim != align_to(actual_head_dim, 128):
        raise ValueError(f"Expected {head_dim=} to equal {align_to(actual_head_dim, 128)=}")

    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(f"Expected {kv_cache.dtype=} to equal {k.dtype=} and {v.dtype=}")
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected floating dtype for {kv_cache.dtype=}")
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(f"{kv_packing=} does not match with {kv_cache.dtype=}")

    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    if num_kv_heads_x2 % 2 != 0:
        raise ValueError(f"Combined KV heads must be divisible by 2, got {num_kv_heads_x2}")

    if not (jnp.int32 == kv_lens.dtype == block_tables.dtype == query_start_loc.dtype == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {block_tables.dtype=},"
            f" {query_start_loc.dtype=}, {distribution.dtype=}"
        )
    if not (len(kv_lens.shape) == len(block_tables.shape) == len(query_start_loc.shape) == 1):
        raise ValueError(f"Expected 1D array for {kv_lens.shape=}, {block_tables.shape=}, {query_start_loc.shape=}")
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = block_tables.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {query_start_loc.shape=} to be ({max_num_seqs + 1},)")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,)")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError(f"{logits_soft_cap=} must not be 0.0.")

    _ = kwargs


def dynamic_validate_inputs(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    **kwargs,
) -> None:
    static_validate_inputs(
        queries, keys, values, kv_cache, kv_lens, block_tables, query_start_loc, distribution, **kwargs
    )

    max_num_tokens = queries.shape[0]
    max_num_seqs = kv_lens.shape[0]
    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")
    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")
    if query_start_loc[k] > max_num_tokens:
        raise ValueError(f"Total q tokens {query_start_loc[k]} must be <= {max_num_tokens=}")

    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    num_page_indices = block_tables.shape[0]
    pages_per_seq = num_page_indices // max_num_seqs
    for s in range(int(k)):
        kv_len = int(kv_lens[s])
        page_cnt = cdiv(kv_len, page_size)
        for p in range(page_cnt):
            page_idx = int(block_tables[s * pages_per_seq + p])
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence {s} where {kv_len=} and {page_size=}."
                )


@jax.jit
def _process_single_sequence(
    q: Float[Array, "q_padded_len num_heads head_dim"],
    k: Float[Array, "kv_padded_len num_heads head_dim"],
    v: Float[Array, "kv_padded_len num_heads head_dim"],
    q_len: int,
    kv_len: int,
    *,
    softmax_scale: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    mask_value: float,
) -> Float[Array, "q_padded_len num_heads head_dim"]:
    """JIT hot path: attention for one sequence; works for any q_len, kv_len."""

    q_padded_len = q.shape[0]
    kv_padded_len = k.shape[0]

    q_valid = jax.lax.dynamic_slice(q, (0, 0, 0), (q_padded_len, q.shape[1], q.shape[2]))
    k_valid = jax.lax.dynamic_slice(k, (0, 0, 0), (kv_padded_len, k.shape[1], k.shape[2]))
    v_valid = jax.lax.dynamic_slice(v, (0, 0, 0), (kv_padded_len, v.shape[1], v.shape[2]))

    attn = jnp.einsum("qhd,khd->hqk", q_valid, k_valid, preferred_element_type=jnp.float32)
    attn *= softmax_scale
    if q_scale is not None:
        attn *= q_scale
    if k_scale is not None:
        attn *= k_scale
    if logits_soft_cap is not None:
        attn = logits_soft_cap * jnp.tanh(attn / logits_soft_cap)

    q_positions = (kv_len - q_len) + jnp.arange(q_padded_len)[:, None]
    kv_positions = jnp.arange(kv_padded_len)[None, :]

    q_valid_mask = jnp.arange(q_padded_len) < q_len
    kv_valid_mask = jnp.arange(kv_padded_len) < kv_len
    valid_mask = q_valid_mask[:, None] & kv_valid_mask[None, :]

    mask = q_positions < kv_positions
    if sliding_window is not None:
        mask = mask | (q_positions - sliding_window >= kv_positions)

    mask = mask | ~valid_mask
    attn = attn + jnp.where(mask[None, :, :], mask_value, 0.0)

    attn = jax.nn.softmax(attn, axis=-1).astype(v_valid.dtype)
    out = jnp.einsum("hqk,khd->qhd", attn, v_valid).astype(q.dtype)
    if v_scale is not None:
        out = out * v_scale

    return out


def _ragged_paged_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    kv_cache: Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
    kv_lens: Int32[Array, "max_num_seqs"],
    block_tables: Int32[Array, "max_num_seqs_times_pages_per_seq"],
    query_start_loc: Int32[Array, "max_num_seqs_plus_1"],
    distribution: Int32[Array, "3"],
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    **kwargs,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_pages page_size num_kv_heads_x2_per_kv_packing kv_packing head_dim_padded"],
]:
    """Optimized and correct ragged paged attention.

    Returns:
      (result, updated_kv_cache)
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        **kwargs,
    )

    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

    merged_kv = merge_kv(keys, values)

    _, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim = kv_cache.shape
    num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
    assert num_kv_heads_x2 % 2 == 0

    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = block_tables.shape[0] // max_num_seqs

    num_seqs = distribution[-1].astype(jnp.int32)

    queries.shape[0]
    outputs = jnp.zeros_like(queries)

    def process_sequence(seq_idx, carry):
        kv_cache_acc, outputs_acc = carry

        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[seq_idx]

        def process_non_empty():
            page_cnt = (kv_len + page_size - 1) // page_size
            indices_start = seq_idx * pages_per_seq

            all_seq_page_indices = jax.lax.dynamic_slice(
                block_tables,
                (indices_start,),
                (pages_per_seq,),
            )

            page_mask = jnp.arange(pages_per_seq) < page_cnt

            safe_page_indices = jnp.where(page_mask, all_seq_page_indices, 0)

            gathered_kv = kv_cache_acc[safe_page_indices]

            flat_shape = (pages_per_seq * page_size, *gathered_kv.shape[-3:])
            flat = gathered_kv.reshape(flat_shape)

            kv_start_idx = kv_len - q_len

            max_update_len = page_size

            q_indices = jnp.arange(max_update_len) + q_start
            valid_mask = jnp.arange(max_update_len) < q_len

            safe_q_indices = jnp.minimum(q_indices, merged_kv.shape[0] - 1)

            update_data = merged_kv[safe_q_indices]

            update_data_masked = jnp.where(valid_mask[:, None, None, None], update_data, jnp.zeros_like(update_data))

            flat = jax.lax.dynamic_update_slice(flat, update_data_masked, (kv_start_idx, 0, 0, 0))

            updated_pages = flat.reshape(gathered_kv.shape)

            scatter_mask = page_mask[:, None, None, None, None]
            scatter_mask = jnp.broadcast_to(scatter_mask, updated_pages.shape)

            updates_to_apply = jnp.where(scatter_mask, updated_pages, gathered_kv)
            new_kv_cache = kv_cache_acc.at[safe_page_indices].set(updates_to_apply)

            flat_kv = flat.reshape(-1, num_kv_heads_x2, head_dim)

            max_kv_len = pages_per_seq * page_size

            kv_padded = jax.lax.dynamic_slice(flat_kv, (0, 0, 0), (max_kv_len, actual_num_kv_heads * 2, head_dim))

            kv_indices = jnp.arange(max_kv_len)
            kv_mask = kv_indices < kv_len
            kv = jnp.where(kv_mask[:, None, None], kv_padded, jnp.zeros_like(kv_padded))

            kv = kv.reshape(-1, actual_num_kv_heads, head_dim * 2)
            k = kv[:, :, :actual_head_dim]
            v = kv[:, :, head_dim : head_dim + actual_head_dim]

            if actual_num_q_heads_per_kv_head > 1:
                k = jnp.repeat(k, actual_num_q_heads_per_kv_head, axis=1)
                v = jnp.repeat(v, actual_num_q_heads_per_kv_head, axis=1)

            max_q_len = page_size
            padded_queries = (jnp.pad(queries, ((0, max_q_len), (0, 0), (0, 0)), mode="constant"),)

            q_padded = jax.lax.dynamic_slice(
                padded_queries,
                (q_start, 0, 0),
                (max_q_len, actual_num_q_heads, actual_head_dim),
            )

            q_indices = jnp.arange(max_q_len)
            q_mask = q_indices < q_len
            q = jnp.where(q_mask[:, None, None], q_padded, jnp.zeros_like(q_padded))

            out = _process_single_sequence(
                q,
                k,
                v,
                q_len,
                kv_len,
                softmax_scale=softmax_scale,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                mask_value=mask_value,
            )
            new_outputs = jax.lax.dynamic_update_slice(outputs_acc, out, (q_start, 0, 0))

            return new_kv_cache, new_outputs

        def process_empty():
            return kv_cache_acc, outputs_acc

        return jax.lax.cond(q_len > 0, process_non_empty, process_empty)

    final_kv_cache, final_outputs = jax.lax.fori_loop(0, num_seqs, process_sequence, (kv_cache, outputs))

    total_q_tokens = query_start_loc[num_seqs]
    token_mask = jnp.arange(final_outputs.shape[0]) < total_q_tokens
    result = jnp.where(token_mask[:, None, None], final_outputs, jnp.zeros_like(final_outputs))

    return result, final_kv_cache
