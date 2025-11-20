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


from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def get_dtype_bitwidth(dtype):
    return dtypes.bit_width(dtype)


def get_dtype_packing(dtype):
    return 32 // get_dtype_bitwidth(dtype)


def align_to(x, a):
    return cdiv(x, a) * a


def merge_kv(k: jax.Array, v: jax.Array) -> jax.Array:
    assert k.shape == v.shape and k.dtype == v.dtype
    T, Hkv, Dact = k.shape
    pack = get_dtype_packing(k.dtype)
    Hx2_act = Hkv * 2
    Hx2 = align_to(Hx2_act, pack)
    Dalign = align_to(Dact, 128)
    kv = jnp.pad(
        jnp.concatenate([k, v], axis=-1).reshape(T, Hx2_act, Dact),
        ((0, 0), (0, Hx2 - Hx2_act), (0, Dalign - Dact)),
        constant_values=0,
    ).reshape(T, Hx2 // pack, pack, Dalign)
    return kv


def _kv_flat_unpack(flat_kv: jax.Array, actual_num_kv_heads: int, Dalign: int):
    Tcap, Hx2_per_pack, pack, _ = flat_kv.shape
    Hx2 = Hx2_per_pack * pack
    kv = flat_kv.reshape(Tcap, Hx2, Dalign)[:, : actual_num_kv_heads * 2, :]
    kv = kv.reshape(Tcap, actual_num_kv_heads, 2 * Dalign)
    K = kv[:, :, :Dalign]
    V = kv[:, :, Dalign:]
    return K, V


def static_validate_inputs(
    q,
    k,
    v,
    kv_cache,
    kv_lens,
    block_tables,
    query_start_loc,
    distribution,
    *,
    softmax_scale=1.0,
    sliding_window=None,
    logits_soft_cap=None,
    mask_value=DEFAULT_MASK_VALUE,
    q_scale=None,
    k_scale=None,
    v_scale=None,
    chunk_prefill_size=None,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes=None,
):
    if not (q.ndim == k.ndim == v.ndim == 3):
        raise ValueError("q,k,v must be 3D")
    if k.shape != v.shape or q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2]:
        raise ValueError("shape mismatch among q,k,v")
    _T, Hq, D = q.shape
    Hkv = k.shape[1]
    if Hq % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")

    _, page_size, _Hx2_per_pack, pack, Dalign = kv_cache.shape
    if Dalign != align_to(D, 128):
        raise ValueError("cache last dim must be align_to(D,128)")
    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError("kv_cache must be float")
    if pack != get_dtype_packing(kv_cache.dtype):
        raise ValueError("packing mismatch")

    if not (kv_lens.dtype == block_tables.dtype == query_start_loc.dtype == distribution.dtype == jnp.int32):
        raise ValueError("index arrays must be int32")
    max_num_seqs = kv_lens.shape[0]
    if block_tables.size % max_num_seqs != 0:
        raise ValueError("block_tables size % num_seqs != 0")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError("query_start_loc bad shape")
    if distribution.shape != (3,):
        raise ValueError("distribution shape must be (3,)")

    if page_size % pack != 0:
        raise ValueError("page_size must be divisible by packing")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window > 0")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError("soft_cap != 0")


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnums=(3,),
)
def ragged_paged_attention(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    attention_sink: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    del chunk_prefill_size, num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    static_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    T, Hq, D = queries.shape
    Hkv = keys.shape[1]
    Hrep = Hq // Hkv

    _P, page_sz, _Hx2_per_pack, _pack, Dalign = kv_cache.shape
    pages_per_seq = block_tables.shape[0] // kv_lens.shape[0]

    # Constants for tiling
    BQ = 128
    BKV = 128
    BKV = max(BKV, page_sz)
    BKV = (BKV // page_sz) * page_sz

    fused_kv = merge_kv(keys, values)

    max_num_seqs = kv_lens.shape[0]
    num_seqs = jnp.asarray(distribution[-1], dtype=jnp.int32)

    # 1. Update KV Cache using scatter
    # Map each token t in 0..T to its cache location

    def get_scatter_indices(t_idx):
        # Find sequence index
        # query_start_loc is [0, len1, len1+len2, ...]
        # searchsorted 'right' gives index in [1, num_seqs+1]
        s_idx = jnp.searchsorted(query_start_loc, t_idx, side="right") - 1

        # Position in sequence (relative to new tokens)
        pos_in_new = t_idx - query_start_loc[s_idx]

        # Position in full KV sequence
        # kv_len includes new tokens
        # q_len is length of new tokens
        q_start = query_start_loc[s_idx]
        q_end = query_start_loc[s_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[s_idx]

        # The new tokens are at the END of the KV sequence: [kv_len - q_len : kv_len]
        pos_in_kv = (kv_len - q_len) + pos_in_new

        # Page index and offset
        page_idx_in_seq = pos_in_kv // page_sz
        page_offset = pos_in_kv % page_sz

        # Physical page index
        # block_tables is [num_seqs * pages_per_seq] (flattened view logic)
        # But input block_tables is [max_num_seqs * pages_per_seq]
        flat_block_idx = s_idx * pages_per_seq + page_idx_in_seq
        physical_page = block_tables[flat_block_idx]

        return physical_page, page_offset

    # vmap over all tokens
    t_indices = jnp.arange(T, dtype=jnp.int32)
    physical_pages, page_offsets = jax.vmap(get_scatter_indices)(t_indices)

    # Flatten cache for scatter: [Total_Pages * Page_Sz, ...]
    # kv_cache shape: [Num_Pages, Page_Size, Hx2, Pack, D]
    # We want to index by (page * page_sz + offset)
    kv_cache_flat = kv_cache.reshape(-1, *kv_cache.shape[2:])

    scatter_indices = (physical_pages * page_sz + page_offsets)[:, None]  # [T, 1]

    # Scatter update
    # updates is fused_kv [T, Hx2, Pack, D]
    # operand is kv_cache_flat [Num_Pages*Page_Size, Hx2, Pack, D]

    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    kv_cache_flat = lax.scatter(
        kv_cache_flat,
        scatter_indices,
        fused_kv,
        dnums,
        mode=lax.GatherScatterMode.CLIP,  # Safe for out of bounds if any
    )

    # Reshape back
    kv_cache = kv_cache_flat.reshape(kv_cache.shape)

    # 2. Compute Attention

    # Pad out_acc to T+1 to handle OOB writes (scatter garbage to index T)
    out_acc = jnp.zeros((T + 1, Hq, D), dtype=queries.dtype)

    def attn_seq_body(seq_idx, out_acc):
        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[seq_idx]

        num_q_blocks = cdiv(q_len, BQ)

        def q_block_body(q_block_idx, out_acc):
            curr_q_start = q_block_idx * BQ

            # Load Q with masking
            indices = q_start + curr_q_start + jnp.arange(BQ)
            safe_indices = jnp.minimum(indices, T - 1)
            mask_load = indices < (q_start + q_len)

            q_chunk = queries[safe_indices]
            q_chunk = jnp.where(mask_load[:, None, None], q_chunk, 0)

            if q_scale is not None:
                info = jnp.finfo(q_chunk.dtype) if jnp.issubdtype(q_chunk.dtype, jnp.floating) else None
                q_chunk = q_chunk / q_scale
                if info is not None:
                    q_chunk = jnp.clip(q_chunk, info.min, info.max)
                q_chunk = q_chunk.astype(keys.dtype)

            # Init accumulators [BQ, H]
            if attention_sink is not None:
                m = jnp.broadcast_to(attention_sink[None, :], (BQ, Hq)).astype(jnp.float32)
                l = jnp.ones((BQ, Hq), dtype=jnp.float32)
            else:
                m = jnp.full((BQ, Hq), -jnp.inf, dtype=jnp.float32)
                l = jnp.zeros((BQ, Hq), dtype=jnp.float32)

            acc = jnp.zeros((BQ, Hq, D), dtype=jnp.float32)

            num_kv_blocks = cdiv(kv_len, BKV)

            def kv_block_body(kv_block_idx, carry):
                m, l, acc = carry
                curr_kv_start = kv_block_idx * BKV

                # Load KV
                start_page = curr_kv_start // page_sz
                num_pages = BKV // page_sz

                block_table_start = seq_idx * pages_per_seq

                page_indices = lax.dynamic_slice_in_dim(block_tables, block_table_start + start_page, num_pages, axis=0)
                pages = kv_cache[page_indices]
                flat_kv = pages.reshape(num_pages * page_sz, *pages.shape[2:])

                K_chunk, V_chunk = _kv_flat_unpack(flat_kv, Hkv, Dalign)
                K_chunk = K_chunk[:, :, :D]
                V_chunk = V_chunk[:, :, :D]

                if Hrep != 1:
                    K_chunk = jnp.repeat(K_chunk, Hrep, axis=1)
                    V_chunk = jnp.repeat(V_chunk, Hrep, axis=1)

                Q_t = q_chunk.transpose(1, 0, 2)  # [H, BQ, D]
                K_t = K_chunk.transpose(1, 0, 2)  # [H, BKV, D]
                V_t = V_chunk.transpose(1, 0, 2)  # [H, BKV, D]

                S = jnp.matmul(Q_t, jnp.swapaxes(K_t, -1, -2))  # [H, BQ, BKV]

                S = S * softmax_scale
                if k_scale is not None:
                    S = S * k_scale
                if q_scale is not None:
                    S = S * q_scale

                q_idx_rel = jnp.arange(BQ)[:, None]
                k_idx_rel = jnp.arange(BKV)[None, :]

                q_pos = (kv_len - q_len) + (curr_q_start + q_idx_rel)
                k_pos = curr_kv_start + k_idx_rel

                k_valid = k_pos < kv_len
                mask = q_pos >= k_pos
                mask = jnp.logical_and(mask, k_valid)

                if sliding_window is not None:
                    mask = jnp.logical_and(mask, (q_pos - sliding_window) < k_pos)

                mask_b = mask[None, :, :]
                S = jnp.where(mask_b, S, mask_value)

                if logits_soft_cap is not None:
                    S = logits_soft_cap * jnp.tanh(S / logits_soft_cap)

                m_curr = jnp.max(S, axis=-1)
                m_new = jnp.maximum(m.transpose(1, 0), m_curr)

                P = jnp.exp(S - m_new[:, :, None])
                l_curr = jnp.sum(P, axis=-1)

                m_prev = m.transpose(1, 0)
                alpha = jnp.exp(m_prev - m_new)

                l_new = l.transpose(1, 0) * alpha + l_curr

                pv = jnp.matmul(P.astype(V_t.dtype), V_t)
                acc_t = acc.transpose(1, 0, 2)
                acc_new = acc_t * alpha[:, :, None] + pv.astype(jnp.float32)

                return (m_new.transpose(1, 0), l_new.transpose(1, 0), acc_new.transpose(1, 0, 2))

            m, l, acc = lax.fori_loop(0, num_kv_blocks, kv_block_body, (m, l, acc))

            output = acc / l[:, :, None]
            output = output.astype(queries.dtype)
            if v_scale is not None:
                output = output * v_scale

            # Scatter to out_acc
            indices = q_start + curr_q_start + jnp.arange(BQ)
            mask_write = indices < (q_start + q_len)

            # Use T as garbage index (out_acc has size T+1)
            safe_indices = jnp.where(mask_write, indices, T)

            dnums_sc = lax.ScatterDimensionNumbers(
                update_window_dims=(1, 2),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )

            out_acc = lax.scatter(
                out_acc,
                safe_indices[:, None],
                output,
                dnums_sc,
                # Default mode (UPDATE) is fine since indices are unique and T is valid
            )

            return out_acc

        out_acc = lax.fori_loop(0, num_q_blocks, q_block_body, out_acc)
        return out_acc

    def masked_seq(i, out_acc):
        return lax.cond(i < num_seqs, lambda: attn_seq_body(i, out_acc), lambda: out_acc)

    out_final = lax.fori_loop(0, max_num_seqs, masked_seq, out_acc)

    # Slice back to T
    out_final = out_final[:T]

    return out_final, kv_cache
