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
    kv_cap = pages_per_seq * page_sz
    positions = jnp.arange(kv_cap, dtype=jnp.int32)
    page_positions = jnp.arange(pages_per_seq, dtype=jnp.int32)
    B = min(kv_cap, int(T))
    B_i32 = jnp.int32(B)
    row_indices = jnp.arange(B, dtype=jnp.int32)
    scatter_dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    T_i32 = jnp.int32(T)
    max_row_idx = jnp.maximum(T_i32 - 1, 0)

    fused_kv = merge_kv(keys, values)

    out_acc = jnp.zeros_like(queries)

    num_seqs = jnp.asarray(distribution[-1], dtype=jnp.int32)
    max_num_seqs = kv_lens.shape[0]

    def seq_body(seq_idx, carry):
        out_accum, cache_accum = carry

        q_start = query_start_loc[seq_idx]
        q_end = query_start_loc[seq_idx + 1]
        q_len = q_end - q_start
        kv_len = kv_lens[seq_idx]

        start_idx = seq_idx * pages_per_seq
        indices_all = lax.dynamic_slice_in_dim(block_tables, start_idx, pages_per_seq, axis=0)
        gathered = cache_accum[indices_all]
        flat = gathered.reshape(kv_cap, *gathered.shape[2:])

        tail_mask = jnp.logical_and(positions >= (kv_len - q_len), positions < kv_len)
        src_idx = q_start + (positions - (kv_len - q_len))
        fused_rows = jnp.take(fused_kv, src_idx, axis=0, mode="fill", fill_value=0)
        flat_updated = jnp.where(tail_mask[:, None, None, None], fused_rows, flat)

        page_cnt = cdiv(kv_len, page_sz)
        page_mask = page_positions < page_cnt
        page_mask_b = page_mask[:, None, None, None, None]
        gathered_upd = flat_updated.reshape(gathered.shape)
        blend = jnp.where(page_mask_b, gathered_upd, gathered)
        cache_accum = cache_accum.at[indices_all].set(blend)

        K_full, V_full = _kv_flat_unpack(flat_updated, Hkv, Dalign)
        K_full = K_full[:, :, :D]
        V_full = V_full[:, :, :D]
        if Hrep != 1:
            K_full = jnp.repeat(K_full, Hrep, axis=1)
            V_full = jnp.repeat(V_full, Hrep, axis=1)

        idx_q = q_start + positions
        Q_full = jnp.take(queries, idx_q, axis=0, mode="fill", fill_value=0)

        if q_scale is not None:
            info = jnp.finfo(K_full.dtype) if jnp.issubdtype(K_full.dtype, jnp.floating) else None
            Q_full = Q_full / q_scale
            if info is not None:
                Q_full = jnp.clip(Q_full, info.min, info.max)
            Q_full = Q_full.astype(K_full.dtype)

        Q_hqd = Q_full.astype(jnp.float32).transpose(1, 0, 2)
        K_hkd = K_full.astype(jnp.float32).transpose(1, 0, 2)
        S = jnp.matmul(Q_hqd, jnp.swapaxes(K_hkd, -1, -2))
        S = S * softmax_scale
        if k_scale is not None:
            S = S * k_scale
        if q_scale is not None:
            S = S * q_scale

        q_pos = positions
        k_pos = positions
        q_valid = q_pos < q_len
        k_valid = k_pos < kv_len
        q_span = (kv_len - q_len) + q_pos
        causal = q_span[:, None] < k_pos[None, :]
        bad = (~q_valid[:, None]) | (~k_valid[None, :]) | causal
        if sliding_window is not None:
            bad = jnp.logical_or(bad, (q_span[:, None] - sliding_window) >= k_pos[None, :])

        if logits_soft_cap is not None:
            S = logits_soft_cap * jnp.tanh(S / logits_soft_cap)
        S = jnp.where(bad[None, :, :], S + mask_value, S)

        P = jax.nn.softmax(S, axis=-1).astype(V_full.dtype)
        V_hkd = V_full.astype(jnp.float32).transpose(1, 0, 2)
        O_hqd = jnp.matmul(P.astype(jnp.float32), V_hkd)
        O = O_hqd.transpose(1, 0, 2)
        if v_scale is not None:
            O = O * v_scale

        row_valid = (positions < q_len).astype(O.dtype)[:, None, None]
        O = O * row_valid

        O_block = lax.dynamic_slice_in_dim(O.astype(out_accum.dtype), 0, B, axis=0)

        fits = (q_start + B_i32) <= T_i32

        def _block(a):
            return lax.dynamic_update_slice(a, O_block, (q_start, 0, 0))

        def _tail(a):
            dest = q_start + row_indices
            dest_valid = dest < T_i32
            updates = jnp.where(dest_valid[:, None, None], O_block, jnp.zeros_like(O_block))
            dest = jnp.clip(dest, 0, max_row_idx)
            return lax.scatter_add(
                a,
                dest[:, None],
                updates,
                scatter_dnums,
                indices_are_sorted=True,
                unique_indices=False,
                mode=lax.GatherScatterMode.CLIP,
            )

        out_accum = lax.cond(fits, _block, _tail, out_accum)
        return (out_accum, cache_accum)

    def masked_seq(i, carry):
        return lax.cond(i < num_seqs, lambda c: seq_body(i, c), lambda c: c, carry)

    out_final, kv_cache_final = lax.fori_loop(0, max_num_seqs, masked_seq, (out_acc, kv_cache))
    return out_final, kv_cache_final
