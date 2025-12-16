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

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax import lax

from ejkernel.callib import ejit, triton_call

DEFAULT_MASK_VALUE = -2.381976426469702e38


def _align_to(x: int, multiple: int) -> int:
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _dtype_packing(dtype: jnp.dtype) -> int:
    bw = jnp.dtype(dtype).itemsize * 8
    if bw not in (16, 32):
        raise ValueError(f"Only 16/32-bit floats supported for packing, got {dtype} ({bw} bits).")
    return 32 // bw  # fp32->1, (b)fp16->2


def _merge_kv(keys: jax.Array, values: jax.Array, *, head_dim_padded: int) -> jax.Array:
    if keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError("keys/values mismatch")
    total_tokens, num_kv_heads, head_dim = map(int, keys.shape)
    pack = _dtype_packing(keys.dtype)
    num_kv_heads_x2 = _align_to(num_kv_heads * 2, pack)
    kv = jnp.pad(
        jnp.concatenate([keys, values], axis=-1).reshape(total_tokens, num_kv_heads * 2, head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - num_kv_heads * 2),
            (0, head_dim_padded - head_dim),
        ),
        constant_values=0,
    ).reshape(total_tokens, num_kv_heads_x2 // pack, pack, head_dim_padded)
    return kv


@triton.jit
def _rpa_v3_attn_fwd(
    q_ptr,
    kv_ptr,
    block_tables_ptr,
    kv_lens_ptr,
    cu_q_lens_ptr,
    distribution_ptr,
    sink_ptr,
    softmax_scale,
    logits_soft_cap,
    sliding_window,
    q_scale,
    k_scale,
    v_scale,
    head_dim,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    kv_stride_p,
    kv_stride_s,
    kv_stride_c,
    kv_stride_d,
    bt_stride_s,
    bt_stride_p,
    o_stride_m,
    o_stride_h,
    o_stride_d,
    o_ptr,
    NUM_REPEATS: tl.constexpr,
    MAX_NUM_SEQS: tl.constexpr,
    PAGES_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NPAGES: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_SLIDING: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    num_seqs = tl.load(distribution_ptr + 2).to(tl.int32)
    seq_active = pid_s < num_seqs

    q_start = tl.load(cu_q_lens_ptr + pid_s, mask=seq_active, other=0).to(tl.int32)
    q_end = tl.load(cu_q_lens_ptr + pid_s + 1, mask=seq_active, other=0).to(tl.int32)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + pid_s, mask=seq_active, other=0).to(tl.int32)

    # New tokens are appended at the end of the sequence KV.
    context_len = kv_len - q_len

    offs_m = pid_qb * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = seq_active & (offs_m < q_len)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < head_dim

    q_ptrs = q_ptr + (q_start + offs_m)[:, None] * q_stride_m + pid_h * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    if HAS_Q_SCALE:
        q = q / q_scale

    kv_head = pid_h // NUM_REPEATS
    k_idx = (2 * kv_head).to(tl.int32)
    v_idx = (2 * kv_head + 1).to(tl.int32)

    # KV positions for the query rows.
    row_idx = context_len + offs_m
    row_idx = tl.where(row_mask, row_idx, 0).to(tl.int32)

    if HAS_SLIDING:
        # Reference semantics: allow kv_pos > row_idx - sliding_window  (exclusive lower bound).
        # => kv_pos >= row_idx - sliding_window + 1.
        left_bound = tl.maximum(row_idx - sliding_window + 1, 0).to(tl.int32)
    else:
        left_bound = tl.zeros([BLOCK_M], tl.int32)

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + pid_h).to(tl.float32)
        m = tl.full([BLOCK_M], sink_val, tl.float32)
        l = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        m = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    # Iterate over KV blocks (in pages). Loop count is (PAGES_PER_SEQ / BLOCK_NPAGES).
    kv_block_tokens: tl.constexpr = BLOCK_NPAGES * PAGE_SIZE
    offs_k = tl.arange(0, kv_block_tokens).to(tl.int32)

    for page_base in tl.static_range(0, PAGES_PER_SEQ, BLOCK_NPAGES):
        kv_pos = (page_base * PAGE_SIZE) + offs_k

        kv_valid = kv_pos < kv_len

        # Map kv_pos -> (page_id, offset_in_page).
        page_idx = page_base + (offs_k // PAGE_SIZE)
        off_in_page = (offs_k % PAGE_SIZE).to(tl.int32)

        bt_ptrs = block_tables_ptr + pid_s * bt_stride_s + page_idx * bt_stride_p
        page_id = tl.load(bt_ptrs, mask=seq_active & (page_idx < PAGES_PER_SEQ), other=0).to(tl.int32)

        base_page = kv_ptr + page_id[:, None] * kv_stride_p + off_in_page[:, None] * kv_stride_s
        k_ptrs = base_page + k_idx * kv_stride_c + offs_d[None, :] * kv_stride_d
        v_ptrs = base_page + v_idx * kv_stride_c + offs_d[None, :] * kv_stride_d

        k_block = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(q, tl.trans(k_block)) * softmax_scale
        if HAS_K_SCALE:
            logits = logits * k_scale
        if HAS_Q_SCALE:
            logits = logits * q_scale
        if HAS_SOFTCAP:
            logits = logits_soft_cap * tl.tanh(logits / logits_soft_cap)

        allowed = (kv_pos[None, :] <= row_idx[:, None]) & (kv_pos[None, :] >= left_bound[:, None])
        mask = row_mask[:, None] & kv_valid[None, :] & allowed
        logits = tl.where(mask, logits, MASK_VALUE)

        has_any = tl.max(mask.to(tl.int32), axis=1) != 0

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m, block_max)
        new_m = tl.where(has_any | (l > 0), new_m, 0.0)

        exp_m = tl.exp(m - new_m)
        exp_logits = tl.exp(logits - new_m[:, None])
        exp_logits = tl.where(mask, exp_logits, 0.0)

        l = exp_m * l + tl.sum(exp_logits, axis=1)
        acc = exp_m[:, None] * acc + tl.dot(exp_logits, v_block)
        m = new_m

    l = tl.maximum(l, 1e-6)
    out = acc / l[:, None]
    if HAS_V_SCALE:
        out = out * v_scale

    o_ptrs = o_ptr + (q_start + offs_m)[:, None] * o_stride_m + pid_h * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, out, mask=row_mask[:, None] & d_mask[None, :])


def _contig_strides_3(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    _m, h, d = map(int, shape)
    return (h * d, d, 1)


def _contig_strides_4(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    _p, s, c, d = map(int, shape)
    return (s * c * d, c * d, d, 1)


def _kv_update_scatter(
    *,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
) -> jax.Array:
    total_tokens, _num_kv_heads, _head_dim = map(int, keys.shape)
    num_pages, page_size, _hx2_per_pack, _pack, head_dim_padded = map(int, kv_cache.shape)
    del num_pages

    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs

    merged_kv = _merge_kv(keys, values, head_dim_padded=head_dim_padded)
    kv_cache_flat = kv_cache.reshape(-1, *kv_cache.shape[2:])

    t_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    # searchsorted is stable/jittable and avoids Python loops.
    s_idx = jnp.searchsorted(query_start_loc, t_idx, side="right") - jnp.int32(1)
    s_idx = jnp.clip(s_idx, 0, max_num_seqs - 1)

    q_start = query_start_loc[s_idx]
    q_end = query_start_loc[s_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[s_idx]

    pos_in_new = t_idx - q_start
    pos_in_kv = (kv_len - q_len) + pos_in_new

    page_idx_in_seq = pos_in_kv // jnp.int32(page_size)
    page_offset = pos_in_kv - page_idx_in_seq * jnp.int32(page_size)
    flat_block_idx = s_idx * jnp.int32(pages_per_seq) + page_idx_in_seq
    physical_page = block_tables[flat_block_idx]

    scatter_row = physical_page * jnp.int32(page_size) + page_offset
    scatter_idx = scatter_row[:, None]

    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    kv_cache_flat = lax.scatter(
        kv_cache_flat,
        scatter_idx,
        merged_kv,
        dnums,
        mode=lax.GatherScatterMode.CLIP,
    )
    return kv_cache_flat.reshape(kv_cache.shape)


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
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
def ragged_paged_attention_triton(
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
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    del chunk_prefill_size, vmem_limit_bytes

    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries/keys/values must be rank-3 arrays")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape")
    if queries.shape[0] != keys.shape[0] or queries.shape[2] != keys.shape[2]:
        raise ValueError("queries/keys/values must agree on (total_tokens, head_dim)")
    if kv_cache.ndim != 5:
        raise ValueError("kv_cache must be rank-5")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")
    if query_start_loc.shape != (kv_lens.shape[0] + 1,):
        raise ValueError("query_start_loc must have shape (max_num_seqs + 1,)")
    if distribution.dtype != jnp.int32 or distribution.shape != (3,):
        raise ValueError("distribution must be int32 with shape (3,)")
    if kv_cache.dtype != keys.dtype:
        raise ValueError("kv_cache dtype must match keys/values dtype for the Triton implementation")

    kv_cache = _kv_update_scatter(
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
    )

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, hx2_per_pack, pack, head_dim_padded = map(int, kv_cache.shape)
    max_num_seqs = int(kv_lens.shape[0])
    if int(block_tables.shape[0]) % max_num_seqs != 0:
        raise ValueError("block_tables length must be divisible by max_num_seqs")
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs

    num_kv_heads = int(keys.shape[1])
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    num_repeats = num_q_heads // num_kv_heads

    combined_heads = hx2_per_pack * pack
    expected_pack = _dtype_packing(kv_cache.dtype)
    if pack != expected_pack:
        raise ValueError(f"kv_cache packing mismatch: got pack={pack}, expected={expected_pack}")
    if page_size % pack != 0:
        raise ValueError("page_size must be divisible by kv_packing")
    expected_combined = _align_to(num_kv_heads * 2, pack)
    if combined_heads != expected_combined:
        raise ValueError(
            f"kv_cache head packing mismatch: combined_heads={combined_heads}, expected={expected_combined} "
            f"(num_kv_heads={num_kv_heads}, pack={pack})"
        )
    if head_dim_padded < head_dim or head_dim_padded != _align_to(head_dim, 128):
        raise ValueError("kv_cache head_dim_padded must equal align_to(head_dim, 128)")
    kv_pages = kv_cache.reshape(_num_pages, page_size, combined_heads, head_dim_padded)

    if num_queries_per_block is None:
        block_m = 128
    else:
        block_m = int(num_queries_per_block)

    if num_kv_pages_per_block is None:
        block_npages = 16
    else:
        block_npages = int(num_kv_pages_per_block)
    block_npages = max(1, min(pages_per_seq, block_npages))
    # Prevent accidentally constructing extremely large (BQ x BK) tiles.
    block_npages = min(block_npages, 32)

    block_dmodel = max(triton.next_power_of_2(head_dim), 16)

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)

    qblocks_max = math.ceil(total_tokens / block_m)

    has_sink = attention_sink is not None
    if attention_sink is None:
        attention_sink = jnp.zeros((1,), dtype=jnp.float32)
    else:
        if attention_sink.shape != (num_q_heads,):
            raise ValueError("attention_sink must have shape (num_q_heads,)")
        attention_sink = attention_sink.astype(jnp.float32)

    has_sliding = sliding_window is not None
    sliding_window_val = int(sliding_window or 0)

    has_softcap = logits_soft_cap is not None
    logits_soft_cap_val = float(logits_soft_cap or 0.0)

    has_q_scale = q_scale is not None
    q_scale_val = float(q_scale or 1.0)

    has_k_scale = k_scale is not None
    k_scale_val = float(k_scale or 1.0)

    has_v_scale = v_scale is not None
    v_scale_val = float(v_scale or 1.0)

    out_shape = [jax.ShapeDtypeStruct(queries.shape, queries.dtype)]
    metaparams = dict(
        NUM_REPEATS=num_repeats,
        MAX_NUM_SEQS=max_num_seqs,
        PAGES_PER_SEQ=pages_per_seq,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_NPAGES=block_npages,
        BLOCK_DMODEL=block_dmodel,
        MASK_VALUE=float(DEFAULT_MASK_VALUE),
        HAS_SINK=bool(has_sink),
        HAS_SLIDING=bool(has_sliding),
        HAS_SOFTCAP=bool(has_softcap),
        HAS_Q_SCALE=bool(has_q_scale),
        HAS_K_SCALE=bool(has_k_scale),
        HAS_V_SCALE=bool(has_v_scale),
        num_warps=4,
        num_stages=1,
    )

    (out,) = triton_call(
        queries,
        kv_pages,
        block_tables,
        kv_lens,
        query_start_loc,
        distribution,
        attention_sink,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(sliding_window_val),
        float(q_scale_val),
        float(k_scale_val),
        float(v_scale_val),
        int(head_dim),
        int(q_sm),
        int(q_sh),
        int(q_sd),
        int(kv_sp),
        int(kv_ss),
        int(kv_sc),
        int(kv_sd),
        int(bt_ss),
        int(bt_sp),
        int(o_sm),
        int(o_sh),
        int(o_sd),
        kernel=_rpa_v3_attn_fwd,
        out_shape=out_shape,
        grid=lambda META: (qblocks_max, num_q_heads, max_num_seqs),
        name="ejkernel::triton::ragged_page_attention_v3_fwd",
        **metaparams,
    )
    return out, kv_cache
