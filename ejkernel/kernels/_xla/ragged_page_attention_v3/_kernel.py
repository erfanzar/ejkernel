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
    with jax.named_scope("rpa_v3_xla.merge_kv"):
        assert k.shape == v.shape
        assert k.dtype == v.dtype
        max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
        kv_packing = get_dtype_packing(k.dtype)
        actual_num_kv_heads_x2 = actual_num_kv_heads * 2
        num_kv_heads_x2 = align_to(actual_num_kv_heads_x2, kv_packing)
        head_dim = align_to(actual_head_dim, 128)
        kv = jnp.pad(
            jnp.concat([k, v], axis=-1).reshape(max_num_tokens, actual_num_kv_heads_x2, actual_head_dim),
            (
                (0, 0),
                (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        ).reshape(
            max_num_tokens,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            head_dim,
        )
        return kv


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
    del chunk_prefill_size, vmem_limit_bytes
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
    if num_kv_pages_per_block is not None and int(num_kv_pages_per_block) <= 0:
        raise ValueError("num_kv_pages_per_block must be > 0")
    if num_queries_per_block is not None and int(num_queries_per_block) <= 0:
        raise ValueError("num_queries_per_block must be > 0")


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
    inline=True,
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
    del chunk_prefill_size, vmem_limit_bytes
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    with jax.named_scope("rpa_v3_xla.validate"):
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
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )

    with jax.named_scope("rpa_v3_xla.setup"):
        actual_head_dim = queries.shape[2]
        total_q = queries.shape[0]
        actual_num_q_heads = queries.shape[1]
        actual_num_kv_heads = keys.shape[1]
        actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

        (
            _total_num_pages,
            page_size,
            num_kv_heads_x2_per_kv_packing,
            kv_packing,
            head_dim_padded,
        ) = kv_cache.shape
        num_kv_heads_x2 = num_kv_heads_x2_per_kv_packing * kv_packing
        max_num_seqs = kv_lens.shape[0]
        pages_per_seq = block_tables.shape[0] // max_num_seqs
        tokens_per_seq = pages_per_seq * page_size

        # Block sizes for a generic, jittable implementation.
        qblocks = 8 if num_queries_per_block is None else int(num_queries_per_block)
        if num_kv_pages_per_block is None:
            # Larger kvblocks reduces Python/XLA loop overhead in `kv_loop`.
            if pages_per_seq >= 256:
                kvblocks = 256
            elif pages_per_seq >= 128:
                kvblocks = 128
            elif pages_per_seq >= 64:
                kvblocks = 64
            else:
                kvblocks = max(1, pages_per_seq)
        else:
            kvblocks = max(1, min(pages_per_seq, int(num_kv_pages_per_block)))
        kv_tokens_per_block = kvblocks * page_size

        # Pad Q/K/V so any qblocks-sized dynamic_slice is in-bounds.
        # This may read across sequence boundaries for the final partial block, but
        # masked writes ensure correctness.
        pad_q = qblocks - 1
        if pad_q:
            queries = jnp.concatenate(
                [queries, jnp.zeros((pad_q, actual_num_q_heads, actual_head_dim), queries.dtype)],
                axis=0,
            )
            keys = jnp.concatenate(
                [keys, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), keys.dtype)],
                axis=0,
            )
            values = jnp.concatenate(
                [values, jnp.zeros((pad_q, actual_num_kv_heads, actual_head_dim), values.dtype)],
                axis=0,
            )

        padded_total_q = queries.shape[0]
        q_grouped = queries.reshape(
            padded_total_q,
            actual_num_kv_heads,
            actual_num_q_heads_per_kv_head,
            actual_head_dim,
        )
        merged_kv = merge_kv(keys, values)
        o_grouped = jnp.zeros_like(q_grouped)

        arange_q = jnp.arange(qblocks, dtype=jnp.int32)
        arange_kv = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)

        # Sliding-window KV start alignment; keep it simple and portable.
        bkv_sz = page_size if sliding_window is not None else None

        sinks_h = None
        if attention_sink is not None:
            sinks_h = attention_sink.reshape(actual_num_kv_heads, actual_num_q_heads_per_kv_head).astype(jnp.float32)

    def _seq_body(seq_idx, carry):
        o_acc, kv_cache_acc = carry

        with jax.named_scope("rpa_v3_xla.seq_setup"):
            q_start = query_start_loc[seq_idx]
            q_end = query_start_loc[seq_idx + 1]
            q_len = q_end - q_start
            kv_len = kv_lens[seq_idx]

            kv_start = jnp.int32(0)
            if sliding_window is not None:
                kv_start = jnp.maximum(kv_len - jnp.int32(sliding_window), 0)
                kv_start = (kv_start // jnp.int32(bkv_sz)) * jnp.int32(bkv_sz)

            write_start = kv_len - q_len
            num_q_blocks = (q_len + qblocks - 1) // qblocks

        with jax.named_scope("rpa_v3_xla.gather_pages"):
            indices_start = seq_idx * pages_per_seq
            page_indices = lax.dynamic_slice(block_tables, (indices_start,), (pages_per_seq,))

            kv_pages = kv_cache_acc[page_indices]
            kv_pages_flat = kv_pages.reshape(
                tokens_per_seq,
                num_kv_heads_x2_per_kv_packing,
                kv_packing,
                head_dim_padded,
            )

        def _update_kv_block(qb, kv_flat_pad):
            with jax.named_scope("rpa_v3_xla.kv_update_block"):
                q_off = qb * qblocks
                dst = write_start + q_off
                src = q_start + q_off
                updates = lax.dynamic_slice(
                    merged_kv,
                    (src, 0, 0, 0),
                    (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                existing = lax.dynamic_slice(
                    kv_flat_pad,
                    (dst, 0, 0, 0),
                    (qblocks, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                )
                q_tok = q_off + arange_q
                q_valid = q_tok < q_len
                updates = jnp.where(q_valid[:, None, None, None], updates, existing)
                return lax.dynamic_update_slice(kv_flat_pad, updates, (dst, 0, 0, 0))

        with jax.named_scope("rpa_v3_xla.kv_update"):
            kv_pages_flat_padded = jnp.concatenate(
                [
                    kv_pages_flat,
                    jnp.zeros(
                        (
                            qblocks - 1,
                            num_kv_heads_x2_per_kv_packing,
                            kv_packing,
                            head_dim_padded,
                        ),
                        dtype=kv_pages_flat.dtype,
                    ),
                ],
                axis=0,
            )
            kv_pages_flat_padded = lax.fori_loop(0, num_q_blocks, _update_kv_block, kv_pages_flat_padded)
            kv_pages_flat = kv_pages_flat_padded[:tokens_per_seq]
            kv_pages = kv_pages_flat.reshape(kv_pages.shape)
            kv_cache_acc = kv_cache_acc.at[page_indices].set(kv_pages)

        with jax.named_scope("rpa_v3_xla.attn_setup"):
            # Pad pages axis to make kvblocks-sized slices safe.
            kv_pages_padded = jnp.concatenate(
                [
                    kv_pages,
                    jnp.zeros((kvblocks - 1, *kv_pages.shape[1:]), dtype=kv_pages.dtype),
                ],
                axis=0,
            )

            num_kv_blocks = (kv_len + kv_tokens_per_block - 1) // kv_tokens_per_block
            kv_block_start = kv_start // jnp.int32(kv_tokens_per_block)

        def _process_query_block(qb, o_inner):
            with jax.named_scope("rpa_v3_xla.q_block"):
                q_off = qb * qblocks
                q_global_start = q_start + q_off
                q_block = lax.dynamic_slice(
                    q_grouped,
                    (q_global_start, 0, 0, 0),
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                )

                if q_scale is not None:
                    q_block = q_block / q_scale
                    if jnp.issubdtype(kv_pages.dtype, jnp.floating):
                        finfo = jnp.finfo(kv_pages.dtype)
                        q_block = jnp.clip(q_block, finfo.min, finfo.max)
                    q_block = q_block.astype(kv_pages.dtype)

                q_tok = q_off + arange_q
                q_valid = q_tok < q_len
                q_pos = write_start + q_tok

                init_acc = jnp.zeros(
                    (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
                    dtype=jnp.float32,
                )
                if sinks_h is not None:
                    init_m = jnp.broadcast_to(
                        sinks_h[None, :, :],
                        (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                    )
                    init_l = jnp.ones_like(init_m)
                else:
                    init_m = jnp.full(
                        (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head),
                        -jnp.inf,
                        dtype=jnp.float32,
                    )
                    init_l = jnp.zeros_like(init_m)

            def _process_kv_block(kb, state):
                with jax.named_scope("rpa_v3_xla.kv_block"):
                    acc, l, m = state
                    page_map_start = kb * kvblocks
                    kv_page_block = lax.dynamic_slice(
                        kv_pages_padded,
                        (page_map_start, 0, 0, 0, 0),
                        (kvblocks, page_size, num_kv_heads_x2_per_kv_packing, kv_packing, head_dim_padded),
                    )
                    kv_tok = kv_page_block.reshape(
                        kv_tokens_per_block,
                        num_kv_heads_x2_per_kv_packing,
                        kv_packing,
                        head_dim_padded,
                    )
                    kv_tok = kv_tok.reshape(kv_tokens_per_block, num_kv_heads_x2, head_dim_padded)
                    kv_tok = kv_tok[:, : actual_num_kv_heads * 2, :]
                    kv_tok = kv_tok.reshape(kv_tokens_per_block, actual_num_kv_heads, 2, head_dim_padded)
                    k_block = kv_tok[:, :, 0, :actual_head_dim]
                    v_block = kv_tok[:, :, 1, :actual_head_dim]

                    with jax.named_scope("logits"):
                        logits = jnp.einsum(
                            "bihd,kid->bihk",
                            q_block,
                            k_block,
                            preferred_element_type=jnp.float32,
                        )
                        logits = logits * softmax_scale
                        if k_scale is not None:
                            logits = logits * k_scale
                        if q_scale is not None:
                            logits = logits * q_scale
                        if logits_soft_cap is not None:
                            logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

                    with jax.named_scope("mask"):
                        kv_pos = kb * jnp.int32(kv_tokens_per_block) + arange_kv
                        kv_valid = jnp.logical_and(kv_pos >= kv_start, kv_pos < kv_len)
                        mask = jnp.logical_or(kv_pos[None, :] > q_pos[:, None], jnp.logical_not(kv_valid[None, :]))
                        if sliding_window is not None:
                            mask = jnp.logical_or(
                                mask,
                                kv_pos[None, :] <= (q_pos[:, None] - jnp.int32(sliding_window)),
                            )
                        mask = jnp.logical_or(mask, jnp.logical_not(q_valid)[:, None])
                        mask = mask[:, None, None, :]

                    logits = logits + jnp.where(mask, mask_value, 0.0)

                    with jax.named_scope("online_softmax"):
                        cur_max = jnp.max(logits, axis=-1)
                        new_m = jnp.maximum(m, cur_max)
                        exp_logits = jnp.exp(logits - new_m[..., None])
                        exp_logits = jnp.where(mask, 0.0, exp_logits)
                        rescale = jnp.exp(m - new_m)
                        l = rescale * l + jnp.sum(exp_logits, axis=-1)
                        acc = rescale[..., None] * acc + jnp.einsum(
                            "bihk,kid->bihd",
                            exp_logits,
                            v_block,
                            preferred_element_type=jnp.float32,
                        )
                    return acc, l, new_m

            with jax.named_scope("rpa_v3_xla.kv_loop"):
                acc, l, _m = lax.fori_loop(kv_block_start, num_kv_blocks, _process_kv_block, (init_acc, init_l, init_m))
            l = jnp.maximum(l, 1e-6)
            out_block = (acc / l[..., None]).astype(queries.dtype)
            if v_scale is not None:
                out_block = out_block * v_scale

            existing = lax.dynamic_slice(
                o_inner,
                (q_global_start, 0, 0, 0),
                (qblocks, actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim),
            )
            out_block = jnp.where(q_valid[:, None, None, None], out_block, existing)
            return lax.dynamic_update_slice(o_inner, out_block, (q_global_start, 0, 0, 0))

        with jax.named_scope("rpa_v3_xla.q_loop"):
            o_acc = lax.fori_loop(0, num_q_blocks, _process_query_block, o_acc)
        return o_acc, kv_cache_acc

    num_seqs = distribution[2]
    with jax.named_scope("rpa_v3_xla.seq_loop"):
        o_grouped, kv_cache = lax.fori_loop(0, num_seqs, _seq_body, (o_grouped, kv_cache))
    with jax.named_scope("rpa_v3_xla.finalize"):
        out = o_grouped.reshape(padded_total_q, actual_num_q_heads, actual_head_dim)[:total_q]
    return out, kv_cache
