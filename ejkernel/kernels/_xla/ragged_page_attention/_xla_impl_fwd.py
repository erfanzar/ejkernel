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

from ejkernel.callib import ejit

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


@ejit(static_argnames=("sliding_window", "logits_soft_cap", "block_q"))
def _ragged_paged_attention_chunked(
    queries,
    kv_pages,
    context_lens,
    block_tables,
    query_start_loc,
    num_seqs,
    *,
    softmax_scale=1.0,
    sliding_window=None,
    logits_soft_cap=None,
    mask_value=DEFAULT_MASK_VALUE,
    k_scale=None,
    v_scale=None,
    block_q: int = 2048,
):
    S = block_tables.shape[0]
    P = block_tables.shape[1]
    G = kv_pages.shape[1]
    Hc = kv_pages.shape[2]
    D = kv_pages.shape[3]
    Hkv = Hc // 2
    Hq = queries.shape[1]
    H_rep = Hq // Hkv
    Lkv_max = P * G
    Q_total = queries.shape[0]
    q_bs = min(block_q, Q_total)
    cap_start = max(0, Q_total - q_bs)

    kv_gathered = kv_pages[block_tables].reshape(S, Lkv_max, Hc, D)
    k = kv_gathered[:, :, 0::2, :]
    v = kv_gathered[:, :, 1::2, :]
    if k_scale is not None:
        k = (k.astype(jnp.float32) * jnp.asarray(k_scale, jnp.float32)).astype(queries.dtype)
    if v_scale is not None:
        v = (v.astype(jnp.float32) * jnp.asarray(v_scale, jnp.float32)).astype(queries.dtype)
    k = jnp.repeat(k, H_rep, axis=2)
    v = jnp.repeat(v, H_rep, axis=2)

    kv_iota = jnp.arange(Lkv_max, dtype=context_lens.dtype)
    kv_valid = kv_iota[None, :] < context_lens[:, None]
    kv_valid_f = kv_valid[..., None, None].astype(k.dtype)
    k = k * kv_valid_f
    v = v * kv_valid_f

    q_idx = jnp.arange(Q_total, dtype=query_start_loc.dtype)
    q_seq_id = jnp.searchsorted(query_start_loc, q_idx, side="right") - 1
    q_seq_id = jnp.clip(q_seq_id, 0, S - 1)
    q_starts = query_start_loc[q_seq_id]
    q_pos = q_idx - q_starts
    q_lens = query_start_loc[1:] - query_start_loc[:-1]
    q_len_q = q_lens[q_seq_id]
    kv_len_q = context_lens[q_seq_id]
    q_span = (kv_len_q - q_len_q) + q_pos

    kv_ids = jnp.arange(Lkv_max, dtype=context_lens.dtype)

    def step(carry, i_block):
        out = carry
        start = i_block * block_q
        start_c = jnp.minimum(start, jnp.asarray(cap_start, dtype=start.dtype))
        idx_rel = jnp.arange(q_bs, dtype=q_idx.dtype)
        valid = (idx_rel + start) < Q_total
        delta = start - start_c

        q_block = jax.lax.dynamic_slice(
            queries,
            (start_c, 0, 0),
            (q_bs, Hq, D),
        )
        seq_id_block = jax.lax.dynamic_slice_in_dim(q_seq_id, start_c, q_bs, axis=0)
        q_span_block = jax.lax.dynamic_slice_in_dim(q_span, start_c, q_bs, axis=0)
        kv_len_block = jax.lax.dynamic_slice_in_dim(kv_len_q, start_c, q_bs, axis=0)

        k_b = k[seq_id_block]
        v_b = v[seq_id_block]

        attn = jnp.einsum("qbd,qkbd->qbk", q_block, k_b, preferred_element_type=jnp.float32)
        attn = attn * jnp.asarray(softmax_scale, attn.dtype)

        if logits_soft_cap is not None:
            sc = jnp.asarray(logits_soft_cap, attn.dtype)
            attn = sc * jnp.tanh(attn / sc)

        mask = kv_ids[None, :] > q_span_block[:, None]
        if sliding_window is not None:
            win = jnp.asarray(sliding_window, q_span.dtype)
            mask_left = kv_ids[None, :] <= (q_span_block[:, None] - win)
            mask = jnp.logical_or(mask, mask_left)
        mask = jnp.logical_or(mask, kv_ids[None, :] >= kv_len_block[:, None])

        attn = attn + jnp.where(mask[:, None, :], jnp.asarray(mask_value, attn.dtype), jnp.array(0.0, attn.dtype))
        attn = jax.nn.softmax(attn, axis=-1).astype(v_b.dtype)

        out_block = jnp.einsum("qbk,qkbd->qbd", attn, v_b, preferred_element_type=jnp.float32).astype(queries.dtype)

        keep = jnp.logical_and(valid, idx_rel >= delta)
        out_block = jnp.where(keep[:, None, None], out_block, jnp.array(0, out_block.dtype))
        out = jax.lax.dynamic_update_slice(out, out_block, (start_c, 0, 0))

        return out, None

    n_blocks = (Q_total + block_q - 1) // block_q
    out_init = jnp.zeros_like(queries)
    out, _ = jax.lax.scan(step, out_init, jnp.arange(n_blocks))
    return out
