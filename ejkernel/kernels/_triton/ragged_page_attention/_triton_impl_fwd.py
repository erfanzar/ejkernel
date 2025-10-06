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


import triton
import triton.language as tl


@triton.jit
def _ragged_paged_attn_prefetch_kernel_combined(
    Q_ptr,
    KV_pages_ptr,
    block_tables_ptr,
    row_seq_ptr,
    row_firstk_ptr,
    row_kvlen_ptr,
    row_valid_ptr,
    Out_ptr,
    T: tl.constexpr,
    KVH: tl.constexpr,
    QHG: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    PAGES_PER_SEQ_MAX: tl.constexpr,
    KV_PAGES_PER_BLOCK: tl.constexpr,
    MAX_KV_SUPERBLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    USE_SOFT_CAP: tl.constexpr,
    SLIDING_WINDOW_SIZE: tl.constexpr,
    USE_SLIDING_WINDOW: tl.constexpr,
    MASK_VALUE: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    g = tl.program_id(2)

    sQ_t: tl.constexpr = KVH * QHG * D
    sQ_h: tl.constexpr = QHG * D
    sQ_g: tl.constexpr = D
    C: tl.constexpr = 2 * KVH
    sKV_p: tl.constexpr = PS * C * D
    sKV_s: tl.constexpr = C * D
    sKV_h: tl.constexpr = D

    head_kv = 2 * h

    q_base = t * sQ_t + h * sQ_h + g * sQ_g
    d_off = tl.arange(0, D)
    ps_off = tl.arange(0, PS)

    NEG_BIG = MASK_VALUE

    validrow = tl.load(row_valid_ptr + t, mask=True, other=False)
    seq_idx = tl.load(row_seq_ptr + t, mask=validrow, other=0)
    first_k = tl.load(row_firstk_ptr + t, mask=validrow, other=-1)
    kv_len = tl.load(row_kvlen_ptr + t, mask=validrow, other=0)

    L = tl.minimum(kv_len, first_k + 1)
    should_continue = validrow and (kv_len > 0) and (L > 0)

    if not should_continue:
        zero_vec = tl.zeros([D], dtype=tl.float32)
        tl.store(Out_ptr + q_base + d_off, zero_vec, mask=(d_off < D))
        return

    num_pages_seq = (kv_len + PS - 1) // PS
    last_page_needed = tl.minimum(num_pages_seq - 1, first_k // PS)
    n_pages_eff = last_page_needed + 1

    if n_pages_eff <= 0:
        zero_vec = tl.zeros([D], dtype=tl.float32)
        tl.store(Out_ptr + q_base + d_off, zero_vec, mask=(d_off < D))
        return

    q_mask = d_off < D
    q_vec = tl.load(Q_ptr + q_base + d_off, mask=q_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    out_vec = tl.zeros([D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    max_val = tl.full((), NEG_BIG, dtype=tl.float32)

    bt_row_ptr = block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX

    neg_big = tl.full([PS], NEG_BIG, dtype=tl.float32)

    for kv_super in tl.static_range(MAX_KV_SUPERBLOCKS):
        page_idx_base = kv_super * KV_PAGES_PER_BLOCK
        kv_token_block_start = page_idx_base * PS

        valid_super = page_idx_base < n_pages_eff

        if valid_super:
            safe0 = tl.minimum(page_idx_base, PAGES_PER_SEQ_MAX - 1)
            pid0 = tl.load(bt_row_ptr + safe0, mask=True, other=0)

            kv_base = KV_pages_ptr + pid0 * sKV_p + ps_off[:, None] * sKV_s + head_kv * sKV_h
            k_tile = tl.load(kv_base + d_off[None, :], eviction_policy="evict_last").to(tl.float32)
            v_tile = tl.load(kv_base + sKV_h + d_off[None, :], eviction_policy="evict_first").to(tl.float32)
        else:
            k_tile = tl.zeros([PS, D], dtype=tl.float32)
            v_tile = tl.zeros([PS, D], dtype=tl.float32)

        for p in tl.static_range(KV_PAGES_PER_BLOCK):
            page_index_idx = page_idx_base + p
            page_start_tok = kv_token_block_start + p * PS

            valid_page = valid_super and (page_index_idx < n_pages_eff)

            tokens_in_page = tl.minimum(PS, tl.maximum(0, L - page_start_tok))
            should_process = valid_page and (tokens_in_page > 0)

            if should_process:
                scores = tl.sum(k_tile * q_vec[None, :], axis=1) * SCALE

                if USE_SOFT_CAP:
                    x = scores / SOFT_CAP
                    x = tl.maximum(tl.minimum(x, 8.0), -8.0)
                    exp_2x = tl.exp(2.0 * x)
                    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
                    scores = SOFT_CAP * tanh_x

                if USE_SLIDING_WINDOW:
                    token_positions = page_start_tok + ps_off

                    window_mask = (first_k - token_positions) < SLIDING_WINDOW_SIZE
                    scores = tl.where(window_mask, scores, NEG_BIG)

                masked_scores = tl.where(ps_off < tokens_in_page, scores, neg_big)

                local_max = tl.max(masked_scores, axis=0)
                new_max = tl.maximum(max_val, local_max)

                exp_old = tl.exp(max_val - new_max)
                score_exp = tl.exp(masked_scores - new_max)

                probs = score_exp * (ps_off < tokens_in_page).to(tl.float32)
                sum_exp = exp_old * sum_exp + tl.sum(probs, axis=0)

                val_update = tl.sum(probs[:, None] * v_tile, axis=0)
                out_vec = exp_old * out_vec + val_update

                max_val = new_max

            if p + 1 < KV_PAGES_PER_BLOCK:
                next_idx = page_idx_base + p + 1
                valid_next = valid_super and (next_idx < n_pages_eff)
                if valid_next:
                    safen = tl.minimum(next_idx, PAGES_PER_SEQ_MAX - 1)
                    pidn = tl.load(bt_row_ptr + safen, mask=True, other=0)

                    kv_base_n = KV_pages_ptr + pidn * sKV_p + ps_off[:, None] * sKV_s + head_kv * sKV_h
                    k_tile = tl.load(kv_base_n + d_off[None, :], eviction_policy="evict_last").to(tl.float32)
                    v_tile = tl.load(kv_base_n + sKV_h + d_off[None, :], eviction_policy="evict_first").to(tl.float32)
                else:
                    k_tile = tl.zeros([PS, D], dtype=tl.float32)
                    v_tile = tl.zeros([PS, D], dtype=tl.float32)

    denom = tl.maximum(sum_exp, 1e-6)
    out = (out_vec / denom).to(tl.float32)

    tl.store(Out_ptr + q_base + d_off, out, mask=q_mask)


def get_autotune_configs():
    """Generate dimension-aware autotune configurations."""
    configs = []

    configs.extend(
        [
            triton.Config({}, num_warps=2, num_stages=4),
            triton.Config({}, num_warps=4, num_stages=3),
            triton.Config({}, num_warps=4, num_stages=4),
        ]
    )

    configs.extend(
        [
            triton.Config({}, num_warps=8, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=4),
        ]
    )

    configs.extend(
        [
            triton.Config({}, num_warps=16, num_stages=2),
            triton.Config({}, num_warps=16, num_stages=3),
            triton.Config({}, num_warps=16, num_stages=4),
        ]
    )

    configs.extend(
        [
            triton.Config({}, num_warps=32, num_stages=2),
            triton.Config({}, num_warps=32, num_stages=3),
        ]
    )

    return configs


try:
    _ragged_paged_attn_prefetch_kernel_combined = triton.autotune(
        configs=get_autotune_configs(),
        key=["T", "KVH", "QHG", "D", "PS"],
    )(_ragged_paged_attn_prefetch_kernel_combined)
except Exception:
    pass
