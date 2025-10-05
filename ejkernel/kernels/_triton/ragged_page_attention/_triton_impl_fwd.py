# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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
# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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
    Q_ptr,  # float*  [T, KVH, QHG, D]
    KV_pages_ptr,  # float*  [P, PS, 2*KVH, D]
    block_tables_ptr,  # int32*  [S, PAGES_PER_SEQ_MAX]
    row_seq_ptr,  # int32*  [T]
    row_firstk_ptr,  # int32*  [T]
    row_kvlen_ptr,  # int32*  [T]
    row_valid_ptr,  # bool*   [T]
    Out_ptr,  # float*  [T, KVH, QHG, D]
    # constexpr meta
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

    # Precompute strides as constexpr where possible
    sQ_t: tl.constexpr = KVH * QHG * D
    sQ_h: tl.constexpr = QHG * D
    sQ_g: tl.constexpr = D
    C: tl.constexpr = 2 * KVH
    sKV_p: tl.constexpr = PS * C * D
    sKV_s: tl.constexpr = C * D
    sKV_h: tl.constexpr = D

    # Compute head indices once
    head_kv = 2 * h

    q_base = t * sQ_t + h * sQ_h + g * sQ_g
    d_off = tl.arange(0, D)
    ps_off = tl.arange(0, PS)

    # Use custom mask value or default
    NEG_BIG = MASK_VALUE

    # Combined early exit check - load all metadata once
    validrow = tl.load(row_valid_ptr + t, mask=True, other=False)
    seq_idx = tl.load(row_seq_ptr + t, mask=validrow, other=0)
    first_k = tl.load(row_firstk_ptr + t, mask=validrow, other=-1)
    kv_len = tl.load(row_kvlen_ptr + t, mask=validrow, other=0)

    # Single combined early exit condition
    L = tl.minimum(kv_len, first_k + 1)  # number of tokens actually attendable
    should_continue = validrow and (kv_len > 0) and (L > 0)

    if not should_continue:
        # Vectorized store for early exit
        zero_vec = tl.zeros([D], dtype=tl.float32)
        tl.store(Out_ptr + q_base + d_off, zero_vec, mask=(d_off < D))
        return

    # Calculate page info once
    num_pages_seq = (kv_len + PS - 1) // PS
    last_page_needed = tl.minimum(num_pages_seq - 1, first_k // PS)
    n_pages_eff = last_page_needed + 1

    if n_pages_eff <= 0:
        zero_vec = tl.zeros([D], dtype=tl.float32)
        tl.store(Out_ptr + q_base + d_off, zero_vec, mask=(d_off < D))
        return

    # Load Q with potential vectorization hint
    q_mask = d_off < D
    q_vec = tl.load(Q_ptr + q_base + d_off, mask=q_mask, other=0.0, eviction_policy="evict_first").to(tl.float32)

    # Initialize accumulators
    out_vec = tl.zeros([D], dtype=tl.float32)
    sum_exp = tl.zeros((), dtype=tl.float32)
    max_val = tl.full((), NEG_BIG, dtype=tl.float32)

    # page-table row base
    bt_row_ptr = block_tables_ptr + seq_idx * PAGES_PER_SEQ_MAX

    # Precompute mask for negative big values
    neg_big = tl.full([PS], NEG_BIG, dtype=tl.float32)

    # superblock loop (static unroll)
    for kv_super in tl.static_range(MAX_KV_SUPERBLOCKS):
        page_idx_base = kv_super * KV_PAGES_PER_BLOCK
        kv_token_block_start = page_idx_base * PS

        valid_super = page_idx_base < n_pages_eff

        # Prefetch both K and V for first page in superblock (coalesced access)
        if valid_super:
            safe0 = tl.minimum(page_idx_base, PAGES_PER_SEQ_MAX - 1)
            pid0 = tl.load(bt_row_ptr + safe0, mask=True, other=0)

            # Load K and V together for better memory coalescing
            kv_base = KV_pages_ptr + pid0 * sKV_p + ps_off[:, None] * sKV_s + head_kv * sKV_h
            k_tile = tl.load(kv_base + d_off[None, :], eviction_policy="evict_last").to(tl.float32)
            v_tile = tl.load(kv_base + sKV_h + d_off[None, :], eviction_policy="evict_first").to(tl.float32)
        else:
            k_tile = tl.zeros([PS, D], dtype=tl.float32)
            v_tile = tl.zeros([PS, D], dtype=tl.float32)

        # pages in superblock (static unroll)
        for p in tl.static_range(KV_PAGES_PER_BLOCK):
            page_index_idx = page_idx_base + p
            page_start_tok = kv_token_block_start + p * PS

            valid_page = valid_super and (page_index_idx < n_pages_eff)

            # scalar mask: how many tokens in this page are valid
            tokens_in_page = tl.minimum(PS, tl.maximum(0, L - page_start_tok))
            should_process = valid_page and (tokens_in_page > 0)

            if should_process:
                # Compute attention scores using prefetched K tile
                scores = tl.sum(k_tile * q_vec[None, :], axis=1) * SCALE

                # Apply soft cap if enabled
                if USE_SOFT_CAP:
                    # logit_soft_cap * tanh(scores / logit_soft_cap)
                    x = scores / SOFT_CAP
                    x = tl.maximum(tl.minimum(x, 8.0), -8.0)  # clamp for stability
                    exp_2x = tl.exp(2.0 * x)
                    tanh_x = (exp_2x - 1.0) / (exp_2x + 1.0)
                    scores = SOFT_CAP * tanh_x

                # Apply sliding window mask if enabled
                if USE_SLIDING_WINDOW:
                    # Compute absolute token positions
                    token_positions = page_start_tok + ps_off
                    # Current query position is first_k
                    # Sliding window: only attend to tokens within window
                    window_mask = (first_k - token_positions) < SLIDING_WINDOW_SIZE
                    scores = tl.where(window_mask, scores, NEG_BIG)

                # Direct mask computation in where clause
                masked_scores = tl.where(ps_off < tokens_in_page, scores, neg_big)

                # Online softmax
                local_max = tl.max(masked_scores, axis=0)
                new_max = tl.maximum(max_val, local_max)

                # Compute exp values (mixed precision controlled by constexpr)
                exp_old = tl.exp(max_val - new_max)
                score_exp = tl.exp(masked_scores - new_max)

                probs = score_exp * (ps_off < tokens_in_page).to(tl.float32)
                sum_exp = exp_old * sum_exp + tl.sum(probs, axis=0)

                # Value aggregation using prefetched V tile
                val_update = tl.sum(probs[:, None] * v_tile, axis=0)
                out_vec = exp_old * out_vec + val_update

                max_val = new_max

            # Prefetch next K and V tiles together
            if p + 1 < KV_PAGES_PER_BLOCK:
                next_idx = page_idx_base + p + 1
                valid_next = valid_super and (next_idx < n_pages_eff)
                if valid_next:
                    safen = tl.minimum(next_idx, PAGES_PER_SEQ_MAX - 1)
                    pidn = tl.load(bt_row_ptr + safen, mask=True, other=0)

                    # Coalesced load of next K and V
                    kv_base_n = KV_pages_ptr + pidn * sKV_p + ps_off[:, None] * sKV_s + head_kv * sKV_h
                    k_tile = tl.load(kv_base_n + d_off[None, :], eviction_policy="evict_last").to(tl.float32)
                    v_tile = tl.load(kv_base_n + sKV_h + d_off[None, :], eviction_policy="evict_first").to(tl.float32)
                else:
                    k_tile = tl.zeros([PS, D], dtype=tl.float32)
                    v_tile = tl.zeros([PS, D], dtype=tl.float32)

    # Final normalization with better numerical stability
    denom = tl.maximum(sum_exp, 1e-6)
    out = (out_vec / denom).to(tl.float32)

    # Vectorized store
    tl.store(Out_ptr + q_base + d_off, out, mask=q_mask)


def get_autotune_configs():
    """Generate dimension-aware autotune configurations."""
    configs = []

    # Optimized configs for different scenarios
    # Small workloads: Less warps, more stages for memory latency hiding
    configs.extend(
        [
            triton.Config({}, num_warps=2, num_stages=4),
            triton.Config({}, num_warps=4, num_stages=3),
            triton.Config({}, num_warps=4, num_stages=4),
        ]
    )

    # Medium workloads: Balanced warps and stages
    configs.extend(
        [
            triton.Config({}, num_warps=8, num_stages=2),
            triton.Config({}, num_warps=8, num_stages=3),
            triton.Config({}, num_warps=8, num_stages=4),
        ]
    )

    # Large workloads: More warps for compute parallelism
    configs.extend(
        [
            triton.Config({}, num_warps=16, num_stages=2),
            triton.Config({}, num_warps=16, num_stages=3),
            triton.Config({}, num_warps=16, num_stages=4),
        ]
    )

    # Very large workloads: Maximum parallelism
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
    # Fallback if autotune fails - use reasonable defaults
    pass
