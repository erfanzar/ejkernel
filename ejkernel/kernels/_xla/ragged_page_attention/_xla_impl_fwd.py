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


@ejit(static_argnums=(7, 8, 9))
def _ragged_paged_attention(
    queries: jnp.ndarray,
    kv_pages: jnp.ndarray,
    context_lens: jnp.ndarray,
    block_tables: jnp.ndarray,
    query_start_loc: jnp.ndarray,
    num_seqs: jnp.ndarray,
    softmax_scale: float,
    logits_soft_cap: float | None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    sliding_window: int | None = None,
    softmax_aux: jnp.ndarray | None = None,
) -> jnp.ndarray:
    total_query_tokens, num_q_heads, head_size = queries.shape
    page_size = kv_pages.shape[1]
    num_kv_heads = kv_pages.shape[2] // 2
    max_pages_per_sequence = block_tables.shape[-1]
    out_shape = (total_query_tokens, num_q_heads, head_size)
    q_heads_per_group = num_q_heads // num_kv_heads

    queries = queries.reshape(total_query_tokens, num_kv_heads, q_heads_per_group, head_size)
    qblocks = 8 if total_query_tokens >= 8 else max(1, total_query_tokens)
    kvblocks = 64 if max_pages_per_sequence >= 64 else max(1, max_pages_per_sequence)
    queries = queries * softmax_scale

    padd = (-total_query_tokens) % qblocks
    if padd > 0:
        padding_shape = (padd, num_kv_heads, q_heads_per_group, head_size)
        padded_queries = jnp.concatenate([queries, jnp.zeros(padding_shape, dtype=queries.dtype)], axis=0)
    else:
        padded_queries = queries

    attention_output = jnp.zeros_like(padded_queries)

    have_sinks = softmax_aux is not None
    if have_sinks:
        if softmax_aux.ndim == 2:
            num_sinks = softmax_aux.shape[-1]
            if softmax_aux.shape[0] == num_q_heads:
                sinks_h = softmax_aux.reshape(num_kv_heads, q_heads_per_group, num_sinks)
            elif softmax_aux.shape[0] == num_kv_heads:
                sinks_h = jnp.broadcast_to(softmax_aux[:, None, :], (num_kv_heads, q_heads_per_group, num_sinks))
            else:
                raise ValueError(
                    f"softmax_aux first dim must be num_q_heads ({num_q_heads}) "
                    f"or num_kv_heads ({num_kv_heads}), got {softmax_aux.shape}"
                )
        elif softmax_aux.ndim == 1:
            num_sinks = softmax_aux.shape[0]
            sinks_h = jnp.broadcast_to(softmax_aux[None, None, :], (num_kv_heads, q_heads_per_group, num_sinks))
        else:
            raise ValueError(f"Unsupported softmax_aux shape: {softmax_aux.shape}")

    def _compute_attention_for_sequence(seq_idx, output_accumulator):
        num_queries_for_seq = query_start_loc[seq_idx + 1] - query_start_loc[seq_idx]

        def _process_sequence_with_queries():
            num_query_blocks = (num_queries_for_seq + qblocks - 1) // qblocks

            def _process_query_block(query_block_idx, block_output_accumulator):
                query_block_offset = query_block_idx * qblocks
                q_global_start = query_start_loc[seq_idx] + query_block_offset
                query_block = jax.lax.dynamic_slice(
                    padded_queries,
                    (q_global_start, 0, 0, 0),
                    (qblocks, num_kv_heads, q_heads_per_group, head_size),
                )

                kv_cache_len_for_seq = context_lens[seq_idx]

                q_start_tok = kv_cache_len_for_seq - num_queries_for_seq + query_block_offset
                query_token_indices = jnp.arange(qblocks, dtype=jnp.int32) + q_start_tok

                kv_tokens_per_block = page_size * kvblocks
                base_k_ids = jnp.arange(kv_tokens_per_block, dtype=jnp.int32)
                num_kv_blocks = (kv_cache_len_for_seq + kv_tokens_per_block - 1) // kv_tokens_per_block

                def _process_kv_block(kv_block_idx, online_softmax_carry):
                    output_block, sum_exp_block, max_score_block = online_softmax_carry

                    page_map_start = kv_block_idx * kvblocks
                    page_indices_for_block = jax.lax.dynamic_slice(
                        block_tables, (seq_idx, page_map_start), (1, kvblocks)
                    )
                    page_indices_for_kv_block = jnp.squeeze(page_indices_for_block, axis=0)

                    key_block_shape = (kvblocks * page_size, num_kv_heads, head_size)
                    key_block = kv_pages[page_indices_for_kv_block, :, 0::2, :].reshape(key_block_shape)
                    value_block = kv_pages[page_indices_for_kv_block, :, 1::2, :].reshape(key_block_shape)

                    kv_token_start_index = kv_block_idx * kv_tokens_per_block
                    kv_token_indices = base_k_ids + kv_token_start_index

                    attention_scores_block = jnp.einsum(
                        "bihd,kid->bihk",
                        query_block.astype(compute_dtype),
                        key_block.astype(compute_dtype),
                        optimize=True,
                    )
                    if logits_soft_cap is not None:
                        attention_scores_block = jnp.tanh(attention_scores_block / logits_soft_cap) * logits_soft_cap

                    causal_mask = jnp.expand_dims(query_token_indices, 1) >= jnp.expand_dims(kv_token_indices, 0)
                    if sliding_window is not None:
                        left_window = int(sliding_window) if isinstance(sliding_window, int) else int(sliding_window[0])
                        left_keep = jnp.expand_dims(kv_token_indices, 0) > jnp.expand_dims(
                            query_token_indices - left_window, 1
                        )
                        causal_mask = jnp.logical_and(causal_mask, left_keep)
                    kv_bound = jnp.expand_dims(kv_token_indices, 0) < kv_cache_len_for_seq
                    attention_mask = (causal_mask & kv_bound)[:, None, None, :]

                    attention_scores_block = jnp.where(attention_mask, attention_scores_block, -jnp.inf)

                    current_max = jnp.max(attention_scores_block, axis=3)
                    new_max = jnp.maximum(max_score_block, current_max)

                    probs = jnp.exp(attention_scores_block - jnp.expand_dims(new_max, axis=3))
                    probs = jnp.where(attention_mask, probs, 0.0)

                    rescale = jnp.exp(max_score_block - new_max)
                    sum_exp_block = (rescale * sum_exp_block) + jnp.sum(probs, axis=3)
                    value_update = jnp.einsum("bihk,kid->bihd", probs, value_block.astype(compute_dtype), optimize=True)
                    output_block = jnp.expand_dims(rescale, 3) * output_block + value_update

                    return output_block, sum_exp_block, new_max

                init_output_block = jnp.zeros((qblocks, num_kv_heads, q_heads_per_group, head_size), dtype=compute_dtype)
                init_sum_exp = jnp.zeros((qblocks, num_kv_heads, q_heads_per_group), dtype=compute_dtype)
                init_max = jnp.full((qblocks, num_kv_heads, q_heads_per_group), -jnp.inf, dtype=compute_dtype)

                output_block, sum_exp_block, max_block = jax.lax.fori_loop(
                    0, num_kv_blocks, _process_kv_block, (init_output_block, init_sum_exp, init_max)
                )

                if have_sinks:
                    S = sinks_h.shape[-1]
                    sinks_block = jnp.broadcast_to(sinks_h[None, ...], (qblocks, num_kv_heads, q_heads_per_group, S))

                    s_max = jnp.max(sinks_block, axis=3)
                    m_tot = jnp.maximum(max_block, s_max)

                    denom = jnp.exp(max_block - m_tot) * sum_exp_block + jnp.sum(
                        jnp.exp(sinks_block - jnp.expand_dims(m_tot, 3)), axis=3
                    )
                    denom = jnp.maximum(denom, jnp.asarray(1e-6, denom.dtype))
                    normalized_output_block = (jnp.exp(max_block - m_tot)[..., None] * output_block) / denom[..., None]
                else:
                    sum_exp_block = jnp.maximum(sum_exp_block, 1e-6)
                    normalized_output_block = (output_block / jnp.expand_dims(sum_exp_block, axis=3)).astype(
                        padded_queries.dtype
                    )

                normalized_output_block = normalized_output_block.astype(padded_queries.dtype)

                return jax.lax.dynamic_update_slice(
                    block_output_accumulator,
                    normalized_output_block,
                    (q_global_start, 0, 0, 0),
                )

            return jax.lax.fori_loop(0, num_query_blocks, _process_query_block, output_accumulator)

        return jax.lax.cond(
            num_queries_for_seq > 0,
            _process_sequence_with_queries,
            lambda: output_accumulator,
        )

    num_S = (num_seqs[0] if num_seqs.shape != () else num_seqs).astype(jnp.int32)

    return jax.lax.slice(
        jax.lax.fori_loop(0, num_S, _compute_attention_for_sequence, attention_output),
        (0, 0, 0, 0),
        (total_query_tokens, num_kv_heads, q_heads_per_group, head_size),
    ).reshape(out_shape)
