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


import os

os.environ["EJKERNEL_LOG_AUTOTUNE"] = "1"

import jax
import jax.numpy as jnp

from ejkernel.kernels import xla
from ejkernel.modules import ragged_page_attention


def make_dummy_ragged_paged_attention_inputs(
    key=jax.random.PRNGKey(0),
    *,
    max_num_seqs: int = 3,
    num_seqs: int | None = None,
    pages_per_seq: int = 4,
    page_size: int = 16,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    head_dim: int = 128,
    q_dtype=jnp.bfloat16,
    kv_dtype=jnp.bfloat16,
):
    """
    Returns a dict with:
      - queries: [total_tokens, num_q_heads, head_dim] (q_dtype)
      - kv_pages: [num_pages, page_size, 2 * num_kv_heads, head_dim] (kv_dtype), K/V interleaved (K at 0::2, V at 1::2)
      - context_lens: [max_num_seqs] int32
      - block_tables: [max_num_seqs, pages_per_seq] int32
      - query_start_loc: [max_num_seqs + 1] int32 (cumulative q lengths)
      - num_seqs: [1] int32
      - per_seq_q_lens: [max_num_seqs] int32 (for reference/testing)
    """
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads"

    if num_seqs is None:
        num_seqs = max_num_seqs
    assert 1 <= num_seqs <= max_num_seqs

    total_num_pages = max_num_seqs * pages_per_seq
    num_combined_kv_heads = 2 * num_kv_heads

    base_pages = jnp.arange(total_num_pages, dtype=jnp.int32).reshape(max_num_seqs, pages_per_seq)
    block_tables = base_pages.copy()

    kv_cap = pages_per_seq * page_size
    key_kv, key_q = jax.random.split(key, 2)

    kv_lens = jax.random.randint(key_kv, (num_seqs,), minval=1, maxval=kv_cap + 1, dtype=jnp.int32)

    q_lens_list = []
    cur_key = key_q
    for i in range(num_seqs):
        cur_key, sub = jax.random.split(cur_key)
        q_lens_list.append(jax.random.randint(sub, (), minval=1, maxval=int(kv_lens[i]) + 1, dtype=jnp.int32).item())
    q_lens = jnp.array(q_lens_list, dtype=jnp.int32)

    context_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32).at[:num_seqs].set(kv_lens)
    per_seq_q_lens = jnp.zeros((max_num_seqs,), dtype=jnp.int32).at[:num_seqs].set(q_lens)

    q_cum = [0]
    total_tokens = 0
    for i in range(max_num_seqs):
        if i < num_seqs:
            total_tokens += int(q_lens[i])
        q_cum.append(total_tokens)
    query_start_loc = jnp.array(q_cum, dtype=jnp.int32)

    key_q_data, key_k_data, key_v_data = jax.random.split(cur_key, 3)
    queries = jax.random.normal(key_q_data, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32).astype(q_dtype)

    k_pages = jax.random.normal(
        key_k_data, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(kv_dtype)
    v_pages = jax.random.normal(
        key_v_data, (total_num_pages, page_size, num_kv_heads, head_dim), dtype=jnp.float32
    ).astype(kv_dtype)

    kv_pages = jnp.stack([k_pages, v_pages], axis=3).reshape(total_num_pages, page_size, num_combined_kv_heads, head_dim)

    return dict(
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
        per_seq_q_lens=per_seq_q_lens,
        meta=dict(
            max_num_seqs=max_num_seqs,
            pages_per_seq=pages_per_seq,
            page_size=page_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            total_tokens=total_tokens,
            total_num_pages=total_num_pages,
        ),
    )


batch = make_dummy_ragged_paged_attention_inputs(page_size=64, max_num_seqs=32, pages_per_seq=64)

ref = ragged_page_attention(
    batch["queries"],
    batch["kv_pages"],
    batch["context_lens"],
    batch["block_tables"],
    batch["query_start_loc"],
    batch["num_seqs"],
    sliding_window=None,
    logits_soft_cap=None,
)
out = xla.ragged_page_attention(
    queries=batch["queries"],
    kv_pages=batch["kv_pages"],
    context_lens=batch["context_lens"],
    block_tables=batch["block_tables"],
    query_start_loc=batch["query_start_loc"],
    num_seqs=batch["num_seqs"],
)

print(out.shape)
print(ref.shape)
print(jnp.allclose(out, ref, atol=0.125))
print(out[-1, -1, -5:], ref[-1, -1, -5:])
