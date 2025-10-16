#!/usr/bin/env python3
"""Example: Benchmarking attention algorithms using ejKernel."""

import jax
from jax import numpy as jnp

from ejkernel.benchmarks import Benchmark
from ejkernel.modules import operations


def create_attention_algorithms():
    """Create dictionary of attention algorithm implementations."""

    def xla_page(queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs):
        return operations.ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs,
        )

    def ejk_page(queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs):
        return operations.ragged_page_attention(
            queries,
            kv_pages,
            context_lens,
            block_tables,
            query_start_loc,
            num_seqs,
            optimized=True,
        )

    return {
        "eop": xla_page,
        "ejk": ejk_page,
    }


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
    return_dict: bool = False,
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
        key_k_data,
        (total_num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(kv_dtype)
    v_pages = jax.random.normal(
        key_v_data,
        (total_num_pages, page_size, num_kv_heads, head_dim),
        dtype=jnp.float32,
    ).astype(kv_dtype)
    kv_pages = jnp.stack([k_pages, v_pages], axis=3).reshape(total_num_pages, page_size, num_combined_kv_heads, head_dim)
    if return_dict:
        return dict(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            per_seq_q_lens=jnp.zeros((max_num_seqs,), dtype=jnp.int32).at[:num_seqs].set(q_lens),
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
    return (
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        jnp.array([num_seqs], dtype=jnp.int32),
    )


def generate_attention_inputs(config):
    """Generate random attention inputs for benchmarking."""

    return make_dummy_ragged_paged_attention_inputs(
        **{
            "max_num_seqs": config["max_num_seqs"],
            "num_seqs": config["num_seqs"],
            "pages_per_seq": config["pages_per_seq"],
            "page_size": config["page_size"],
            "num_q_heads": config["num_q_heads"],
            "num_kv_heads": config["num_kv_heads"],
            "head_dim": config["head_dim"],
        }
    )


def main():
    """Run attention algorithm benchmarks."""

    algorithms = create_attention_algorithms()
    configs = [
        {
            "max_num_seqs": max_num_seqs,
            "num_seqs": num_seqs,
            "pages_per_seq": pages_per_seq,
            "page_size": page_size,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
        for max_num_seqs in [8, 16, 32]
        for num_seqs in [None]
        for pages_per_seq in [16]
        for page_size in [8, 16, 32, 64, 128]
        for num_q_heads in [16]
        for num_kv_heads in [4]
        for head_dim in [128]
    ]

    bench = Benchmark(
        algorithms=algorithms,
        configs=configs,
        input_generator=generate_attention_inputs,
        warmup=5,
        iterations=50,
        bench_bwd=False,
        unpack_inputs=True,
    )

    results = bench.run(verbose=True)

    bench.plot("attention_plots")

    return results


if __name__ == "__main__":
    main()
