#!/usr/bin/env python3
"""Example: Benchmarking attention algorithms using ejKernel."""

import jax

from ejkernel.benchmarks import Benchmark
from ejkernel.modules import operations


def create_attention_algorithms():
    """Create dictionary of attention algorithm implementations."""

    def vanilla_attention(q, k, v, causal, sliding_windows):
        return operations.attention(q, k, v, causal=causal, sliding_window=sliding_windows)

    def flash_attention(q, k, v, causal, sliding_windows):
        return operations.flash_attention(q, k, v, causal=causal, sliding_window=sliding_windows)

    def sparse_attention(q, k, v, causal, sliding_windows):
        return operations.blocksparse_attention(
            q.transpose(0, 2, 1, 3),
            k.transpose(0, 2, 1, 3),
            v.transpose(0, 2, 1, 3),
            causal=causal,
            sliding_window=sliding_windows,
        ).transpose(0, 2, 1, 3)

    def sdpa_attention(q, k, v, causal, sliding_windows):
        return operations.scaled_dot_product_attention(q, k, v, causal=causal, sliding_window=sliding_windows)

    return {
        "vanilla": vanilla_attention,
        "flash": flash_attention,
        "sparse": sparse_attention,
        "sdpa": sdpa_attention,
    }


def generate_attention_inputs(config):
    """Generate random attention inputs for benchmarking."""

    batch = config["batch"]
    seq = config["seq"]
    dim = config["dim"]

    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 3)

    q = jax.random.normal(keys[0], (batch, seq, config["qheads"], dim), dtype="f2")
    k = jax.random.normal(keys[1], (batch, seq, config["kvheads"], dim), dtype="f2")
    v = jax.random.normal(keys[2], (batch, seq, config["kvheads"], dim), dtype="f2")

    return (q, k, v, config["causal"], config["sliding"])


def main():
    """Run attention algorithm benchmarks."""

    algorithms = create_attention_algorithms()
    configs = [
        {"batch": b, "seq": s, "qheads": qh, "kvheads": kvh, "dim": dim, "causal": iscausal, "sliding": sliding_window}
        for b in [1, 4, 8]
        for s in [1024, 2048, 4096, 8192]
        for qh in [8, 16, 32]
        for kvh in [2, 4, 8]
        for dim in [64, 128]
        for iscausal in [True, False]
        for sliding_window in [None, (256, 256)]
    ]

    bench = Benchmark(
        algorithms=algorithms,
        configs=configs,
        input_generator=generate_attention_inputs,
        warmup=5,
        iterations=50,
        bench_bwd=False,
        static_kwargs=["causal", "sliding_windows"],
        unpack_inputs=True,
    )

    results = bench.run(verbose=True)

    bench.plot("attention_plots")

    return results


if __name__ == "__main__":
    main()
