#!/usr/bin/env python3
"""Benchmark GDR: XLA vs Pallas across sequence lengths 2k-64k.

Reports wall-clock time for both forward-only and fwd+bwd.
Generates a 4-panel plot as PNG.
"""

import math
import time

import jax
import jax.numpy as jnp
from jax import random

from ejkernel.modules.operations import gated_delta_rule
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig


def make_inputs(batch, seq_len, num_heads, head_dim, dtype=jnp.bfloat16):
    rng = random.PRNGKey(42)
    q = random.normal(random.fold_in(rng, 0), (batch, seq_len, num_heads, head_dim), dtype=dtype)
    k = random.normal(random.fold_in(rng, 1), (batch, seq_len, num_heads, head_dim), dtype=dtype)
    v = random.normal(random.fold_in(rng, 2), (batch, seq_len, num_heads, head_dim), dtype=dtype)
    beta = jax.nn.sigmoid(random.normal(random.fold_in(rng, 3), (batch, seq_len, num_heads), dtype=dtype))
    decay = random.normal(random.fold_in(rng, 4), (batch, seq_len, num_heads), dtype=dtype) * -0.01
    return q, k, v, beta, decay


def bench(fn, warmup=5, iters=10):
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _make_fwd_fn(plat, c, q, k, v, beta, decay):
    """Create a JIT-compiled forward-only benchmark function."""
    @jax.jit
    def fn(q, k, v, beta, decay):
        return gated_delta_rule(q, k, v, beta, decay, platform=plat, use_chunked=True, cfg=c)
    return lambda: fn(q, k, v, beta, decay)


def _make_fwd_bwd_fn(plat, c, q, k, v, beta, decay):
    """Create a JIT-compiled fwd+bwd benchmark function."""
    @jax.jit
    def step(q, k, v, beta, decay):
        def loss(q, k, v, beta, decay):
            return jnp.sum(gated_delta_rule(q, k, v, beta, decay, platform=plat, use_chunked=True, cfg=c))
        return jax.grad(loss, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, decay)
    return lambda: step(q, k, v, beta, decay)


def run():
    B, H, D = 4, 32, 128
    seq_lens = [2048, 4096, 8192, 16384, 32768, 65536]
    platforms = ["xla", "pallas"]
    chunk_sizes = {"xla": 256, "pallas": 256}
    modes = ["fwd", "fwd+bwd"]

    print(f"Device: {jax.devices()[0]}", flush=True)
    print(f"Config: B={B} H={H} D={D} dtype=bf16", flush=True)
    print(f"Chunk sizes: XLA={chunk_sizes['xla']}, Pallas={chunk_sizes['pallas']}", flush=True)
    print(flush=True)
    print(
        f"{'SeqLen':<8} {'Mode':<8} {'Platform':<8} {'Mean(ms)':<10} {'Min(ms)':<10} {'Status'}",
        flush=True,
    )
    print("-" * 55, flush=True)

    # results[mode][platform] = {"seq_lens": [], "min_ms": []}
    results = {m: {p: {"seq_lens": [], "min_ms": []} for p in platforms} for m in modes}

    for L in seq_lens:
        q, k, v, beta, decay = make_inputs(B, L, H, D)

        for platform in platforms:
            cs = chunk_sizes[platform]
            cfg = GatedDeltaRuleConfig(chunk_size=cs, platform=platform, backend="any")

            for mode in modes:
                try:
                    if mode == "fwd":
                        fn = _make_fwd_fn(platform, cfg, q, k, v, beta, decay)
                    else:
                        fn = _make_fwd_bwd_fn(platform, cfg, q, k, v, beta, decay)

                    jax.block_until_ready(fn())  # compile

                    times = bench(fn, warmup=3, iters=8)
                    mean_ms = sum(times) / len(times) * 1000
                    min_ms = min(times) * 1000

                    results[mode][platform]["seq_lens"].append(L)
                    results[mode][platform]["min_ms"].append(min_ms)

                    print(
                        f"{L:<8} {mode:<8} {platform:<8} {mean_ms:<10.1f} {min_ms:<10.1f} OK",
                        flush=True,
                    )
                except Exception as e:
                    err = str(e).split("\n")[0][:40]
                    print(
                        f"{L:<8} {mode:<8} {platform:<8} {'--':<10} {'--':<10} FAIL: {err}",
                        flush=True,
                    )
                    results[mode][platform]["seq_lens"].append(L)
                    results[mode][platform]["min_ms"].append(float("nan"))

    # Speedup summary
    print(flush=True)
    for mode in modes:
        print(f"Speedup {mode} (XLA / Pallas):", flush=True)
        for i, L in enumerate(seq_lens):
            xla_ms = results[mode]["xla"]["min_ms"][i]
            pal_ms = results[mode]["pallas"]["min_ms"][i]
            if not (math.isnan(xla_ms) or math.isnan(pal_ms)):
                print(f"  L={L}: {xla_ms / pal_ms:.1f}x", flush=True)

    # Generate plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        colors = {"xla": "#2196F3", "pallas": "#FF5722"}
        labels = {"xla": "XLA (chunk=64)", "pallas": "Pallas 2-Phase (chunk=256)"}

        # Plot 1: Forward-only latency
        ax = axes[0, 0]
        for p in platforms:
            sl = results["fwd"][p]["seq_lens"]
            ms = results["fwd"][p]["min_ms"]
            valid = [(s, m) for s, m in zip(sl, ms, strict=False) if not math.isnan(m)]
            if valid:
                ss, mm = zip(*valid, strict=False)
                ax.plot(ss, mm, "o-", color=colors[p], label=labels[p], linewidth=2.5, markersize=7)
        ax.fill_between(
            results["fwd"]["pallas"]["seq_lens"],
            results["fwd"]["pallas"]["min_ms"],
            results["fwd"]["xla"]["min_ms"],
            alpha=0.08, color=colors["pallas"],
        )
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Forward Latency (ms)", fontsize=12)
        ax.set_title("Forward Only (lower is better)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([f"{s // 1024}k" for s in seq_lens], fontsize=10)
        for s, m in zip(results["fwd"]["pallas"]["seq_lens"], results["fwd"]["pallas"]["min_ms"], strict=False):
            if not math.isnan(m):
                ax.annotate(f"{m:.0f}ms", (s, m), textcoords="offset points", xytext=(0, -15),
                            fontsize=9, ha="center", color=colors["pallas"])

        # Plot 2: Fwd+bwd latency
        ax = axes[0, 1]
        for p in platforms:
            sl = results["fwd+bwd"][p]["seq_lens"]
            ms = results["fwd+bwd"][p]["min_ms"]
            valid = [(s, m) for s, m in zip(sl, ms, strict=False) if not math.isnan(m)]
            if valid:
                ss, mm = zip(*valid, strict=False)
                ax.plot(ss, mm, "o-", color=colors[p], label=labels[p], linewidth=2.5, markersize=7)
        ax.fill_between(
            results["fwd+bwd"]["pallas"]["seq_lens"],
            results["fwd+bwd"]["pallas"]["min_ms"],
            results["fwd+bwd"]["xla"]["min_ms"],
            alpha=0.08, color=colors["pallas"],
        )
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("fwd+bwd Latency (ms)", fontsize=12)
        ax.set_title("Forward + Backward (lower is better)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([f"{s // 1024}k" for s in seq_lens], fontsize=10)

        # Plot 3: Forward speedup bars
        ax = axes[1, 0]
        fwd_speedup = []
        for i in range(len(seq_lens)):
            x_ms = results["fwd"]["xla"]["min_ms"][i]
            p_ms = results["fwd"]["pallas"]["min_ms"][i]
            fwd_speedup.append(x_ms / p_ms if not (math.isnan(x_ms) or math.isnan(p_ms)) else 0)
        bars = ax.bar(
            [f"{s // 1024}k" for s in seq_lens], fwd_speedup,
            color=colors["pallas"], alpha=0.85, edgecolor="white", linewidth=1.5,
        )
        for bar, s in zip(bars, fwd_speedup, strict=False):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                        f"{s:.1f}x", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Speedup (XLA / Pallas)", fontsize=12)
        ax.set_title("Forward Speedup", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(fwd_speedup) * 1.25 if fwd_speedup else 1)

        # Plot 4: Fwd+bwd speedup bars
        ax = axes[1, 1]
        bwd_speedup = []
        for i in range(len(seq_lens)):
            x_ms = results["fwd+bwd"]["xla"]["min_ms"][i]
            p_ms = results["fwd+bwd"]["pallas"]["min_ms"][i]
            bwd_speedup.append(x_ms / p_ms if not (math.isnan(x_ms) or math.isnan(p_ms)) else 0)
        bars = ax.bar(
            [f"{s // 1024}k" for s in seq_lens], bwd_speedup,
            color=colors["pallas"], alpha=0.85, edgecolor="white", linewidth=1.5,
        )
        for bar, s in zip(bars, bwd_speedup, strict=False):
            if s > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                        f"{s:.1f}x", ha="center", va="bottom", fontweight="bold", fontsize=12)
        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Speedup (XLA / Pallas)", fontsize=12)
        ax.set_title("Fwd+Bwd Speedup", fontsize=14, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, max(bwd_speedup) * 1.25 if bwd_speedup else 1)

        plt.suptitle(
            "Gated Delta Rule (GDR): 2-Phase Pallas vs XLA on TPU v4\n"
            f"B={B}, H={H}, D={D}, bf16",
            fontsize=15, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig("benchmarks/gdr_benchmark.png", dpi=150, bbox_inches="tight")
        print("\nPlot saved to benchmarks/gdr_benchmark.png", flush=True)

    except ImportError:
        print("\nmatplotlib not available — skipping plots", flush=True)
    except Exception as e:
        print(f"\nPlot error: {e}", flush=True)


if __name__ == "__main__":
    run()
