#!/usr/bin/env python3
"""Extended GDR benchmark with paper-quality plots using ejkernel.benchmarks."""

import math
import os
import sys

import jax
import jax.numpy as jnp
from jax import random

sys.path.append(os.path.dirname(__file__))

from _op_benchmark_registry import _gen_gdr_inputs, _grid, _limit_configs

from ejkernel.benchmarks import Benchmark
from ejkernel.modules.operations import gated_delta_rule
from ejkernel.modules.operations.configs import GatedDeltaRuleConfig

OUT_DIR = "benchmark_plots/gated_delta_rule_extended"
OOM = float("inf")


def _gdr_configs():
    return _limit_configs(
        _grid(batch=[4], seq=[2048, 4096, 8192, 16384, 32768], heads=[32], dim=[128])
    )


def _gdr_wrapper(platform, chunk_size=256):
    cfg = GatedDeltaRuleConfig(chunk_size=chunk_size, platform=platform, backend="any")

    def fn(q, k, v, beta, decay):
        out = gated_delta_rule(q, k, v, beta, decay, platform=platform, use_chunked=True, cfg=cfg)
        return out[0] if isinstance(out, tuple) else out

    return fn


def _run_main_benchmark():
    configs = _gdr_configs()
    algorithms = {
        "XLA (chunk=256)": _gdr_wrapper("xla", 256),
        "Pallas 2-Phase (chunk=256)": _gdr_wrapper("pallas", 256),
    }

    bench_fwd = Benchmark(
        algorithms=algorithms,
        configs=configs,
        input_generator=_gen_gdr_inputs,
        warmup=5,
        iterations=8,
        bench_bwd=False,
    )
    bench_fwd.run(verbose=True)
    bench_fwd.plot(OUT_DIR)
    bench_fwd.save(f"{OUT_DIR}/gdr_fwd_results.json")

    bench_train = Benchmark(
        algorithms=algorithms,
        configs=configs,
        input_generator=_gen_gdr_inputs,
        warmup=5,
        iterations=8,
        bench_bwd=True,
    )
    bench_train.run(verbose=True)
    bench_train.plot(OUT_DIR)
    bench_train.save(f"{OUT_DIR}/gdr_train_results.json")

    return bench_fwd, bench_train


def _run_chunk_sweep():
    configs = _limit_configs(_grid(batch=[4], seq=[16384], heads=[32], dim=[128]))
    algorithms = {}
    for cs in [64, 128, 256, 512]:
        algorithms[f"C={cs}"] = _gdr_wrapper("pallas", cs)

    bench = Benchmark(
        algorithms=algorithms,
        configs=configs,
        input_generator=_gen_gdr_inputs,
        warmup=5,
        iterations=8,
        bench_bwd=True,
    )
    bench.run(verbose=True)
    bench.plot(OUT_DIR)
    bench.save(f"{OUT_DIR}/gdr_chunk_sweep.json")
    return bench


def _run_numerical_error():
    B, H, D = 4, 32, 128
    seq_lens = [2048, 4096, 8192, 16384, 32768]
    chunk_sizes = [64, 128, 256, 512]
    error_results = {}

    print("\n=== Numerical Error (Pallas vs XLA exact solve) ===", flush=True)
    for cs in chunk_sizes:
        error_results[cs] = {"seq_lens": [], "mean_err": [], "max_err": []}
        for L in seq_lens:
            try:
                rng = random.PRNGKey(42)
                q = random.normal(random.fold_in(rng, 0), (B, L, H, D), dtype=jnp.bfloat16)
                k = random.normal(random.fold_in(rng, 1), (B, L, H, D), dtype=jnp.bfloat16)
                v = random.normal(random.fold_in(rng, 2), (B, L, H, D), dtype=jnp.bfloat16)
                beta = jax.nn.sigmoid(
                    random.normal(random.fold_in(rng, 3), (B, L, H), dtype=jnp.bfloat16)
                )
                decay = random.normal(random.fold_in(rng, 4), (B, L, H), dtype=jnp.bfloat16) * -0.01

                cfg_xla = GatedDeltaRuleConfig(chunk_size=cs, platform="xla", backend="any")
                cfg_pal = GatedDeltaRuleConfig(chunk_size=cs, platform="pallas", backend="any")
                out_xla = jax.block_until_ready(
                    gated_delta_rule(q, k, v, beta, decay, platform="xla", use_chunked=True, cfg=cfg_xla)
                )
                out_pal = jax.block_until_ready(
                    gated_delta_rule(q, k, v, beta, decay, platform="pallas", use_chunked=True, cfg=cfg_pal)
                )
                diff = jnp.abs(out_xla.astype(jnp.float32) - out_pal.astype(jnp.float32))
                me, mx = float(jnp.mean(diff)), float(jnp.max(diff))
                error_results[cs]["seq_lens"].append(L)
                error_results[cs]["mean_err"].append(me)
                error_results[cs]["max_err"].append(mx)
                print(f"  C={cs:<4} L={L:<6} mean={me:.6f} max={mx:.6f}", flush=True)
            except Exception:
                error_results[cs]["seq_lens"].append(L)
                error_results[cs]["mean_err"].append(None)
                error_results[cs]["max_err"].append(None)
                print(f"  C={cs:<4} L={L:<6} FAIL", flush=True)
    return error_results


def _extract_results(bench, mode="fwd"):
    seq_lens = sorted({r.config["seq"] for r in bench.results})
    algo_names = sorted({r.algorithm for r in bench.results})
    data = {}
    for algo in algo_names:
        data[algo] = {"seq_lens": [], "min_ms": []}
        for sl in seq_lens:
            matches = [r for r in bench.results if r.algorithm == algo and r.config["seq"] == sl]
            if matches:
                r = matches[0]
                if mode == "fwd":
                    ms = r.min_ms
                else:
                    ms = r.min_ms_bwd if r.min_ms_bwd is not None else r.min_ms
                data[algo]["seq_lens"].append(sl)
                data[algo]["min_ms"].append(ms)
    return seq_lens, data


def _savefig(fig, name):
    from pathlib import Path
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    path = f"{OUT_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved {path}", flush=True)


def _plot_all(bench_fwd, bench_train, bench_chunk, error_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    colors = {"xla": "#2196F3", "pallas": "#FF5722"}
    oom_color = "#BDBDBD"

    fwd_seqs, fwd_data = _extract_results(bench_fwd, "fwd")
    _, train_data = _extract_results(bench_train, "fwd")
    x_labels = [f"{s // 1024}k" for s in fwd_seqs]

    xla_key = next(k for k in fwd_data if "XLA" in k)
    pal_key = next(k for k in fwd_data if "Pallas" in k)

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))

    for panel_idx, (data, title_prefix, ylabel) in enumerate([
        (fwd_data, "Forward Only", "Forward Latency (ms)"),
        (train_data, "Forward + Backward", "Fwd+Bwd Latency (ms)"),
    ]):
        ax = axes[panel_idx]
        for algo, c in [(xla_key, colors["xla"]), (pal_key, colors["pallas"])]:
            sl = data[algo]["seq_lens"]
            ms = data[algo]["min_ms"]
            valid = [
                (s, m) for s, m in zip(sl, ms, strict=False)
                if m is not None and math.isfinite(m)
            ]
            if valid:
                ss, mm = zip(*valid, strict=False)
                ax.plot(ss, mm, "o-", color=c, label=algo, linewidth=2.5, markersize=7)

        if panel_idx == 0:
            for s, m in zip(
                data[pal_key]["seq_lens"], data[pal_key]["min_ms"], strict=False,
            ):
                if m is not None and math.isfinite(m):
                    ax.annotate(
                        f"{m:.0f}ms", (s, m), textcoords="offset points",
                        xytext=(0, -15), fontsize=9, ha="center", color=colors["pallas"],
                    )

        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{title_prefix} (lower is better)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fwd_seqs)
        ax.set_xticklabels(x_labels, fontsize=10)

    for panel_idx, data in enumerate([fwd_data, train_data]):
        ax = axes[2 + panel_idx]
        speedups = []
        oom_mask = []
        for i in range(len(fwd_seqs)):
            x_ms = data[xla_key]["min_ms"][i] if i < len(data[xla_key]["min_ms"]) else None
            p_ms = data[pal_key]["min_ms"][i] if i < len(data[pal_key]["min_ms"]) else None
            x_ok = x_ms is not None and math.isfinite(x_ms)
            p_ok = p_ms is not None and math.isfinite(p_ms)
            if not x_ok and p_ok:
                speedups.append(0)
                oom_mask.append(True)
            elif x_ok and p_ok:
                speedups.append(x_ms / p_ms)
                oom_mask.append(False)
            else:
                speedups.append(0)
                oom_mask.append(False)

        max_real = max(
            (s for s, o in zip(speedups, oom_mask, strict=False) if not o and s > 0),
            default=1,
        )
        oom_h = max_real * 1.3
        for i, (lbl, spd, is_oom) in enumerate(
            zip(x_labels, speedups, oom_mask, strict=False),
        ):
            if is_oom:
                ax.bar(
                    lbl, oom_h, color=oom_color, edgecolor=colors["pallas"],
                    linewidth=2, hatch="///", alpha=0.6,
                )
                ax.text(
                    i, oom_h + 0.1, "XLA\nOOM", ha="center", va="bottom",
                    fontweight="bold", fontsize=10, color="#D32F2F",
                )
            elif spd > 0:
                ax.bar(
                    lbl, spd, color=colors["pallas"], alpha=0.85,
                    edgecolor="white", linewidth=1.5,
                )
                ax.text(
                    i, spd + 0.08, f"{spd:.1f}x", ha="center", va="bottom",
                    fontweight="bold", fontsize=12,
                )

        ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
        title = "Forward Speedup" if panel_idx == 0 else "Fwd+Bwd Speedup"
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Speedup (XLA / Pallas)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(0, oom_h * 1.25 if any(oom_mask) else max_real * 1.3)
        if any(oom_mask):
            oom_patch = mpatches.Patch(
                facecolor=oom_color, edgecolor=colors["pallas"],
                hatch="///", alpha=0.6, label="XLA OOM (Pallas OK)",
            )
            ax.legend(handles=[oom_patch], fontsize=9)

    fig.suptitle(
        "Gated Delta Rule (GDR): 2-Phase Pallas vs XLA on TPU v4"
        "  |  B=4, H=32, D=128, bf16",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, "gdr_benchmark.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    bar_width = 0.35
    x_pos = list(range(len(fwd_seqs)))
    for pi, (algo, c) in enumerate([(xla_key, colors["xla"]), (pal_key, colors["pallas"])]):
        throughputs = []
        for i, L in enumerate(fwd_seqs):
            ms = fwd_data[algo]["min_ms"][i] if i < len(fwd_data[algo]["min_ms"]) else None
            if ms is not None and math.isfinite(ms) and ms > 0:
                throughputs.append((4 * L) / (ms / 1000) / 1e6)
            else:
                throughputs.append(0)
        offsets = [x + pi * bar_width for x in x_pos]
        for off, thr in zip(offsets, throughputs, strict=False):
            if thr == 0:
                ax.bar(
                    off, 0.01, bar_width, color=oom_color, edgecolor=c,
                    hatch="///", alpha=0.4, linewidth=1.5,
                )
            else:
                ax.bar(off, thr, bar_width, color=c, alpha=0.85, edgecolor="white", linewidth=1)

    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Throughput (M tokens/sec)", fontsize=12)
    ax.set_title("Forward Throughput (higher is better)", fontsize=13, fontweight="bold")
    ax.set_xticks([x + bar_width / 2 for x in x_pos])
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.legend(
        [mpatches.Patch(color=colors["xla"], alpha=0.85),
         mpatches.Patch(color=colors["pallas"], alpha=0.85)],
        [xla_key, pal_key], fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    _, chunk_data = _extract_results(bench_chunk, "fwd")
    cs_names = sorted(chunk_data.keys(), key=lambda x: int(x.split("=")[1]))
    for ci, cs_name in enumerate(cs_names):
        r_fwd = [r for r in bench_chunk.results if r.algorithm == cs_name]
        if r_fwd:
            fwd_ms = r_fwd[0].min_ms
            bwd_ms = r_fwd[0].min_ms_bwd if r_fwd[0].min_ms_bwd else 0
            ax.bar(ci - 0.1, fwd_ms, 0.2, color=colors["pallas"], alpha=0.85, edgecolor="white")
            ax.text(ci - 0.1, fwd_ms + 1, f"{fwd_ms:.0f}", ha="center", va="bottom", fontsize=9)
            if bwd_ms > 0:
                ax.bar(ci + 0.1, bwd_ms, 0.2, color="#FF9800", alpha=0.85, edgecolor="white")
                ax.text(ci + 0.1, bwd_ms + 1, f"{bwd_ms:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Chunk Size", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Pallas Chunk Size Sweep (L=16k)", fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(cs_names)))
    ax.set_xticklabels([n.split("=")[1] for n in cs_names], fontsize=11)
    ax.legend(
        [mpatches.Patch(color=colors["pallas"], alpha=0.85),
         mpatches.Patch(color="#FF9800", alpha=0.85)],
        ["Forward", "Fwd+Bwd"], fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "GDR Throughput & Chunk Sensitivity  |  B=4, H=32, D=128, bf16",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, "gdr_throughput_chunk.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    chunk_colors = {64: "#4CAF50", 128: "#FF9800", 256: "#FF5722", 512: "#9C27B0"}

    for ax_idx, (err_key, err_label) in enumerate([
        ("mean_err", "Mean Absolute Error"),
        ("max_err", "Max Absolute Error"),
    ]):
        ax = axes[ax_idx]
        for cs in sorted(error_results.keys()):
            sl = error_results[cs]["seq_lens"]
            errs = error_results[cs][err_key]
            valid = [(s, e) for s, e in zip(sl, errs, strict=False) if e is not None]
            if valid:
                ss, ee = zip(*valid, strict=False)
                ax.plot(
                    ss, ee, "o-", color=chunk_colors[cs],
                    label=f"C={cs}", linewidth=2, markersize=6,
                )
        ax.axhline(y=0.02, color="gray", linestyle="--", alpha=0.5, label="bf16 tol (0.02)")
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel(err_label, fontsize=12)
        ax.set_title(
            f"{err_label} (Pallas vs XLA exact solve)",
            fontsize=13, fontweight="bold",
        )
        ax.legend(fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(fwd_seqs)
        ax.set_xticklabels(x_labels, fontsize=10)

    fig.suptitle(
        "Numerical Fidelity: Neumann Approximation vs Exact Triangular Solve"
        "  |  B=4, H=32, D=128, bf16",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, "gdr_numerical_error.png")
    plt.close(fig)


def run():
    bench_fwd, bench_train = _run_main_benchmark()
    bench_chunk = _run_chunk_sweep()
    error_results = _run_numerical_error()

    try:
        _plot_all(bench_fwd, bench_train, bench_chunk, error_results)
        print("\nAll plots saved.", flush=True)
    except ImportError:
        print("\nmatplotlib not available — skipping plots", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nPlot error: {e}", flush=True)


if __name__ == "__main__":
    run()
