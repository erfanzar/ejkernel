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


"""Fast, minimal benchmarking system for kernel performance comparison.

This module provides a simple API for benchmarking JAX kernels with automatic
performance analysis and comparison metrics using rich for beautiful output.

Example:
    >>>
    >>> def flash_attn(q, k, v, causal, sliding_window):
    ...     return flash_attention(q, k, v, causal, sliding_window)
    >>>
    >>> def sparse_attn(q, k, v, causal, sliding_window):
    ...     return sparse_attention(q, k, v, causal, sliding_window)
    >>>
    >>> algorithms = {"flash": flash_attn, "sparse": sparse_attn}
    >>> configs = [
    ...     {"batch": 4, "seq": 1024, "heads": 8, "dim": 64},
    ...     {"batch": 8, "seq": 2048, "heads": 16, "dim": 128},
    ... ]
    >>> def input_gen(config):
    ...
    ...     q, k, v = generate_qkv(**config)
    ...     return (q, k, v, True, 128)
    >>>
    >>>
    >>> bench = Benchmark(
    ...     algorithms, configs, input_gen,
    ...     static_kwargs=['causal', 'sliding_window']
    ... )
    >>> results = bench.run()
"""

import gc
import json
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text

try:
    import matplotlib.patches as mpatches  # type:ignore
    import matplotlib.pyplot as plt  # type:ignore

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class BenchmarkResult:
    """Encapsulates a single benchmark measurement."""

    algorithm: str
    config: dict[str, Any]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    mean_ms_bwd: float | None = None
    std_ms_bwd: float | None = None
    min_ms_bwd: float | None = None
    max_ms_bwd: float | None = None
    median_ms_bwd: float | None = None

    @property
    def throughput_estimate(self) -> float:
        """Calculate GFLOPS throughput for attention-like operations."""
        if all(k in self.config for k in ["batch", "seq", "heads", "dim"]):
            b, s, h, d = self.config["batch"], self.config["seq"], self.config["heads"], self.config["dim"]
            flops = 4 * b * h * s * s * d
            return flops / (self.mean_ms / 1000) / 1e9
        return 0.0

    @property
    def has_backward(self) -> bool:
        """Check if backward metrics are available."""
        return self.mean_ms_bwd is not None


class Benchmark:
    """Performance benchmarking harness for JAX kernels.

    Provides automated benchmarking with warmup, statistical analysis,
    and comparative performance metrics across multiple algorithms and
    configurations.
    """

    def __init__(
        self,
        algorithms: dict[str, Callable],
        configs: list[dict[str, Any]],
        input_generator: Callable[[dict[str, Any]], Any],
        warmup: int = 5,
        iterations: int = 50,
        bench_bwd: bool = False,
        unpack_inputs: bool = True,
        static_kwargs: list[str] | None = None,
    ):
        """
        Args:
            algorithms: Dict mapping algorithm names to functions
            configs: List of config dicts to test
            input_generator: Function that takes config dict and returns inputs (as tuple if unpack_inputs=True)
            warmup: Number of warmup iterations
            iterations: Number of benchmark iterations
            bench_bwd: If True, also benchmark backward pass
            unpack_inputs: If True, unpack tuple/list from input_generator as *args (default: True)
            static_kwargs: List of argument names to mark as static for JAX JIT
                          (e.g., ['causal', 'sliding_window'])
        """
        self.algorithms = algorithms
        self.configs = configs
        self.input_generator = input_generator
        self.warmup = warmup
        self.iterations = iterations
        self.bench_bwd = bench_bwd
        self.unpack_inputs = unpack_inputs
        self.static_kwargs = static_kwargs or []
        self.results: list[BenchmarkResult] = []
        self.console = Console()

    def _make_scalar_loss(self, output: Any) -> Any:
        """Convert function output to scalar for gradient computation.

        Handles tuple/list outputs by taking first element and applying mean if needed.
        """

        if isinstance(output, tuple | list):
            output = output[0]

        if hasattr(output, "shape") and output.shape != ():
            output = jnp.mean(output)

        return output

    def benchmark_single(self, algo_name: str, algo_fn: Callable, config: dict[str, Any]) -> BenchmarkResult:
        """Execute performance measurement for a single algorithm/config pair.

        Returns BenchmarkResult with timing statistics in milliseconds."""

        inputs = self.input_generator(config)

        if self.static_kwargs:
            compiled_fn = jax.jit(algo_fn, static_argnames=self.static_kwargs)
        else:
            compiled_fn = jax.jit(algo_fn)

        for _ in range(self.warmup):
            if self.unpack_inputs:
                output = compiled_fn(*inputs)
            else:
                output = compiled_fn(inputs)
            if hasattr(output, "block_until_ready"):
                output.block_until_ready()

        gc.collect()

        times_fwd = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            if self.unpack_inputs:
                output = compiled_fn(*inputs)
            else:
                output = compiled_fn(inputs)
            if hasattr(output, "block_until_ready"):
                output.block_until_ready()
            end = time.perf_counter()
            times_fwd.append((end - start) * 1000)

        times_fwd = np.array(times_fwd)

        times_bwd = None
        if self.bench_bwd:
            if self.unpack_inputs:

                def loss_fn(*args):
                    out = algo_fn(*args)
                    return self._make_scalar_loss(out)
            else:

                def loss_fn(inp):
                    out = algo_fn(inp)
                    return self._make_scalar_loss(out)

            if self.static_kwargs:
                grad_fn = jax.jit(jax.grad(loss_fn), static_argnames=self.static_kwargs)
            else:
                grad_fn = jax.jit(jax.grad(loss_fn))

            for _ in range(self.warmup):
                if self.unpack_inputs:
                    grads = grad_fn(*inputs)
                else:
                    grads = grad_fn(inputs)
                if hasattr(grads, "block_until_ready"):
                    grads.block_until_ready()
                elif isinstance(grads, tuple | list):
                    for g in jax.tree_util.tree_leaves(grads):
                        if hasattr(g, "block_until_ready"):
                            g.block_until_ready()

            gc.collect()

            times_bwd = []
            for _ in range(self.iterations):
                start = time.perf_counter()
                if self.unpack_inputs:
                    grads = grad_fn(*inputs)
                else:
                    grads = grad_fn(inputs)
                if hasattr(grads, "block_until_ready"):
                    grads.block_until_ready()
                elif isinstance(grads, tuple | list):
                    for g in jax.tree_util.tree_leaves(grads):
                        if hasattr(g, "block_until_ready"):
                            g.block_until_ready()
                end = time.perf_counter()
                times_bwd.append((end - start) * 1000)

            times_bwd = np.array(times_bwd)

        return BenchmarkResult(
            algorithm=algo_name,
            config=config,
            mean_ms=float(np.mean(times_fwd)),
            std_ms=float(np.std(times_fwd)),
            min_ms=float(np.min(times_fwd)),
            max_ms=float(np.max(times_fwd)),
            median_ms=float(np.median(times_fwd)),
            mean_ms_bwd=float(np.mean(times_bwd)) if times_bwd is not None else None,
            std_ms_bwd=float(np.std(times_bwd)) if times_bwd is not None else None,
            min_ms_bwd=float(np.min(times_bwd)) if times_bwd is not None else None,
            max_ms_bwd=float(np.max(times_bwd)) if times_bwd is not None else None,
            median_ms_bwd=float(np.median(times_bwd)) if times_bwd is not None else None,
        )

    def run(self, verbose: bool = True) -> dict[str, Any]:
        """Execute complete benchmark suite and generate performance analysis.

        Returns dict containing summary statistics, speedup comparisons,
        and per-configuration results."""

        self.results = []
        total = len(self.algorithms) * len(self.configs)

        if verbose:
            header_text = Text("RUNNING BENCHMARKS", style="bold white")
            stats_text = (
                f"Algorithms: {len(self.algorithms)} | Configurations: {len(self.configs)} | Total runs: {total}"
            )
            panel = Panel(
                Text(stats_text, justify="center"),
                title=header_text,
                box=box.DOUBLE,
                style="cyan",
                padding=(1, 2),
            )
            self.console.print(panel)

        if verbose:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[config]}", justify="right"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                TextColumn("[cyan]{task.fields[algo]}"),
                TextColumn("•"),
                TextColumn("{task.fields[status]}"),
                console=self.console,
            )

            with progress:
                task = progress.add_task(
                    "[cyan]Benchmarking...",
                    total=total,
                    config="",
                    algo="",
                    status="",
                )

                for config in self.configs:
                    config_str = " • ".join([f"{k}={v}" for k, v in config.items()])

                    for algo_name, algo_fn in self.algorithms.items():
                        progress.update(
                            task,
                            advance=0,
                            config=config_str,
                            algo=algo_name,
                            status="[yellow]Running...[/yellow]",
                        )

                        try:
                            result = self.benchmark_single(algo_name, algo_fn, config)
                            self.results.append(result)

                            if result.mean_ms < 10:
                                icon = "⚡"
                                color = "green"
                            elif result.mean_ms < 50:
                                icon = "✓"
                                color = "green"
                            elif result.mean_ms < 100:
                                icon = "→"
                                color = "yellow"
                            else:
                                icon = "⚠"
                                color = "red"

                            fwd_str = f"{icon} FWD: {result.mean_ms:.2f}ms"
                            if result.has_backward:
                                bwd_str = f"BWD: {result.mean_ms_bwd:.2f}ms"
                                status = f"[{color}]{fwd_str} | {bwd_str}[/{color}]"
                            else:
                                status = f"[{color}]{fwd_str}[/{color}]"

                            progress.update(task, advance=1, status=status)

                        except Exception as e:
                            traceback.print_exc()
                            error_msg = str(e)[:30] + "..." if len(str(e)) > 30 else str(e)
                            progress.update(
                                task,
                                advance=1,
                                status=f"[red]✗ FAILED: {error_msg}[/red]",
                            )

                            self.results.append(
                                BenchmarkResult(
                                    algorithm=algo_name,
                                    config=config,
                                    mean_ms=float("inf"),
                                    std_ms=0,
                                    min_ms=float("inf"),
                                    max_ms=float("inf"),
                                    median_ms=float("inf"),
                                )
                            )
        else:
            for config in self.configs:
                for algo_name, algo_fn in self.algorithms.items():
                    try:
                        result = self.benchmark_single(algo_name, algo_fn, config)
                        self.results.append(result)
                    except Exception:
                        self.results.append(
                            BenchmarkResult(
                                algorithm=algo_name,
                                config=config,
                                mean_ms=float("inf"),
                                std_ms=0,
                                min_ms=float("inf"),
                                max_ms=float("inf"),
                                median_ms=float("inf"),
                            )
                        )

        analysis = self.analyze()

        if verbose:
            self.print_analysis(analysis)

        return analysis

    def analyze(self) -> dict[str, Any]:
        """Generate comprehensive performance analysis from benchmark results.

        Returns structured analysis including algorithm summaries, speedup
        comparisons, and configuration-specific rankings."""

        analysis = {
            "summary": {},
            "comparisons": {},
            "by_config": {},
            "raw_results": [
                {
                    "algorithm": r.algorithm,
                    "config": r.config,
                    "mean_ms": r.mean_ms,
                    "std_ms": r.std_ms,
                    "min_ms": r.min_ms,
                    "throughput_gflops": r.throughput_estimate,
                }
                for r in self.results
            ],
        }

        by_algo = defaultdict(list)
        by_config = defaultdict(list)

        for result in self.results:
            if result.mean_ms != float("inf"):
                by_algo[result.algorithm].append(result)
                config_key = str(sorted(result.config.items()))
                by_config[config_key].append(result)

        for algo, results in by_algo.items():
            times = [r.mean_ms for r in results]
            throughputs = [r.throughput_estimate for r in results]

            analysis["summary"][algo] = {
                "mean_time_ms": np.mean(times),
                "median_time_ms": np.median(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "mean_throughput_gflops": np.mean(throughputs) if throughputs else 0,
                "success_rate": len(results) / len([r for r in self.results if r.algorithm == algo]),
                "num_configs": len(results),
            }

        if len(by_algo) > 1:
            baseline_algo = next(iter(by_algo.keys()))
            baseline_results = {str(sorted(r.config.items())): r.mean_ms for r in by_algo[baseline_algo]}

            for algo in by_algo:
                if algo == baseline_algo:
                    continue

                speedups = []
                for result in by_algo[algo]:
                    config_key = str(sorted(result.config.items()))
                    if config_key in baseline_results:
                        speedup = baseline_results[config_key] / result.mean_ms
                        speedups.append(speedup)

                if speedups:
                    analysis["comparisons"][f"{algo}_vs_{baseline_algo}"] = {
                        "mean_speedup": np.mean(speedups),
                        "median_speedup": np.median(speedups),
                        "min_speedup": np.min(speedups),
                        "max_speedup": np.max(speedups),
                    }

        for config_key, results in by_config.items():
            config_items = eval(config_key)
            config_dict = dict(config_items)

            analysis["by_config"][config_key] = {
                "config": config_dict,
                "results": {
                    r.algorithm: {
                        "mean_ms": r.mean_ms,
                        "throughput_gflops": r.throughput_estimate,
                    }
                    for r in results
                },
                "fastest": min(results, key=lambda r: r.mean_ms).algorithm,
                "fastest_time_ms": min(r.mean_ms for r in results),
            }

        return analysis

    def print_analysis(self, analysis: dict[str, Any]):
        """Display formatted benchmark analysis to console using rich."""

        self.console.print()
        panel = Panel(
            Text("BENCHMARK ANALYSIS", justify="center", style="bold white"),
            box=box.DOUBLE,
            style="magenta",
            padding=(1, 2),
        )
        self.console.print(panel)

        has_bwd = any(r.has_backward for r in self.results)

        summary_table = Table(
            title="Algorithm Performance Summary" + (" (FWD + BWD)" if has_bwd else " (FWD)"),
            box=box.ROUNDED,
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
        )

        summary_table.add_column("Algorithm", style="cyan", no_wrap=True)
        summary_table.add_column("FWD Time", justify="right", style="white")
        if has_bwd:
            summary_table.add_column("BWD Time", justify="right", style="magenta")
            summary_table.add_column("Total", justify="right", style="yellow")
        summary_table.add_column("Range", justify="center", style="dim")
        summary_table.add_column("Throughput", justify="right", style="yellow")
        summary_table.add_column("Success", justify="center")
        summary_table.add_column("Runs", justify="center", style="dim")

        for algo, stats in analysis["summary"].items():
            fwd_time = f"{stats['mean_time_ms']:.2f}ms"
            time_range = f"±{(stats['max_time_ms'] - stats['min_time_ms']) / 2:.2f}ms"

            throughput = ""
            if stats["mean_throughput_gflops"] > 0:
                throughput = f"{stats['mean_throughput_gflops']:.1f} GFLOPS"

            success_pct = stats["success_rate"] * 100
            if success_pct == 100:
                success = Text(f"✓ {success_pct:.0f}%", style="green")
            else:
                success = Text(f"⚠ {success_pct:.0f}%", style="yellow")

            runs = f"{stats['num_configs']}"

            if has_bwd:
                algo_results = [r for r in self.results if r.algorithm == algo and r.has_backward]
                if algo_results:
                    mean_bwd = np.mean([r.mean_ms_bwd for r in algo_results])
                    mean_fwd = stats["mean_time_ms"]
                    bwd_time = f"{mean_bwd:.2f}ms"
                    total_time = f"{mean_fwd + mean_bwd:.2f}ms"
                    summary_table.add_row(algo, fwd_time, bwd_time, total_time, time_range, throughput, success, runs)
                else:
                    summary_table.add_row(algo, fwd_time, "-", "-", time_range, throughput, success, runs)
            else:
                summary_table.add_row(algo, fwd_time, time_range, throughput, success, runs)

        self.console.print(summary_table)

        if analysis["comparisons"]:
            speedup_table = Table(
                title="Relative Performance (Speedup)",
                box=box.ROUNDED,
                title_style="bold cyan",
                show_header=True,
                header_style="bold",
            )

            speedup_table.add_column("Comparison", style="cyan")
            speedup_table.add_column("Mean Speedup", justify="right")
            speedup_table.add_column("Range", justify="center", style="dim")

            for comparison, stats in analysis["comparisons"].items():
                parts = comparison.replace("_vs_", " vs ")
                mean_sp = stats["mean_speedup"]

                if mean_sp >= 2.0:
                    indicator = "🚀"
                    style = "bright_green"
                elif mean_sp >= 1.5:
                    indicator = "⚡"
                    style = "green"
                elif mean_sp >= 1.1:
                    indicator = "↑"
                    style = "yellow"
                elif mean_sp >= 0.9:
                    indicator = "≈"
                    style = "white"
                else:
                    indicator = "↓"
                    style = "red"

                comparison_text = f"{indicator} {parts}"
                mean_text = Text(f"{mean_sp:.2f}x", style=style)
                range_text = f"[{stats['min_speedup']:.2f}x - {stats['max_speedup']:.2f}x]"

                speedup_table.add_row(comparison_text, mean_text, range_text)

            self.console.print(speedup_table)

        config_table = Table(
            title="Fastest Algorithm by Configuration",
            box=box.ROUNDED,
            title_style="bold cyan",
            show_header=True,
            header_style="bold",
        )

        config_table.add_column("Configuration", style="yellow")
        config_table.add_column("Winner", style="green")
        config_table.add_column("Time", justify="right")
        config_table.add_column("vs Others", style="dim")

        for config_data in analysis["by_config"].values():
            config_items = [f"{k}={v}" for k, v in config_data["config"].items()]
            config_str = " • ".join(config_items)

            winner = config_data["fastest"]
            time_ms = config_data["fastest_time_ms"]
            winner_text = f"🏆 {winner}"
            time_text = f"{time_ms:.2f}ms"

            other_results = []
            for algo, result in config_data["results"].items():
                if algo != winner:
                    speedup = time_ms / result["mean_ms"] if result["mean_ms"] != float("inf") else 0
                    if speedup > 0:
                        other_results.append(f"{algo} ({speedup:.1f}x slower)")

            others_text = ", ".join(other_results) if other_results else "-"

            config_table.add_row(config_str, winner_text, time_text, others_text)

        self.console.print(config_table)

    def save(self, filepath: str = "benchmark_results.json"):
        """Export benchmark results and analysis to JSON file."""

        analysis = self.analyze()
        output = {
            "metadata": {
                "num_algorithms": len(self.algorithms),
                "num_configs": len(self.configs),
                "warmup": self.warmup,
                "iterations": self.iterations,
            },
            "analysis": analysis,
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2, default=str)

    def plot(self, output_dir: str = "benchmark_plots", figsize: tuple[int, int] = (14, 7)):
        """Create column-style comparison plots for each configuration.

        Args:
            output_dir: Directory to save plot images (one per config)
            figsize: Figure size as (width, height) tuple
        """
        if not HAS_MATPLOTLIB:
            self.console.print("[red]Error: matplotlib is not installed. Install it with: pip install matplotlib[/red]")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        algorithms_list = list(self.algorithms.keys())
        has_bwd = any(r.has_backward for r in self.results)

        by_config = defaultdict(dict)
        config_keys = []
        for result in self.results:
            if result.mean_ms != float("inf"):
                config_key = tuple(sorted(result.config.items()))
                if config_key not in config_keys:
                    config_keys.append(config_key)
                by_config[config_key][result.algorithm] = result

        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms_list)))

        saved_files = []
        for config_key in config_keys:
            results_dict = by_config[config_key]

            filename_parts = [f"{k}_{v}" for k, v in config_key]
            filename = "_".join(filename_parts) + ".png"
            filepath = output_path / filename

            if has_bwd:
                fig, (ax_fwd, ax_bwd) = plt.subplots(1, 2, figsize=figsize)
            else:
                fig, ax_fwd = plt.subplots(1, 1, figsize=(figsize[0] * 0.6, figsize[1]))

            fwd_times = []
            bwd_times = []
            algo_names = []

            for algo in algorithms_list:
                if algo in results_dict:
                    result = results_dict[algo]
                    fwd_times.append(result.mean_ms)
                    if result.has_backward:
                        bwd_times.append(result.mean_ms_bwd)
                    else:
                        bwd_times.append(0)
                    algo_names.append(algo)

            x_pos = np.arange(len(algo_names))
            bar_width = 0.65

            bars_fwd = ax_fwd.bar(
                x_pos, fwd_times, bar_width, color=colors[: len(algo_names)], alpha=0.8, edgecolor="black"
            )
            ax_fwd.set_xticks(x_pos)
            ax_fwd.set_xticklabels(algo_names, rotation=45, ha="right")
            ax_fwd.set_ylabel("Time (ms)", fontweight="bold", fontsize=12)
            ax_fwd.set_title("Forward Pass", fontweight="bold", fontsize=12)
            ax_fwd.grid(axis="y", alpha=0.3, linestyle="--")

            ax_fwd.set_xlim(-0.8, len(algo_names) - 0.2)
            ax_fwd.margins(x=0.15)

            for bar, val in zip(bars_fwd, fwd_times, strict=False):
                height = bar.get_height()
                ax_fwd.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            if fwd_times:
                fastest_idx = np.argmin(fwd_times)
                bars_fwd[fastest_idx].set_edgecolor("gold")
                bars_fwd[fastest_idx].set_linewidth(3)

            if has_bwd and any(bwd_times):
                bars_bwd = ax_bwd.bar(
                    x_pos, bwd_times, bar_width, color=colors[: len(algo_names)], alpha=0.8, edgecolor="black"
                )
                ax_bwd.set_xticks(x_pos)
                ax_bwd.set_xticklabels(algo_names, rotation=45, ha="right")
                ax_bwd.set_ylabel("Time (ms)", fontweight="bold", fontsize=12)
                ax_bwd.set_title("Backward Pass", fontweight="bold", fontsize=12)
                ax_bwd.grid(axis="y", alpha=0.3, linestyle="--")

                ax_bwd.set_xlim(-0.8, len(algo_names) - 0.2)
                ax_bwd.margins(x=0.15)

                for bar, val in zip(bars_bwd, bwd_times, strict=False):
                    if val > 0:
                        height = bar.get_height()
                        ax_bwd.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height,
                            f"{val:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                        )

                fastest_bwd_idx = np.argmin([t if t > 0 else float("inf") for t in bwd_times])
                if bwd_times[fastest_bwd_idx] > 0:
                    bars_bwd[fastest_bwd_idx].set_edgecolor("gold")
                    bars_bwd[fastest_bwd_idx].set_linewidth(3)

            config_title = " • ".join([f"{k}={v}" for k, v in config_key])
            fig.suptitle(config_title, fontsize=14, fontweight="bold")

            gold_patch = mpatches.Patch(edgecolor="gold", facecolor="lightgray", linewidth=3, label="Fastest")
            fig.legend(handles=[gold_patch], loc="upper right", frameon=True, fontsize=10)

            plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], pad=2.0)

            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            saved_files.append(str(filepath))

        self.console.print(f"[green]✓ Created {len(saved_files)} plot(s) in: {output_dir}/[/green]")
        for file in saved_files:
            self.console.print(f"  [dim]• {Path(file).name}[/dim]")
