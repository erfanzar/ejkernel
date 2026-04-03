#!/usr/bin/env python3
"""Run quant/dequant microbenchmarks and enforce regression gates.

This is a local/dev perf gate (no CI workflow dependency). It runs
`benchmark_quantize.py` and `benchmark_dequantize.py`, then compares medians
against provided baseline JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path


def _normalize_backend(name: str) -> str:
    name = str(name).strip().lower()
    if name == "cuda":
        return "gpu"
    if name in {"gpu", "tpu", "cpu", "mps"}:
        return name
    return "gpu"


def _default_backend_slug() -> str:
    """Best-effort backend detection without importing JAX.

    This script spawns subprocesses that initialize JAX. Importing/initializing
    JAX in the parent process can lock exclusive devices (notably TPU) and make
    the subprocess benchmarks fail. Keep detection env/probe-only.
    """
    override = os.getenv("EJKERNEL_PERF_GATE_BACKEND") or os.getenv("EJKERNEL_BASELINE_BACKEND")
    if override:
        return _normalize_backend(override)

    jax_platform = os.getenv("JAX_PLATFORM_NAME") or os.getenv("JAX_PLATFORMS")
    if jax_platform:
        first = str(jax_platform).split(",")[0].strip()
        if first:
            return _normalize_backend(first)

    if os.path.exists("/dev/accel0") or os.path.exists("/dev/accel"):
        return "tpu"
    for key in ("TPU_NAME", "TPU_WORKER_ID", "TPU_ACCELERATOR_TYPE", "XRT_TPU_CONFIG", "COLAB_TPU_ADDR"):
        if os.getenv(key):
            return "tpu"

    cuda_vis = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_vis and cuda_vis.strip() not in {"", "-1", "none", "void"}:
        return "gpu"
    nvidia_vis = os.getenv("NVIDIA_VISIBLE_DEVICES")
    if nvidia_vis and nvidia_vis.strip() not in {"", "void"}:
        return "gpu"
    if os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
        return "gpu"

    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mps"

    return "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    slug = _default_backend_slug()
    default_quant = Path(f"benchmarks/baselines/quantize_{slug}.json")
    default_dequant = Path(f"benchmarks/baselines/dequantize_{slug}.json")
    parser.add_argument(
        "--baseline-quant",
        type=Path,
        default=default_quant,
        help=f"Quantize baseline JSON path (default: {default_quant}).",
    )
    parser.add_argument(
        "--baseline-dequant",
        type=Path,
        default=default_dequant,
        help=f"Dequantize baseline JSON path (default: {default_dequant}).",
    )
    parser.add_argument("--workdir", type=Path, default=Path("benchmark_outputs"))
    parser.add_argument("--warmup", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--axis", type=str, default="row", choices=["row", "col"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--compute-dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--affine-metadata-dtype",
        type=str,
        default="input",
        choices=["input", "fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--dequant-output-dtype",
        type=str,
        default="fp32",
        choices=["compute", "fp32", "fp16", "bf16"],
    )
    parser.add_argument(
        "--dequant-unpack-policy",
        type=str,
        default="auto",
        choices=["auto", "fast", "generic"],
    )
    parser.add_argument(
        "--minifloat-decode-policy",
        type=str,
        default="auto",
        choices=["auto", "table", "arith"],
    )
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--hard-regression-threshold", type=float, default=0.02)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat the full microbench suite N times and aggregate medians for more stable gating (default: 3).",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=3,
        help="If > 0, aggregate only the last N repeats (default: 3).",
    )
    parser.add_argument("--write-new-baseline", action="store_true")
    return parser.parse_args()


def _run_bench(script_name: str, output_json: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name(script_name)),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seeds",
        args.seeds,
        "--axis",
        args.axis,
        "--dtype",
        args.dtype,
        "--compute-dtype",
        args.compute_dtype,
        "--affine-metadata-dtype",
        args.affine_metadata_dtype,
        "--dequant-output-dtype",
        args.dequant_output_dtype,
        "--dequant-unpack-policy",
        args.dequant_unpack_policy,
        "--minifloat-decode-policy",
        args.minifloat_decode_policy,
        "--output-json",
        str(output_json),
    ]
    if args.max_configs > 0:
        cmd.extend(["--max-configs", str(args.max_configs)])
    print(f"[quant_perf_gate] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list of benchmark rows.")
    return [row for row in payload if isinstance(row, dict)]


def _aggregate_rows(runs: list[list[dict]]) -> list[dict]:
    if not runs:
        raise ValueError("Expected at least one run to aggregate.")

    by_key: dict[str, list[dict]] = {}
    for rows in runs:
        for row in rows:
            key = row.get("benchmark_key")
            if key is None:
                continue
            by_key.setdefault(str(key), []).append(row)

    out_rows: list[dict] = []
    for key, rs in by_key.items():
        base = dict(rs[0])
        base["benchmark_key"] = key
        base["repeats"] = len(rs)

        for field in ("compile_ms", "median_ms", "p95_ms", "p99_ms", "throughput_elems_per_s"):
            vals = [r.get(field) for r in rs if r.get(field) is not None]
            if len(vals) != len(rs):
                base[field] = None
                continue
            base[field] = float(statistics.median(float(v) for v in vals))

        # This is only meaningful when running the microbench scripts directly
        # with a baseline. For aggregated gate artifacts, keep it unset.
        base["relative_delta_vs_baseline"] = None

        out_rows.append(base)

    out_rows.sort(key=lambda r: str(r.get("benchmark_key", "")))
    return out_rows


def _to_median_map(rows: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        key = row.get("benchmark_key")
        median_ms = row.get("median_ms")
        if key is None or median_ms is None:
            continue
        out[str(key)] = float(median_ms)
    return out


def _evaluate_gate(
    *,
    baseline: dict[str, float],
    current: dict[str, float],
    threshold: float,
    label: str,
) -> tuple[bool, list[tuple[str, float, float, float]]]:
    regressions: list[tuple[str, float, float, float]] = []
    missing = sorted(set(current).difference(baseline))
    if missing:
        print(f"[quant_perf_gate] {label}: FAIL (missing {len(missing)} baseline rows; rerun with --write-new-baseline)")
        for key in missing[:20]:
            print(f"  - {key}")
        return False, regressions

    for key, cur_ms in current.items():
        base_ms = baseline[key]
        delta = (cur_ms - base_ms) / base_ms
        if delta > threshold:
            regressions.append((key, base_ms, cur_ms, delta))
    regressions.sort(key=lambda x: x[3], reverse=True)

    if regressions:
        print(f"[quant_perf_gate] {label}: FAIL ({len(regressions)} regressions > {threshold * 100:.2f}%)")
        for key, base_ms, cur_ms, delta in regressions[:20]:
            print(f"  - {key}: baseline={base_ms:.4f}ms current={cur_ms:.4f}ms delta={delta * 100:.2f}%")
        return False, regressions

    print(f"[quant_perf_gate] {label}: PASS (no regressions > {threshold * 100:.2f}%)")
    return True, regressions


def _maybe_write_baseline(path: Path, rows: list[dict], enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))
    print(f"[quant_perf_gate] wrote baseline: {path}")


def main() -> int:
    args = _parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    repeats = max(1, int(args.repeats))
    window = max(0, int(args.rolling_window))

    quant_runs: list[list[dict]] = []
    dequant_runs: list[list[dict]] = []
    for idx in range(repeats):
        suffix = "" if repeats == 1 else f"_run{idx + 1}"
        quant_out = args.workdir / f"quantize_current{suffix}.json"
        dequant_out = args.workdir / f"dequantize_current{suffix}.json"

        _run_bench("benchmark_quantize.py", quant_out, args)
        _run_bench("benchmark_dequantize.py", dequant_out, args)

        quant_runs.append(_load_rows(quant_out))
        dequant_runs.append(_load_rows(dequant_out))

    if window and repeats > window:
        quant_runs = quant_runs[-window:]
        dequant_runs = dequant_runs[-window:]

    quant_rows = _aggregate_rows(quant_runs)
    dequant_rows = _aggregate_rows(dequant_runs)

    quant_current = _to_median_map(quant_rows)
    dequant_current = _to_median_map(dequant_rows)
    quant_baseline = _to_median_map(_load_rows(args.baseline_quant)) if args.baseline_quant.exists() else {}
    dequant_baseline = _to_median_map(_load_rows(args.baseline_dequant)) if args.baseline_dequant.exists() else {}

    if args.write_new_baseline:
        _maybe_write_baseline(args.baseline_quant, quant_rows, True)
        _maybe_write_baseline(args.baseline_dequant, dequant_rows, True)
        return 0

    if not quant_baseline or not dequant_baseline:
        print("[quant_perf_gate] baseline missing: writing current results as baseline and exiting success.")
        _maybe_write_baseline(args.baseline_quant, quant_rows, True)
        _maybe_write_baseline(args.baseline_dequant, dequant_rows, True)
        return 0

    q_ok, _ = _evaluate_gate(
        baseline=quant_baseline,
        current=quant_current,
        threshold=args.hard_regression_threshold,
        label="quantize",
    )
    dq_ok, _ = _evaluate_gate(
        baseline=dequant_baseline,
        current=dequant_current,
        threshold=args.hard_regression_threshold,
        label="dequantize",
    )

    return 0 if (q_ok and dq_ok) else 2


if __name__ == "__main__":
    raise SystemExit(main())
