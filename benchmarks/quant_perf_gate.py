#!/usr/bin/env python3
"""Run quant/dequant microbenchmarks and enforce regression gates.

This is a local/dev perf gate (no CI workflow dependency). It runs
`benchmark_quantize.py` and `benchmark_dequantize.py`, then compares medians
against provided baseline JSON files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-quant",
        type=Path,
        default=Path("benchmarks/baselines/quantize_gpu.json"),
        help="Quantize baseline JSON path (default: benchmarks/baselines/quantize_gpu.json).",
    )
    parser.add_argument(
        "--baseline-dequant",
        type=Path,
        default=Path("benchmarks/baselines/dequantize_gpu.json"),
        help="Dequantize baseline JSON path (default: benchmarks/baselines/dequantize_gpu.json).",
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
    for key, cur_ms in current.items():
        base_ms = baseline.get(key)
        if base_ms is None:
            continue
        delta = (cur_ms - base_ms) / base_ms
        if delta > threshold:
            regressions.append((key, base_ms, cur_ms, delta))
    regressions.sort(key=lambda x: x[3], reverse=True)

    if regressions:
        print(f"[quant_perf_gate] {label}: FAIL ({len(regressions)} regressions > {threshold * 100:.2f}%)")
        for key, base_ms, cur_ms, delta in regressions[:20]:
            print(
                f"  - {key}: baseline={base_ms:.4f}ms current={cur_ms:.4f}ms "
                f"delta={delta * 100:.2f}%"
            )
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

    quant_out = args.workdir / "quantize_current.json"
    dequant_out = args.workdir / "dequantize_current.json"

    _run_bench("benchmark_quantize.py", quant_out, args)
    _run_bench("benchmark_dequantize.py", dequant_out, args)

    quant_rows = _load_rows(quant_out)
    dequant_rows = _load_rows(dequant_out)

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
