#!/usr/bin/env python3
"""Run fused QMM LLM benchmark and enforce regression gates.

This is a local/dev perf gate (no GitHub workflows). It runs
`benchmark_qmm_llm.py` and compares p50 medians against a baseline JSON file.

Hard gates:
  - any protected-cell median_ms regression > threshold (default 2%) fails
  - any protected-cell error fails
  - any protected-cell `uses_dense_reference=True` fails (i.e., dequantize+matmul fallback)
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

    # TPU probes.
    if os.path.exists("/dev/accel0") or os.path.exists("/dev/accel"):
        return "tpu"
    for key in ("TPU_NAME", "TPU_WORKER_ID", "TPU_ACCELERATOR_TYPE", "XRT_TPU_CONFIG", "COLAB_TPU_ADDR"):
        if os.getenv(key):
            return "tpu"

    # GPU probes (CUDA/ROCm containers often set these).
    cuda_vis = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_vis and cuda_vis.strip() not in {"", "-1", "none", "void"}:
        return "gpu"
    nvidia_vis = os.getenv("NVIDIA_VISIBLE_DEVICES")
    if nvidia_vis and nvidia_vis.strip() not in {"", "void"}:
        return "gpu"
    if os.getenv("ROCR_VISIBLE_DEVICES") or os.getenv("HIP_VISIBLE_DEVICES"):
        return "gpu"

    # Apple Silicon probe for dev-only MPS lane.
    if sys.platform == "darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
        return "mps"

    return "cpu"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_baseline = Path(f"benchmarks/baselines/qmm_llm_{_default_backend_slug()}_xla_strict.json")
    p.add_argument(
        "--baseline",
        type=Path,
        default=default_baseline,
        help=f"Baseline JSON path (default: {default_baseline}).",
    )
    p.add_argument("--workdir", type=Path, default=Path("benchmark_outputs"))
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--axis", type=str, default="both", choices=["row", "col", "both"])
    p.add_argument("--shape-grid", type=str, default="llm_all", choices=["llm_all", "llm_mlp"])
    p.add_argument("--m-values", type=str, default="1,128")
    p.add_argument("--platforms", type=str, default="xla")
    p.add_argument("--modes", type=str, default="affine,nf4,mxfp4,mxfp8,nvfp4,nvfp8")
    p.add_argument("--affine-bits", type=str, default="4,8")
    p.add_argument("--affine-group-size", type=int, default=128)
    p.add_argument("--nf4-group-sizes", type=str, default="64,128")
    p.add_argument("--max-configs", type=int, default=0)
    p.add_argument(
        "--gate-platforms",
        type=str,
        default="",
        help="Comma-separated platform names to gate. Empty means all platforms in the baseline.",
    )
    p.add_argument("--hard-regression-threshold", type=float, default=0.02)
    p.add_argument(
        "--strict-fuse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run benchmark with strict_fuse enabled (disallow XLA dense fallback).",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=4,
        help=(
            "Repeat the full benchmark N times and aggregate medians for more stable gating "
            "(default: 4; see --rolling-window)."
        ),
    )
    p.add_argument(
        "--rolling-window",
        type=int,
        default=3,
        help=(
            "If > 0, aggregate only the last N repeats (rolling median window). "
            "This is useful when the first run is a cold-start outlier (default: 3)."
        ),
    )
    p.add_argument("--write-new-baseline", action="store_true")
    p.add_argument(
        "--allow-metadata-mismatch",
        action="store_true",
        help="Allow gating against a baseline with different run settings (dtype/shape grid/platforms/etc).",
    )
    return p.parse_args()


def _run_bench(output_json: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("benchmark_qmm_llm.py")),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--seeds",
        args.seeds,
        "--dtype",
        args.dtype,
        "--axis",
        args.axis,
        "--shape-grid",
        args.shape_grid,
        "--m-values",
        args.m_values,
        "--platforms",
        args.platforms,
        "--modes",
        args.modes,
        "--affine-bits",
        args.affine_bits,
        "--affine-group-size",
        str(args.affine_group_size),
        "--nf4-group-sizes",
        args.nf4_group_sizes,
        "--output-json",
        str(output_json),
    ]
    if args.strict_fuse:
        cmd.append("--strict-fuse")
    if args.max_configs > 0:
        cmd.extend(["--max-configs", str(args.max_configs)])
    print(f"[qmm_perf_gate] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _load_payload(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError(f"{path} must contain a dict with a 'rows' field.")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{path}['rows'] must be a list.")
    return payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}, [
        r for r in rows if isinstance(r, dict)
    ]


def _rows_by_key(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        key = row.get("benchmark_key")
        if key is None:
            continue
        out[str(key)] = row
    return out


def _parse_list(spec: str) -> set[str]:
    return {c.strip() for c in spec.split(",") if c.strip()}


def _aggregate_payloads(payloads: list[tuple[dict, list[dict]]]) -> tuple[dict, list[dict]]:
    """Aggregate multiple benchmark runs into a single (metadata, rows) payload."""
    if not payloads:
        raise ValueError("Expected at least one payload to aggregate.")

    metas, rows_list = zip(*payloads, strict=True)
    meta = dict(metas[0])
    meta["repeats"] = len(payloads)

    by_key: dict[str, list[dict]] = {}
    for rows in rows_list:
        for row in rows:
            key = row.get("benchmark_key")
            if key is None:
                continue
            by_key.setdefault(str(key), []).append(row)

    out_rows: list[dict] = []
    for key, runs in by_key.items():
        base = dict(runs[0])
        base["benchmark_key"] = key

        errors = [r.get("error") for r in runs if r.get("error") is not None]
        if errors:
            base["error"] = str(errors[0])
            out_rows.append(base)
            continue

        # If any run used dense fallback, keep that fact.
        base["uses_dense_reference"] = any(bool(r.get("uses_dense_reference")) for r in runs)

        for field in ("compile_ms", "median_ms", "p95_ms", "p99_ms", "gflops_estimate"):
            vals = [r.get(field) for r in runs if r.get(field) is not None]
            if len(vals) != len(runs):
                base[field] = None
                continue
            base[field] = float(statistics.median(float(v) for v in vals))

        out_rows.append(base)

    out_rows.sort(key=lambda r: str(r.get("benchmark_key", "")))
    return meta, out_rows


def _maybe_write_baseline(path: Path, payload: dict, enabled: bool) -> None:
    if not enabled:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[qmm_perf_gate] wrote baseline: {path}", flush=True)


def main() -> int:
    args = _parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    payloads: list[tuple[dict, list[dict]]] = []
    repeats = max(1, int(args.repeats))
    for idx in range(repeats):
        suffix = "" if repeats == 1 else f"_run{idx + 1}"
        cur_out = args.workdir / f"qmm_llm_current{suffix}.json"
        _run_bench(cur_out, args)
        payloads.append(_load_payload(cur_out))

    window = max(0, int(args.rolling_window))
    if window and len(payloads) > window:
        payloads = payloads[-window:]

    cur_meta, cur_rows = _aggregate_payloads(payloads)
    cur_meta["repeats_total"] = repeats
    cur_meta["rolling_window"] = window
    cur_by_key = _rows_by_key(cur_rows)

    if args.write_new_baseline:
        payload = {"metadata": cur_meta, "rows": cur_rows}
        _maybe_write_baseline(args.baseline, payload, True)
        return 0

    if not args.baseline.exists():
        print("[qmm_perf_gate] baseline missing: writing current results as baseline and exiting success.", flush=True)
        payload = {"metadata": cur_meta, "rows": cur_rows}
        _maybe_write_baseline(args.baseline, payload, True)
        return 0

    base_meta, base_rows = _load_payload(args.baseline)
    base_by_key = _rows_by_key(base_rows)

    gate_platforms = _parse_list(args.gate_platforms)
    regressions: list[tuple[str, float, float, float]] = []
    errors: list[tuple[str, str]] = []
    dense_fallbacks: list[str] = []
    missing: list[str] = []

    # Gating is only meaningful when the baseline/current runs match. Require
    # key run-protocol fields unless explicitly overridden.
    if base_meta and cur_meta and not args.allow_metadata_mismatch:
        mismatches = []
        for field in (
            "backend",
            "dtype",
            "shape_grid",
            "m_values",
            "axis",
            "platforms",
            "strict_fuse",
            "warmup",
            "iterations",
            "seeds",
            "repeats",
            "repeats_total",
            "rolling_window",
        ):
            if base_meta.get(field) != cur_meta.get(field):
                mismatches.append(field)
        if mismatches:
            print(
                "[qmm_perf_gate] FAIL: baseline/current metadata differ in: "
                + ", ".join(mismatches)
                + " (rerun with --write-new-baseline, or pass --allow-metadata-mismatch)",
                flush=True,
            )
            return 2

    for key, base_row in base_by_key.items():
        plat = str(base_row.get("platform", ""))
        if gate_platforms and plat not in gate_platforms:
            continue
        base_ms = base_row.get("median_ms")
        if base_ms is None:
            continue
        base_ms_f = float(base_ms)

        cur_row = cur_by_key.get(key)
        if cur_row is None:
            missing.append(key)
            continue
        if cur_row.get("error") is not None:
            errors.append((key, str(cur_row.get("error"))))
            continue
        if cur_row.get("uses_dense_reference"):
            dense_fallbacks.append(key)
            continue
        cur_ms = cur_row.get("median_ms")
        if cur_ms is None:
            errors.append((key, "missing median_ms"))
            continue
        cur_ms_f = float(cur_ms)
        delta = (cur_ms_f - base_ms_f) / base_ms_f if base_ms_f > 0 else 0.0
        if delta > float(args.hard_regression_threshold):
            regressions.append((key, base_ms_f, cur_ms_f, delta))

    ok = True
    if missing:
        ok = False
        print(f"[qmm_perf_gate] FAIL: missing {len(missing)} protected rows (showing up to 20)", flush=True)
        for key in missing[:20]:
            print(f"  - {key}", flush=True)
    if errors:
        ok = False
        print(f"[qmm_perf_gate] FAIL: {len(errors)} protected rows errored (showing up to 20)", flush=True)
        for key, err in errors[:20]:
            print(f"  - {key}: {err}", flush=True)
    if dense_fallbacks:
        ok = False
        print(
            f"[qmm_perf_gate] FAIL: {len(dense_fallbacks)} protected rows used dense fallback (showing up to 20)",
            flush=True,
        )
        for key in dense_fallbacks[:20]:
            print(f"  - {key}", flush=True)

    regressions.sort(key=lambda x: x[3], reverse=True)
    if regressions:
        ok = False
        thr = float(args.hard_regression_threshold) * 100.0
        print(f"[qmm_perf_gate] FAIL: {len(regressions)} regressions > {thr:.2f}% (showing up to 20)", flush=True)
        for key, base_ms, cur_ms, delta in regressions[:20]:
            print(
                f"  - {key}: baseline={base_ms:.4f}ms current={cur_ms:.4f}ms delta={delta * 100:.2f}%",
                flush=True,
            )

    if ok:
        print("[qmm_perf_gate] PASS", flush=True)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
