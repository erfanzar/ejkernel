#!/usr/bin/env python3
"""Microbenchmark dequantize() latency/throughput with percentile reporting.

This script reports:
- compile_ms
- median_ms
- p95_ms
- p99_ms
- throughput_elems_per_s
- relative_delta_vs_baseline (optional)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ejkernel.quantization import QuantRuntimeConfig, dequantize, quantize


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--axis", type=str, default="row", choices=["row", "col"])
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument(
        "--shape-grid",
        type=str,
        default="default",
        choices=["default", "llm_mlp", "llm_all"],
        help="Preset set of (N,K) weight shapes to benchmark.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="",
        help="Optional override for shapes: comma-separated 'NxK' pairs, e.g. 11008x4096,4096x11008.",
    )
    parser.add_argument(
        "--compute-dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Runtime dequant compute dtype policy.",
    )
    parser.add_argument(
        "--affine-metadata-dtype",
        type=str,
        default="input",
        choices=["input", "fp32", "fp16", "bf16"],
        help="Storage dtype for affine scales/zeros metadata during quantize setup.",
    )
    parser.add_argument(
        "--dequant-output-dtype",
        type=str,
        default="fp32",
        choices=["compute", "fp32", "fp16", "bf16"],
        help="Output dtype for dequantized tensors.",
    )
    parser.add_argument(
        "--dequant-unpack-policy",
        type=str,
        default="auto",
        choices=["auto", "fast", "generic"],
        help="Unpack strategy policy used inside dequantize.",
    )
    parser.add_argument(
        "--minifloat-decode-policy",
        type=str,
        default="auto",
        choices=["auto", "table", "arith"],
        help="FP4/FP8 codebook decode policy (table lookup vs arithmetic decode).",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--max-configs",
        type=int,
        default=0,
        help="Optional cap for local/dev quick runs (0 means no cap).",
    )
    return parser.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    return {"fp32": jnp.float32, "fp16": jnp.float16, "bf16": jnp.bfloat16}[name]


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _mode_grid() -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    for bits in (1, 2, 4, 8):
        for group_size in (32, 64, 128):
            rows.append(("affine", bits, group_size))
    for group_size in (32, 64, 128):
        rows.append(("nf4", 4, group_size))
    rows.extend(
        [
            ("mxfp4", 4, 32),
            ("mxfp8", 8, 32),
            ("nvfp4", 4, 16),
            ("nvfp8", 8, 16),
        ]
    )
    return rows


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for chunk in spec.split(","):
        item = chunk.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"Invalid --shapes item {chunk!r}; expected 'NxK' like '11008x4096'.")
        n_s, k_s = item.split("x", 1)
        n = int(n_s.strip())
        k = int(k_s.strip())
        if n <= 0 or k <= 0:
            raise ValueError(f"Invalid --shapes item {chunk!r}; N and K must be positive.")
        out.append((n, k))
    if not out:
        raise ValueError("--shapes was provided but no valid shapes were parsed.")
    return out


def _shape_grid(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.shapes.strip():
        return _parse_shapes(args.shapes)

    if args.shape_grid == "default":
        # Weight shapes (N, K) pulled from quantized-matmul relevant cells.
        return [
            (128, 4096),
            (512, 4096),
            (1024, 4096),
            (4096, 4096),
            (4096, 8192),
        ]
    if args.shape_grid == "llm_mlp":
        # LLaMA-7B-ish MLP + proj weights (out_features, in_features).
        return [
            (11008, 4096),  # up/gate proj
            (4096, 11008),  # down proj
            (4096, 4096),  # attn out / generic proj
        ]
    if args.shape_grid == "llm_all":
        return [
            (11008, 4096),
            (4096, 11008),
            (12288, 4096),  # QKV proj
            (4096, 4096),
        ]
    raise ValueError(f"Unsupported --shape-grid: {args.shape_grid!r}.")


def _key(mode: str, bits: int, group_size: int, n: int, k: int, seed: int, axis: str) -> str:
    return f"{mode}|b{bits}|g{group_size}|n{n}|k{k}|seed{seed}|axis={axis}"


def _load_baseline(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out: dict[str, float] = {}
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and "benchmark_key" in row and "median_ms" in row:
                out[str(row["benchmark_key"])] = float(row["median_ms"])
    return out


def _maybe_trim(
    configs: list[tuple[str, int, int, int, int, int]],
    cap: int,
) -> list[tuple[str, int, int, int, int, int]]:
    if cap <= 0:
        return configs
    return configs[:cap]


def main() -> int:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    runtime_cfg = QuantRuntimeConfig(
        prefer_compute_dtype=args.compute_dtype,
        affine_metadata_dtype=args.affine_metadata_dtype,
        dequant_output_dtype=args.dequant_output_dtype,
        dequant_unpack_policy=args.dequant_unpack_policy,
        minifloat_decode_policy=args.minifloat_decode_policy,
    )
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    baseline = _load_baseline(args.baseline)

    matrix: list[tuple[str, int, int, int, int, int]] = []
    for mode, bits, group_size in _mode_grid():
        for n, k in _shape_grid(args):
            for seed in seeds:
                matrix.append((mode, bits, group_size, n, k, seed))
    matrix = _maybe_trim(matrix, args.max_configs)

    rows: list[dict[str, Any]] = []
    backend = jax.default_backend()
    device = str(jax.devices()[0]) if jax.devices() else "unknown"
    print(f"[benchmark_dequantize] backend={backend} device={device} configs={len(matrix)}")

    for idx, (mode, bits, group_size, n, k, seed) in enumerate(matrix, start=1):
        key = _key(mode, bits, group_size, n, k, seed, args.axis)
        rng = jax.random.PRNGKey(seed)
        w = jax.random.normal(rng, (n, k), dtype=dtype)
        qout = quantize(
            w,
            mode=mode,
            bits=bits,
            group_size=group_size,
            axis=args.axis,
            runtime_config=runtime_cfg,
        )
        _block_tree(qout)

        if mode == "affine":
            w_q, scales, zeros = qout
        else:
            w_q, scales = qout
            zeros = None

        fn = jax.jit(
            lambda q, s, z, m=mode, b=bits, g=group_size: dequantize(
                q,
                s,
                z,
                mode=m,
                bits=b,
                group_size=g,
                axis=args.axis,
                runtime_config=runtime_cfg,
            )
        )

        t0 = time.perf_counter()
        out = fn(w_q, scales, zeros)
        _block_tree(out)
        compile_ms = (time.perf_counter() - t0) * 1000.0

        for _ in range(args.warmup):
            out = fn(w_q, scales, zeros)
            _block_tree(out)

        samples_ms: list[float] = []
        for _ in range(args.iterations):
            t1 = time.perf_counter()
            out = fn(w_q, scales, zeros)
            _block_tree(out)
            samples_ms.append((time.perf_counter() - t1) * 1000.0)

        arr = np.asarray(samples_ms, dtype=np.float64)
        median_ms = float(np.median(arr))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        throughput = float((n * k) / (median_ms / 1000.0))

        baseline_ms = baseline.get(key)
        rel_delta = None if baseline_ms is None else float((median_ms - baseline_ms) / baseline_ms)
        rows.append(
            {
                "benchmark_key": key,
                "mode": mode,
                "bits": bits,
                "group_size": group_size,
                "n": n,
                "k": k,
                "seed": seed,
                "axis": args.axis,
                "compile_ms": compile_ms,
                "median_ms": median_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "throughput_elems_per_s": throughput,
                "relative_delta_vs_baseline": rel_delta,
            }
        )
        print(
            f"[{idx:04d}/{len(matrix):04d}] "
            f"{mode:6s} b{bits} g{group_size:3d} n={n:4d} k={k:4d} "
            f"median={median_ms:8.3f}ms p95={p95_ms:8.3f}ms p99={p99_ms:8.3f}ms"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2))
        print(f"[benchmark_dequantize] wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
