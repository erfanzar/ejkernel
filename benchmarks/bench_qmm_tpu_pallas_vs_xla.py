#!/usr/bin/env python3
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

"""Benchmark packed TPU Pallas QMM vs XLA QMM (latency + HBM usage).

Runs each implementation in an isolated subprocess to avoid cross-run memory
pollution, then prints a concise table with:
  - compile time
  - steady-state step latency
  - tokens/s (rows per second, where rows == token dimension M)
  - effective TOPS (2*M*K*N / time)
  - live and peak HBM usage from ``jax.Device.memory_stats()``

Example:
  ./.venv/bin/python bench/bench_qmm_tpu_pallas_vs_xla.py --m 32 --k 4096 --n 4096 --iters 20
"""

from __future__ import annotations

import sys
from pathlib import Path

# Avoid stdlib shadowing from sibling dirs when executing this file directly.
if sys.path and (sys.path[0].endswith("/test") or sys.path[0].endswith("/bench")):
    sys.path.pop(0)
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse  # noqa
import json  # noqa
import os  # noqa
import subprocess  # noqa
import time  # noqa
from dataclasses import dataclass  # noqa

ModeBits = tuple[str, int]
DEFAULT_MODE_BITS: tuple[ModeBits, ...] = (
    ("affine", 4),
    ("nf4", 4),
    ("mxfp4", 4),
    ("mxfp8", 8),
    ("nvfp4", 4),
    ("nvfp8", 8),
)


@dataclass(frozen=True)
class BenchArgs:
    m: int
    k: int
    n: int
    iters: int
    warmup: int
    seed: int
    mode: str
    bits: int
    impl: str


def _parse_mode_bits(raw: str | None) -> list[ModeBits]:
    if raw is None or not raw.strip():
        return list(DEFAULT_MODE_BITS)
    pairs: list[ModeBits] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mode spec {item!r}; expected 'mode:bits'.")
        mode, bits_s = item.split(":", 1)
        pairs.append((mode.strip(), int(bits_s.strip())))
    if not pairs:
        raise ValueError("No valid mode:bits entries were provided.")
    return pairs


def _build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--impl", type=str, required=True, choices=("pallas", "xla"))
    return p


def _run_worker(cfg: BenchArgs) -> int:
    import jax
    import jax.numpy as jnp

    from ejkernel.kernels._pallas.tpu.quantized_matmul import quantized_matmul as pallas_quantized_matmul
    from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
    from ejkernel.quantization import prepack_quantized_weights

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"TPU backend required, got {jax.default_backend()!r}.")
    dev = jax.devices("tpu")[0]

    key = jax.random.PRNGKey(cfg.seed)
    kx, kw = jax.random.split(key, 2)
    x = jax.random.normal(kx, (cfg.m, cfg.k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (cfg.n, cfg.k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=cfg.mode, bits=cfg.bits)
    if cfg.mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None
    group_size = cfg.n // int(scales.shape[-1])

    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    kernel = pallas_quantized_matmul if cfg.impl == "pallas" else xla_quantized_matmul
    common_kwargs = dict(
        transpose=False,
        mode=cfg.mode,
        bits=cfg.bits,
        group_size=group_size,
    )
    if cfg.impl == "pallas":
        # For strict packed mode, ensure block_n is TPU-legal for packed
        # w/scales block specs by covering n_pad (or a legal multiple).
        n_pad = ((cfg.n + 127) // 128) * 128
        m_pad = max(8, ((cfg.m + 7) // 8) * 8)
        common_kwargs["tpu_path"] = "packed"
        common_kwargs["allow_dense_fallback"] = False
        common_kwargs["block_m"] = m_pad
        common_kwargs["block_n"] = n_pad
    if biases is None:

        def fn(xi, wi, si):
            return kernel(
                xi,
                wi,
                si,
                None,
                **common_kwargs,
            )

        compiled = jax.jit(fn)
        args = (x, w_q, scales)
    else:

        def fn(xi, wi, si, bi):
            return kernel(
                xi,
                wi,
                si,
                bi,
                **common_kwargs,
            )

        compiled = jax.jit(fn)
        args = (x, w_q, scales, biases)

    base_stats = dev.memory_stats() or {}
    base_bytes = int(base_stats.get("bytes_in_use", 0))
    base_peak = int(base_stats.get("peak_bytes_in_use", base_bytes))

    t_compile0 = time.perf_counter()
    y0 = compiled(*args)
    jax.block_until_ready(y0)
    t_compile1 = time.perf_counter()

    for _ in range(cfg.warmup):
        y = compiled(*args)
        jax.block_until_ready(y)

    lat_s: list[float] = []
    max_live = base_bytes
    max_peak = base_peak
    for _ in range(cfg.iters):
        t0 = time.perf_counter()
        y = compiled(*args)
        jax.block_until_ready(y)
        t1 = time.perf_counter()
        lat_s.append(t1 - t0)
        stats = dev.memory_stats() or {}
        max_live = max(max_live, int(stats.get("bytes_in_use", max_live)))
        max_peak = max(max_peak, int(stats.get("peak_bytes_in_use", max_peak)))

    mean_s = sum(lat_s) / max(1, len(lat_s))
    min_s = min(lat_s) if lat_s else 0.0
    max_s = max(lat_s) if lat_s else 0.0
    std_s = (sum((x - mean_s) ** 2 for x in lat_s) / max(1, len(lat_s))) ** 0.5 if lat_s else 0.0
    tokens_per_s = float(cfg.m) / mean_s if mean_s > 0 else 0.0
    tops = (2.0 * float(cfg.m) * float(cfg.k) * float(cfg.n) / mean_s) / 1e12 if mean_s > 0 else 0.0

    result = {
        "impl": cfg.impl,
        "mode": cfg.mode,
        "bits": cfg.bits,
        "shape": [cfg.m, cfg.k, cfg.n],
        "group_size": group_size,
        "compile_s": t_compile1 - t_compile0,
        "mean_ms": mean_s * 1e3,
        "min_ms": min_s * 1e3,
        "max_ms": max_s * 1e3,
        "std_ms": std_s * 1e3,
        "tokens_per_s": tokens_per_s,
        "effective_tops": tops,
        "base_bytes_in_use": base_bytes,
        "base_peak_bytes_in_use": base_peak,
        "max_bytes_in_use": max_live,
        "max_peak_bytes_in_use": max_peak,
        "delta_peak_bytes": max_peak - base_peak,
        "delta_live_bytes": max_live - base_bytes,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


def _bytes_to_gib(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0**3)


def _format_summary(rows: list[dict]) -> str:
    if not rows:
        return "No benchmark rows."
    headers = [
        "mode",
        "bits",
        "impl",
        "compile_s",
        "mean_ms",
        "tokens/s",
        "TOPS",
        "live_GiB",
        "delta_peak_GiB",
    ]
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append(
            "\t".join(
                [
                    str(r["mode"]),
                    str(r["bits"]),
                    str(r["impl"]),
                    f"{r['compile_s']:.3f}",
                    f"{r['mean_ms']:.3f}",
                    f"{r['tokens_per_s']:.1f}",
                    f"{r['effective_tops']:.3f}",
                    f"{_bytes_to_gib(int(r['max_bytes_in_use'])):.3f}",
                    f"{_bytes_to_gib(int(r['delta_peak_bytes'])):.3f}",
                ]
            )
        )
    return "\n".join(lines)


def _print_speedups(rows: list[dict]) -> None:
    by_key: dict[tuple[str, int], dict[str, dict]] = {}
    for r in rows:
        by_key.setdefault((str(r["mode"]), int(r["bits"])), {})[str(r["impl"])] = r
    print("\nSpeedup (XLA mean_ms / Pallas mean_ms):")
    for key in sorted(by_key.keys()):
        bucket = by_key[key]
        if "xla" not in bucket or "pallas" not in bucket:
            continue
        xla_ms = float(bucket["xla"]["mean_ms"])
        pallas_ms = float(bucket["pallas"]["mean_ms"])
        speedup = xla_ms / pallas_ms if pallas_ms > 0 else 0.0
        print(f"  {key[0]}:{key[1]} -> {speedup:.3f}x")


def _run_parent(args: argparse.Namespace) -> int:
    mode_bits = _parse_mode_bits(args.mode_bits)
    rows: list[dict] = []
    for mode, bits in mode_bits:
        for impl in ("pallas", "xla"):
            cmd = [
                sys.executable,
                __file__,
                "--worker",
                "--m",
                str(args.m),
                "--k",
                str(args.k),
                "--n",
                str(args.n),
                "--iters",
                str(args.iters),
                "--warmup",
                str(args.warmup),
                "--seed",
                str(args.seed),
                "--mode",
                mode,
                "--bits",
                str(bits),
                "--impl",
                impl,
            ]
            env = dict(os.environ)
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
            if proc.returncode != 0:
                sys.stderr.write(f"[{mode}:{bits} {impl}] worker failed:\n")
                sys.stderr.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                return proc.returncode
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            if not lines:
                sys.stderr.write(f"[{mode}:{bits} {impl}] empty worker output.\n")
                return 1
            rows.append(json.loads(lines[-1]))

    print(_format_summary(rows))
    _print_speedups(rows)
    return 0


def main() -> int:
    worker_parser = _build_worker_parser()
    if "--worker" in sys.argv:
        ns = worker_parser.parse_args()
        cfg = BenchArgs(
            m=ns.m,
            k=ns.k,
            n=ns.n,
            iters=ns.iters,
            warmup=ns.warmup,
            seed=ns.seed,
            mode=ns.mode,
            bits=ns.bits,
            impl=ns.impl,
        )
        return _run_worker(cfg)

    p = argparse.ArgumentParser(description="Benchmark TPU Pallas QMM vs XLA QMM.")
    p.add_argument("--m", type=int, default=32, help="Rows (token dimension).")
    p.add_argument("--k", type=int, default=4096, help="Contraction dimension.")
    p.add_argument("--n", type=int, default=4096, help="Output dimension.")
    p.add_argument("--iters", type=int, default=20, help="Timed iterations.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations after compile.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed.")
    p.add_argument(
        "--mode-bits",
        type=str,
        default=None,
        help="Comma-separated list like 'affine:4,nf4:4,mxfp8:8'.",
    )
    args = p.parse_args()
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
