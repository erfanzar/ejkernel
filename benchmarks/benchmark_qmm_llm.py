#!/usr/bin/env python3
"""Benchmark quantized_matmul on common LLM weight shapes with percentile reporting.

This is a targeted benchmark for fused QMM coverage/diagnostics on GPU. It
benchmarks explicit platforms (cuda/cute/triton/xla) plus optional "auto"
dispatch, and records:
  - compile_ms
  - median_ms (p50)
  - p95_ms
  - p99_ms
  - gflops_estimate (2*M*N*K / time)
  - path tags (e.g., triton_two_stage, xla_blocked, cuda_custom_call)

Example:
  .venv/bin/python benchmarks/benchmark_qmm_llm.py \\
    --shape-grid llm_all --m-values 1,128 --axis both \\
    --output-json benchmark_outputs/qmm_llm_baseline.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from ejkernel.kernels._registry import Platform
from ejkernel.modules.base import detect_platform
from ejkernel.quantization import QuantRuntimeConfig, prepack_quantized_weights

qmm_op = importlib.import_module("ejkernel.modules.operations.quantized_matmul")

Axis = Literal["row", "col"]
PlatformName = Literal["auto", "cuda", "cute", "triton", "xla", "pallas"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=30)
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument(
        "--strict-fuse",
        action="store_true",
        help="Disallow internal dense dequantize+matmul fallbacks inside fused implementations (e.g. XLA).",
    )
    p.add_argument("--axis", type=str, default="row", choices=["row", "col", "both"])
    p.add_argument(
        "--shape-grid",
        type=str,
        default="llm_all",
        choices=["llm_all", "llm_mlp"],
        help="Preset set of (N,K) weight shapes to benchmark.",
    )
    p.add_argument(
        "--m-values",
        type=str,
        default="1,128",
        help="Comma-separated M values (tokens) to benchmark, e.g. 1,4,16,128.",
    )
    p.add_argument(
        "--platforms",
        type=str,
        default="auto,cuda,cute,triton,xla",
        help="Comma-separated platforms to benchmark: auto,cuda,cute,triton,xla,pallas.",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="affine,nf4",
        help="Comma-separated quantization modes to benchmark (affine,nf4,...).",
    )
    p.add_argument(
        "--affine-bits",
        type=str,
        default="4,8",
        help="Comma-separated affine bit-widths to test (only for mode=affine).",
    )
    p.add_argument("--affine-group-size", type=int, default=128)
    p.add_argument(
        "--nf4-group-sizes",
        type=str,
        default="64,128",
        help="Comma-separated nf4 group sizes to test.",
    )
    p.add_argument("--max-configs", type=int, default=0, help="Optional cap for quick runs (0 means no cap).")
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def _dtype_from_name(name: str) -> jnp.dtype:
    return {"bf16": jnp.bfloat16, "fp16": jnp.float16, "fp32": jnp.float32}[name]


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _parse_str_list(spec: str) -> list[str]:
    return [c.strip() for c in spec.split(",") if c.strip()]


def _shape_grid(grid: str) -> list[tuple[int, int]]:
    if grid == "llm_mlp":
        return [
            (11008, 4096),  # up/gate proj
            (4096, 11008),  # down proj
            (4096, 4096),  # proj
        ]
    if grid == "llm_all":
        return [
            (11008, 4096),
            (4096, 11008),
            (12288, 4096),  # QKV
            (4096, 4096),
        ]
    raise ValueError(f"Unsupported --shape-grid: {grid!r}.")


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return int(b)
    if b <= 0:
        return int(a)
    return abs(a * b) // math.gcd(a, b)


def _blocked_xla_ok(*, transpose: bool, group_size: int, bits: int, block_n: int = 128, block_k: int = 64) -> bool:
    """Fast check matching XLA blocked-kernel ValueError preconditions.

    NOTE: XLA currently auto-adjusts block_k upward for transpose=True to keep
    the blocked path enabled for common (group_size=128) configurations.
    """
    values_per_word = 32 // int(bits)
    if transpose:
        if block_k % values_per_word != 0 or block_k % int(group_size) != 0:
            block_k = _lcm(int(block_k), int(values_per_word))
            block_k = _lcm(int(block_k), int(group_size))
        return (block_k % values_per_word == 0) and (block_k % int(group_size) == 0)
    return (block_n % values_per_word == 0) and (block_n % int(group_size) == 0)


def _triton_two_stage_enabled() -> bool:
    raw = os.getenv("EJKERNEL_QMM_TWO_STAGE", "1").lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _triton_two_stage_path(*, m: int, n: int, k: int) -> bool:
    # Mirrors heuristic in ejkernel/kernels/_triton/quantized_matmul/_triton_impl_fwd.py.
    return _triton_two_stage_enabled() and (m >= 4096 and n >= 4096 and k >= 4096)


def _path_tag(
    *,
    platform: PlatformName,
    effective_platform: str,
    mode: str,
    bits: int,
    group_size: int,
    axis: Axis,
    m: int,
    n: int,
    k: int,
) -> str:
    transpose = axis == "col"
    plat = effective_platform
    if plat != platform:
        prefix = f"{platform}_dispatch_to_{plat}|"
    else:
        prefix = ""

    if plat == "xla":
        ok = _blocked_xla_ok(transpose=transpose, group_size=group_size, bits=bits)
        return prefix + ("xla_blocked" if ok else "xla_dequant_matmul")
    if plat == "triton":
        return prefix + ("triton_two_stage" if _triton_two_stage_path(m=m, n=n, k=k) else "triton_fused")
    if plat == "cuda":
        return prefix + ("cuda_custom_call_transpose" if transpose else "cuda_custom_call")
    if plat == "cute":
        return prefix + "cute_dsl"
    if plat == "pallas":
        return prefix + "pallas"
    return prefix + f"{plat}"


def _key(
    *,
    platform: PlatformName,
    mode: str,
    bits: int,
    group_size: int,
    axis: Axis,
    dtype: str,
    m: int,
    n: int,
    k: int,
    seed: int,
) -> str:
    return f"qmm|plat={platform}|mode={mode}|b{bits}|g{group_size}|axis={axis}|dtype={dtype}|m{m}|n{n}|k{k}|seed{seed}"


def _block_until_ready(x: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


@dataclass(frozen=True)
class _Cfg:
    m: int
    n: int
    k: int
    mode: str
    bits: int
    group_size: int
    axis: Axis
    seed: int


def _build_cfgs(args: argparse.Namespace) -> list[_Cfg]:
    axes: list[Axis]
    if args.axis == "both":
        axes = ["row", "col"]
    else:
        axes = [args.axis]

    m_values = _parse_int_list(args.m_values)
    shapes = _shape_grid(args.shape_grid)
    modes = _parse_str_list(args.modes)
    affine_bits = _parse_int_list(args.affine_bits)
    nf4_group_sizes = _parse_int_list(args.nf4_group_sizes)
    seeds = _parse_int_list(args.seeds)

    cfgs: list[_Cfg] = []
    for axis in axes:
        for n, k in shapes:
            for m in m_values:
                for mode in modes:
                    mode_l = mode.strip().lower()
                    if mode_l == "affine":
                        for bits in affine_bits:
                            group_size = int(args.affine_group_size)
                            # Grouping dim is N for axis=row, and K for axis=col.
                            group_dim = n if axis == "row" else k
                            if group_dim % group_size != 0:
                                continue
                            for seed in seeds:
                                cfgs.append(
                                    _Cfg(
                                        m=m,
                                        n=n,
                                        k=k,
                                        mode=mode_l,
                                        bits=int(bits),
                                        group_size=group_size,
                                        axis=axis,
                                        seed=seed,
                                    )
                                )
                        continue
                    if mode_l == "nf4":
                        for group_size in nf4_group_sizes:
                            group_dim = n if axis == "row" else k
                            if group_dim % int(group_size) != 0:
                                continue
                            for seed in seeds:
                                cfgs.append(
                                    _Cfg(
                                        m=m,
                                        n=n,
                                        k=k,
                                        mode=mode_l,
                                        bits=4,
                                        group_size=int(group_size),
                                        axis=axis,
                                        seed=seed,
                                    )
                                )
                        continue
                    if mode_l in {"mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
                        # Explicit MX/NV modes have fixed (bits, group_size).
                        if mode_l in {"mxfp4", "mxfp8"}:
                            bits = 4 if mode_l == "mxfp4" else 8
                            group_size = 32
                        else:
                            bits = 4 if mode_l == "nvfp4" else 8
                            group_size = 16
                        group_dim = n if axis == "row" else k
                        if group_dim % int(group_size) != 0:
                            continue
                        for seed in seeds:
                            cfgs.append(
                                _Cfg(
                                    m=m,
                                    n=n,
                                    k=k,
                                    mode=mode_l,
                                    bits=int(bits),
                                    group_size=int(group_size),
                                    axis=axis,
                                    seed=seed,
                                )
                            )
                        continue
                    # Other modes can be added later; keep this benchmark focused.
    if args.max_configs and args.max_configs > 0:
        cfgs = cfgs[: int(args.max_configs)]
    return cfgs


def _bench_one(
    *,
    cfg: _Cfg,
    dtype_name: str,
    platform: PlatformName,
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    zeros: jax.Array | None,
    strict_fuse: bool,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    transpose = cfg.axis == "col"
    eff = _effective_platform(platform=platform, axis=cfg.axis)
    tag = _path_tag(
        platform=platform,
        effective_platform=eff,
        mode=cfg.mode,
        bits=cfg.bits,
        group_size=cfg.group_size,
        axis=cfg.axis,
        m=cfg.m,
        n=cfg.n,
        k=cfg.k,
    )

    fn = jax.jit(
        lambda xi, wi, si, zi, plat=platform: qmm_op.quantized_matmul(
            xi,
            wi,
            si,
            zi,
            mode=cfg.mode,
            bits=cfg.bits,
            group_size=cfg.group_size,
            axis=cfg.axis,
            transpose=transpose,
            platform=plat,
            fuse=True,
            strict_fuse=bool(strict_fuse),
        )
    )

    try:
        t0 = time.perf_counter()
        y = fn(x, w_q, scales, zeros)
        _block_until_ready(y)
        compile_ms = (time.perf_counter() - t0) * 1000.0

        for _ in range(warmup):
            y = fn(x, w_q, scales, zeros)
            _block_until_ready(y)

        samples_ms: list[float] = []
        for _ in range(iters):
            t1 = time.perf_counter()
            y = fn(x, w_q, scales, zeros)
            _block_until_ready(y)
            samples_ms.append((time.perf_counter() - t1) * 1000.0)

        arr = np.asarray(samples_ms, dtype=np.float64)
        median_ms = float(np.median(arr))
        p95_ms = float(np.percentile(arr, 95))
        p99_ms = float(np.percentile(arr, 99))
        flops = float(2.0 * cfg.m * cfg.n * cfg.k)
        gflops = float(flops / (median_ms / 1000.0) / 1e9) if median_ms > 0 else 0.0

        return {
            "benchmark_key": _key(
                platform=platform,
                mode=cfg.mode,
                bits=cfg.bits,
                group_size=cfg.group_size,
                axis=cfg.axis,
                dtype=dtype_name,
                m=cfg.m,
                n=cfg.n,
                k=cfg.k,
                seed=cfg.seed,
            ),
            "platform": platform,
            "effective_platform": eff,
            "path_tag": tag,
            "mode": cfg.mode,
            "bits": cfg.bits,
            "group_size": cfg.group_size,
            "axis": cfg.axis,
            "dtype": dtype_name,
            "m": cfg.m,
            "n": cfg.n,
            "k": cfg.k,
            "seed": cfg.seed,
            "compile_ms": compile_ms,
            "median_ms": median_ms,
            "p95_ms": p95_ms,
            "p99_ms": p99_ms,
            "gflops_estimate": gflops,
            "uses_dense_reference": "xla_dequant_matmul" in tag,
            "strict_fuse": bool(strict_fuse),
            "error": None,
        }
    except Exception as e:
        return {
            "benchmark_key": _key(
                platform=platform,
                mode=cfg.mode,
                bits=cfg.bits,
                group_size=cfg.group_size,
                axis=cfg.axis,
                dtype=dtype_name,
                m=cfg.m,
                n=cfg.n,
                k=cfg.k,
                seed=cfg.seed,
            ),
            "platform": platform,
            "effective_platform": eff,
            "path_tag": tag,
            "mode": cfg.mode,
            "bits": cfg.bits,
            "group_size": cfg.group_size,
            "axis": cfg.axis,
            "dtype": dtype_name,
            "m": cfg.m,
            "n": cfg.n,
            "k": cfg.k,
            "seed": cfg.seed,
            "compile_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "gflops_estimate": None,
            "uses_dense_reference": "xla_dequant_matmul" in tag,
            "strict_fuse": bool(strict_fuse),
            "error": f"{type(e).__name__}: {e}",
        }


def _effective_platform(*, platform: PlatformName, axis: Axis) -> str:
    """Best-effort reproduction of quantized_matmul() dispatch for attribution."""
    if platform == "auto":
        resolved = detect_platform(
            "quantized_matmul",
            "auto",
            prefer_pallas=jax.default_backend() == "tpu",
            prefer_cuda=jax.default_backend() in ("gpu", "cuda") and axis != "col",
            prefer_triton=jax.default_backend() in ("gpu", "cuda") and axis == "col",
        )
    else:
        resolved = Platform(platform)

    return resolved.value


def main() -> int:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    runtime_cfg = QuantRuntimeConfig.fastest_for_backend()
    strict_fuse = bool(args.strict_fuse)
    platforms_raw = _parse_str_list(args.platforms)
    platforms: list[PlatformName] = []
    for plat in platforms_raw:
        if plat not in ("auto", "cuda", "cute", "triton", "xla", "pallas"):
            raise ValueError(f"Unsupported platform {plat!r}.")
        platforms.append(plat)  # type: ignore[arg-type]

    cfgs = _build_cfgs(args)
    backend = jax.default_backend()
    device = str(jax.devices()[0]) if jax.devices() else "unknown"
    print(
        f"[benchmark_qmm_llm] backend={backend} device={device} dtype={args.dtype} configs={len(cfgs)}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    x_cache: dict[tuple[int, int, int, str], jax.Array] = {}
    pack_cache: dict[tuple[int, int, str, int, int, str, int, str], tuple[jax.Array, jax.Array, jax.Array | None]] = {}
    for idx, cfg in enumerate(cfgs, start=1):
        # Generate and cache inputs on-device. Many configs share the same
        # packed weights across different M values; caching avoids redundant
        # quantization work during benchmark sweeps.
        x_key = (int(cfg.m), int(cfg.k), int(cfg.seed), args.dtype)
        x = x_cache.get(x_key)
        if x is None:
            rng = jax.random.PRNGKey(int(cfg.seed))
            k1, _ = jax.random.split(rng, 2)
            x = jax.random.normal(k1, (int(cfg.m), int(cfg.k)), dtype=dtype)
            x_cache[x_key] = x

        pack_key = (
            int(cfg.n),
            int(cfg.k),
            str(cfg.mode),
            int(cfg.bits),
            int(cfg.group_size),
            str(cfg.axis),
            int(cfg.seed),
            args.dtype,
        )
        packed = pack_cache.get(pack_key)
        if packed is None:
            rng = jax.random.PRNGKey(int(cfg.seed))
            _, k2 = jax.random.split(rng, 2)
            w = jax.random.normal(k2, (int(cfg.n), int(cfg.k)), dtype=dtype)
            q = prepack_quantized_weights(
                w,
                mode=cfg.mode,
                bits=int(cfg.bits) if cfg.mode == "affine" else None,
                group_size=int(cfg.group_size),
                axis=cfg.axis,
                runtime_config=runtime_cfg,
            )
            if cfg.mode == "affine":
                w_q, scales, zeros = q  # type: ignore[misc]
            else:
                w_q, scales = q  # type: ignore[misc]
                zeros = None
            packed = (w_q, scales, zeros)
            pack_cache[pack_key] = packed

        w_q, scales, zeros = packed
        _block_until_ready((x, w_q, scales, zeros))

        for plat in platforms:
            row = _bench_one(
                cfg=cfg,
                dtype_name=args.dtype,
                platform=plat,
                x=x,
                w_q=w_q,
                scales=scales,
                zeros=zeros,
                strict_fuse=strict_fuse,
                warmup=int(args.warmup),
                iters=int(args.iterations),
            )
            rows.append(row)

        # Print a compact status line with the winning platform for this config.
        ok = [r for r in rows[-len(platforms) :] if r.get("median_ms") is not None]
        winner = None
        if ok:
            winner = min(ok, key=lambda r: float(r["median_ms"]))["platform"]
        win_s = winner if winner is not None else "none"
        print(
            f"[{idx:04d}/{len(cfgs):04d}] m={cfg.m:4d} n={cfg.n:5d} k={cfg.k:5d} {cfg.mode:6s} "
            f"b{cfg.bits} g{cfg.group_size:3d} axis={cfg.axis} winner={win_s}",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "backend": backend,
            "device": device,
            "dtype": args.dtype,
            "runtime_config": {
                "prefer_compute_dtype": runtime_cfg.prefer_compute_dtype,
                "affine_metadata_dtype": runtime_cfg.affine_metadata_dtype,
                "dequant_output_dtype": runtime_cfg.dequant_output_dtype,
                "dequant_unpack_policy": runtime_cfg.dequant_unpack_policy,
                "minifloat_decode_policy": runtime_cfg.minifloat_decode_policy,
            },
            "warmup": int(args.warmup),
            "iterations": int(args.iterations),
            "seeds": args.seeds,
            "shape_grid": args.shape_grid,
            "m_values": args.m_values,
            "axis": args.axis,
            "platforms": ",".join(platforms),
            "strict_fuse": bool(strict_fuse),
            "env": {
                "EJKERNEL_QMM_TWO_STAGE": os.getenv("EJKERNEL_QMM_TWO_STAGE", None),
                "EJKERNEL_QMM_DEQUANT_CACHE": os.getenv("EJKERNEL_QMM_DEQUANT_CACHE", None),
            },
        },
        "rows": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[benchmark_qmm_llm] wrote {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
