#!/usr/bin/env python3
"""Benchmark quantized_matmul CuTe vs CUDA across modes/shapes/dtypes.

Sweeps every combination of quantization mode (affine-4/8, nf4, mxfp4/8,
nvfp4/8), matrix shape, and data type (fp16, bf16), measuring wall-clock
latency for both the CuTe and CUDA platforms.  Reports per-config speedups,
aggregate statistics, and the 5 worst / 5 best cases.

Environment variables:
    EJKERNEL_QMM_SWEEP_WARMUP: Number of warmup iterations (default 6).
    EJKERNEL_QMM_SWEEP_ITERS: Number of timed iterations (default 25).
    EJKERNEL_QMM_SWEEP_SEED: RNG seed for weight generation (default 0).
"""

from __future__ import annotations

import os
import time
from typing import Any

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

MODES: tuple[tuple[str, int], ...] = (
    ("affine", 4),
    ("affine", 8),
    ("nf4", 4),
    ("mxfp4", 4),
    ("mxfp8", 8),
    ("nvfp4", 4),
    ("nvfp8", 8),
)

SHAPES: tuple[tuple[int, int, int], ...] = (
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (256, 1024, 512),
    (512, 1024, 256),
    (128, 4096, 1024),
)

DTYPES: tuple[jnp.dtype, ...] = (jnp.float16, jnp.bfloat16)


def _dtype_name(dtype: jnp.dtype) -> str:
    """Return a short human-readable string for a JAX dtype."""
    return "bf16" if jnp.dtype(dtype) == jnp.bfloat16 else "fp16"


def _bench_case(
    *,
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    mode: str,
    bits: int,
    platform: str,
    warmup: int,
    iters: int,
) -> float:
    """Time a single quantized matmul configuration on a given platform.

    Performs one compilation call, then *warmup* untimed iterations, then
    *iters* timed iterations with ``block_until_ready``.

    Args:
        x: Activation matrix.
        w_q: Quantized weight matrix.
        scales: Per-group scale factors.
        biases: Per-group biases (only for affine mode).
        mode: Quantization mode name.
        bits: Bit width of the quantization.
        platform: Kernel platform to use (``"cute"`` or ``"cuda"``).
        warmup: Number of warmup iterations.
        iters: Number of timed iterations.

    Returns:
        Average latency in milliseconds per iteration.
    """
    y = quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
        platform=platform,
    )
    y.block_until_ready()

    for _ in range(warmup):
        y = quantized_matmul(
            x,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
            platform=platform,
        )
    y.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = quantized_matmul(
            x,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
            platform=platform,
        )
    y.block_until_ready()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / iters


def main() -> int:
    """Run the full CuTe-vs-CUDA quantized matmul sweep and print results.

    Returns:
        0 on success, 1 if the GPU backend is not available.
    """
    if jax.default_backend() != "gpu":
        print("GPU backend required.")
        return 1
    dev = jax.devices("gpu")[0]

    warmup = int(os.getenv("EJKERNEL_QMM_SWEEP_WARMUP", "6"))
    iters = int(os.getenv("EJKERNEL_QMM_SWEEP_ITERS", "25"))
    seed = int(os.getenv("EJKERNEL_QMM_SWEEP_SEED", "0"))

    key = jax.random.PRNGKey(seed)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for dtype in DTYPES:
        for mode, bits in MODES:
            for m, k, n in SHAPES:
                key, kx, kw = jax.random.split(key, 3)
                x = jax.random.normal(kx, (m, k), dtype=dtype)
                w = jax.random.normal(kw, (n, k), dtype=dtype)
                packed = prepack_quantized_weights(w, mode=mode, bits=bits, transpose=True)
                if mode == "affine":
                    w_q, scales, biases = packed
                else:
                    w_q, scales = packed
                    biases = None

                x = jax.device_put(x, dev)
                w_q = jax.device_put(w_q, dev)
                scales = jax.device_put(scales, dev)
                if biases is not None:
                    biases = jax.device_put(biases, dev)

                cute_ms = _bench_case(
                    x=x,
                    w_q=w_q,
                    scales=scales,
                    biases=biases,
                    mode=mode,
                    bits=bits,
                    platform="cute",
                    warmup=warmup,
                    iters=iters,
                )
                cuda_ms = _bench_case(
                    x=x,
                    w_q=w_q,
                    scales=scales,
                    biases=biases,
                    mode=mode,
                    bits=bits,
                    platform="cuda",
                    warmup=warmup,
                    iters=iters,
                )

                speedup = cuda_ms / cute_ms if cute_ms > 0 else float("inf")
                rows.append(
                    {
                        "dtype": _dtype_name(dtype),
                        "mode": mode,
                        "bits": bits,
                        "shape": (m, k, n),
                        "cute": cute_ms,
                        "cuda": cuda_ms,
                        "speedup": speedup,
                    }
                )
                print(
                    f"{_dtype_name(dtype):>4} {mode:>6} b{bits} {m}x{k}x{n}: "
                    f"cute={cute_ms:.3f} ms, cuda={cuda_ms:.3f} ms, x{speedup:.2f}"
                )

    wall = time.perf_counter() - start
    print(f"\nSweep wall-time: {wall:.2f}s for {len(rows)} configs")

    for dtype_name in ("fp16", "bf16"):
        vals = [float(row["speedup"]) for row in rows if row["dtype"] == dtype_name]
        wins = sum(1 for v in vals if v > 1.0)
        mean = sum(vals) / len(vals)
        print(f"{dtype_name}: mean speedup x{mean:.2f}, cute faster in {wins}/{len(vals)}")

    rows.sort(key=lambda item: float(item["speedup"]))
    print("\nWorst 5 (cuda/cute speedup):")
    for row in rows[:5]:
        m, k, n = row["shape"]
        print(
            f"{row['dtype']} {row['mode']} b{row['bits']} {m}x{k}x{n}: "
            f"cute={row['cute']:.3f} cuda={row['cuda']:.3f} x{row['speedup']:.2f}"
        )

    print("\nBest 5 (cuda/cute speedup):")
    for row in rows[-5:]:
        m, k, n = row["shape"]
        print(
            f"{row['dtype']} {row['mode']} b{row['bits']} {m}x{k}x{n}: "
            f"cute={row['cute']:.3f} cuda={row['cuda']:.3f} x{row['speedup']:.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
