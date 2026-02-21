#!/usr/bin/env python3
"""Benchmark quantized_matmul across all available implementations.

Compares quantized matmul kernels against full-precision ``jnp.matmul`` and
optionally reports Pallas/XLA median speed ratios.  The benchmark sweeps over
matrix shapes, quantization modes (affine, nf4, mxfp4/8, nvfp4/8), and
data types.  Results are plotted to ``benchmark_plots/quantized_matmul``.

Environment variables:
    EJKERNEL_QMM_CUDA_CACHE: Enable CUDA compilation caching (default ``"1"``).
    EJKERNEL_QMM_BENCH_COMPARE_CUDA_ONLY: When ``"1"``, only benchmark cuda
        and triton platforms without the fp baseline.
    EJKERNEL_QMM_TPU_ACCEPT_RATIO: If set, the script exits with code 2 when
        the Pallas/XLA median ratio exceeds this threshold.
"""

import os
import sys
from collections import defaultdict

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(__file__))
from _op_benchmark_registry import SPECS, _build_algorithms, _default_dtype

from ejkernel.benchmarks import Benchmark
from ejkernel.quantization import prepack_quantized_weights

os.environ.setdefault("EJKERNEL_QMM_CUDA_CACHE", "1")


def _gen_quantized_inputs_with_fp(config):
    """Generate quantized matmul inputs plus the full-precision weight for comparison.

    Returns:
        Tuple of ``(x, w_q, scales, biases, mode, w_full)`` where ``w_full``
        is the original unquantized weight matrix.
    """
    m = config.get("m", 64)
    k = config.get("k", 64)
    n = config.get("n", 64)
    mode = config.get("mode", "affine")
    dtype = config.get("dtype", _default_dtype())
    if isinstance(dtype, str):
        if dtype == "fp16":
            dtype = jnp.float16
        elif dtype == "bf16":
            dtype = jnp.bfloat16
        elif dtype == "fp32":
            dtype = jnp.float32
        else:
            raise ValueError(f"Unsupported dtype string: {dtype}")
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2 = jax.random.split(key, 2)
    x = jax.random.normal(k1, (m, k), dtype=dtype)
    w = jax.random.normal(k2, (n, k), dtype=dtype)
    packed = prepack_quantized_weights(w, mode=mode)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None
    return x, w_q, scales, biases, mode, w


def _attach_fp_weight(fn):
    """Wrap *fn* to accept and discard the trailing ``w_full`` argument.

    This adapter allows quantized kernels (which do not need the
    full-precision weight) to share the same input signature as the
    full-precision baseline.
    """

    def _fn(x, w_q, scales, biases, mode, w_full):
        return fn(x, w_q, scales, biases, mode)

    return _fn


def _fp_matmul(x, w_q, scales, biases, mode, w_full):
    """Full-precision matmul baseline using the unquantized weight."""
    return jnp.matmul(x, w_full.T)


if __name__ == "__main__":
    spec = SPECS.get("quantized_matmul")
    if spec is None:
        print("No benchmark spec registered for quantized_matmul")
        raise SystemExit(1)

    algorithms = _build_algorithms(spec)
    if not algorithms:
        print(f"No implementations found for {spec.algorithm} on this backend.")
        raise SystemExit(1)

    algorithms = {name: _attach_fp_weight(fn) for name, fn in algorithms.items()}

    if os.getenv("EJKERNEL_QMM_BENCH_COMPARE_CUDA_ONLY", "0") == "1":
        algorithms = {k: v for k, v in algorithms.items() if k in {"cuda", "triton"}}
    else:
        algorithms["matmul_fp"] = _fp_matmul

    bench = Benchmark(
        algorithms=algorithms,
        configs=spec.configs,
        input_generator=_gen_quantized_inputs_with_fp,
        warmup=5,
        iterations=30,
        bench_bwd=spec.bench_bwd,
        static_kwargs=spec.static_kwargs,
        unpack_inputs=True,
    )

    bench.run(verbose=True)
    # Report Pallas/XLA median ratio over common configs.
    per_cfg: dict[tuple[tuple[str, object], ...], dict[str, float]] = defaultdict(dict)
    for result in bench.results:
        cfg_key = tuple(sorted(result.config.items()))
        per_cfg[cfg_key][result.algorithm] = result.median_ms

    ratios = []
    for vals in per_cfg.values():
        if "pallas" in vals and "xla" in vals and vals["xla"] > 0:
            ratios.append(vals["pallas"] / vals["xla"])

    if ratios:
        ratios_sorted = sorted(ratios)
        median_ratio = ratios_sorted[len(ratios_sorted) // 2]
        mean_ratio = sum(ratios_sorted) / len(ratios_sorted)
        print(
            f"[quantized_matmul] Pallas/XLA ratio: "
            f"median={median_ratio:.3f}, mean={mean_ratio:.3f}, samples={len(ratios_sorted)}"
        )
        target_raw = os.getenv("EJKERNEL_QMM_TPU_ACCEPT_RATIO")
        if target_raw:
            target = float(target_raw)
            if median_ratio > target:
                print(f"[quantized_matmul] Acceptance failed: median ratio {median_ratio:.3f} > target {target:.3f}")
                raise SystemExit(2)

    bench.plot(f"benchmark_plots/{spec.op_name}")
    raise SystemExit(0)
