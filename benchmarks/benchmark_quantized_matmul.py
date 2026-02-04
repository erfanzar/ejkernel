#!/usr/bin/env python3
"""Benchmark quantized_matmul across all available implementations."""

import os
import sys

import jax
import jax.numpy as jnp

sys.path.append(os.path.dirname(__file__))
from _op_benchmark_registry import SPECS, _build_algorithms, _default_dtype, _ignored_platforms

from ejkernel.benchmarks import Benchmark
from ejkernel.quantization import prepack_quantized_weights

os.environ.setdefault("EJKERNEL_QMM_CUDA_CACHE", "1")


def _gen_quantized_inputs_with_fp(config):
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
    def _fn(x, w_q, scales, biases, mode, w_full):
        return fn(x, w_q, scales, biases, mode)

    return _fn


def _fp_matmul(x, w_q, scales, biases, mode, w_full):
    return jnp.matmul(x, w_full.T)


if __name__ == "__main__":
    spec = SPECS.get("quantized_matmul")
    if spec is None:
        print("No benchmark spec registered for quantized_matmul")
        raise SystemExit(1)

    algorithms = _build_algorithms(spec, ignore_platforms=_ignored_platforms(["triton"]))
    if not algorithms:
        print(f"No implementations found for {spec.algorithm} on this backend.")
        raise SystemExit(1)

    algorithms = {name: _attach_fp_weight(fn) for name, fn in algorithms.items()}
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
    bench.plot(f"benchmark_plots/{spec.op_name}")
    raise SystemExit(0)
