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

"""TPU quantized matmul strategy sweep (modes x scenarios x kernel ideas).

This benchmark compares multiple execution ideas:
  - op_pallas_strict: `modules.operations.quantized_matmul` with strict packed TPU
    and internal candidate search for packed block config
  - op_xla_fused: fused XLA quantized matmul
  - op_xla_unfused: dequantize+matmul per-call (`fuse=False`)
  - dense_predecode_once: explicit dequantize-once, then dense bf16 matmul

The parent process runs each case in an isolated worker subprocess to reduce
cross-case cache and memory contamination.
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
Scenario = tuple[str, int, int, int]

DEFAULT_MODE_BITS: tuple[ModeBits, ...] = (
    ("affine", 4),
    ("nf4", 4),
    ("mxfp4", 4),
    ("mxfp8", 8),
    ("nvfp4", 4),
    ("nvfp8", 8),
)

DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    ("decode_m1_k4096_n1024", 1, 4096, 1024),
    ("decode_m4_k4096_n1024", 4, 4096, 1024),
    ("decode_m16_k4096_n1024", 16, 4096, 1024),
    ("prefill_m128_k4096_n1024", 128, 4096, 1024),
)

DEFAULT_STRATEGIES: tuple[str, ...] = (
    "op_pallas_strict",
    "op_xla_fused",
    "op_xla_unfused",
    "dense_predecode_once",
)


@dataclass(frozen=True)
class WorkerArgs:
    scenario: str
    m: int
    k: int
    n: int
    mode: str
    bits: int
    strategy: str
    warmup: int
    iters: int
    seed: int
    pallas_probe_warmup: int
    pallas_probe_iters: int
    pallas_max_candidates: int


def _parse_mode_bits(raw: str | None) -> list[ModeBits]:
    if raw is None or not raw.strip():
        return list(DEFAULT_MODE_BITS)
    out: list[ModeBits] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid mode-bits entry {item!r}; expected mode:bits.")
        mode, bits_s = item.split(":", 1)
        out.append((mode.strip(), int(bits_s.strip())))
    if not out:
        raise ValueError("No valid mode-bits entries found.")
    return out


def _parse_scenarios(raw: str | None) -> list[Scenario]:
    if raw is None or not raw.strip():
        return list(DEFAULT_SCENARIOS)
    out: list[Scenario] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        fields = item.split(":")
        if len(fields) != 4:
            raise ValueError(f"Invalid scenario {item!r}; expected name:m:k:n.")
        name, m_s, k_s, n_s = fields
        out.append((name, int(m_s), int(k_s), int(n_s)))
    if not out:
        raise ValueError("No valid scenarios found.")
    return out


def _parse_strategies(raw: str | None) -> list[str]:
    if raw is None or not raw.strip():
        return list(DEFAULT_STRATEGIES)
    out = [x.strip() for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("No valid strategies found.")
    return out


def _build_worker_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--worker", action="store_true")
    p.add_argument("--scenario", type=str, required=True)
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--mode", type=str, required=True)
    p.add_argument("--bits", type=int, required=True)
    p.add_argument("--strategy", type=str, required=True)
    p.add_argument("--warmup", type=int, required=True)
    p.add_argument("--iters", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--pallas-probe-warmup", type=int, required=True)
    p.add_argument("--pallas-probe-iters", type=int, required=True)
    p.add_argument("--pallas-max-candidates", type=int, required=True)
    return p


def _bench_compiled(
    *,
    compiled_fn,
    args: tuple,
    dev,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    import jax

    base_stats = dev.memory_stats() or {}
    base_bytes = int(base_stats.get("bytes_in_use", 0))
    base_peak = int(base_stats.get("peak_bytes_in_use", base_bytes))

    t_compile0 = time.perf_counter()
    y0 = compiled_fn(*args)
    jax.block_until_ready(y0)
    t_compile1 = time.perf_counter()

    for _ in range(max(0, warmup)):
        y = compiled_fn(*args)
        jax.block_until_ready(y)

    times: list[float] = []
    max_live = base_bytes
    max_peak = base_peak
    for _ in range(max(1, iters)):
        t0 = time.perf_counter()
        y = compiled_fn(*args)
        jax.block_until_ready(y)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        stats = dev.memory_stats() or {}
        max_live = max(max_live, int(stats.get("bytes_in_use", max_live)))
        max_peak = max(max_peak, int(stats.get("peak_bytes_in_use", max_peak)))

    mean_s = sum(times) / len(times)
    min_s = min(times)
    max_s = max(times)
    std_s = (sum((x - mean_s) ** 2 for x in times) / len(times)) ** 0.5
    return {
        "compile_s": t_compile1 - t_compile0,
        "mean_s": mean_s,
        "min_s": min_s,
        "max_s": max_s,
        "std_s": std_s,
        "base_bytes_in_use": base_bytes,
        "base_peak_bytes_in_use": base_peak,
        "max_bytes_in_use": max_live,
        "max_peak_bytes_in_use": max_peak,
        "delta_live_bytes": max_live - base_bytes,
        "delta_peak_bytes": max_peak - base_peak,
    }


def _run_worker(cfg: WorkerArgs) -> int:
    import jax
    import jax.numpy as jnp

    from ejkernel.modules.operations import quantized_matmul
    from ejkernel.modules.operations.quantized_matmul import QuantizedMatmulConfig, _pallas_tpu_candidate_cfgs
    from ejkernel.ops import Invocation
    from ejkernel.quantization import dequantize, prepack_quantized_weights

    if jax.default_backend() != "tpu":
        raise RuntimeError(f"TPU backend required, got {jax.default_backend()!r}.")
    dev = jax.devices("tpu")[0]

    key = jax.random.PRNGKey(cfg.seed)
    kx, kw = jax.random.split(key, 2)
    x = jax.random.normal(kx, (cfg.m, cfg.k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (cfg.n, cfg.k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=cfg.mode, bits=cfg.bits)
    if cfg.mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None
    group_size = cfg.n // int(scales.shape[-1])

    x = jax.device_put(x, dev)
    w_q = jax.device_put(w_q, dev)
    scales = jax.device_put(scales, dev)
    if zeros is not None:
        zeros = jax.device_put(zeros, dev)

    strategy = cfg.strategy
    chosen_cfg: QuantizedMatmulConfig | None = None

    def _make_qmm_fn(extra_cfg: QuantizedMatmulConfig | None = None):
        kwargs = dict(
            transpose=False,
            mode=cfg.mode,
            bits=cfg.bits,
            group_size=group_size,
            axis="row",
        )
        if strategy == "op_pallas_strict":
            kwargs.update(
                platform="pallas",
                fuse=True,
                strict_fuse=True,
                tpu_path="packed",
            )
        elif strategy == "op_xla_fused":
            kwargs.update(
                platform="xla",
                fuse=True,
                strict_fuse=True,
            )
        elif strategy == "op_xla_unfused":
            kwargs.update(
                platform="xla",
                fuse=False,
            )
        else:
            raise ValueError(f"Unknown qmm strategy: {strategy}")
        if extra_cfg is not None:
            kwargs["cfg"] = extra_cfg

        if zeros is None:

            def fn(xi, wi, si):
                return quantized_matmul(xi, wi, si, None, **kwargs)

            return fn, (x, w_q, scales)

        def fn(xi, wi, si, zi):
            return quantized_matmul(xi, wi, si, zi, **kwargs)

        return fn, (x, w_q, scales, zeros)

    if strategy == "dense_predecode_once":
        dense_w = dequantize(
            w_q,
            scales,
            zeros,
            group_size=group_size,
            bits=cfg.bits,
            mode=cfg.mode,
            axis="row",
        ).astype(jnp.bfloat16)
        dense_w = jax.device_put(dense_w, dev)

        def fn(xi):
            return jax.lax.dot_general(
                xi.astype(jnp.bfloat16),
                dense_w,
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )

        compiled = jax.jit(fn)
        bench = _bench_compiled(
            compiled_fn=compiled,
            args=(x,),
            dev=dev,
            warmup=cfg.warmup,
            iters=cfg.iters,
        )
    elif strategy == "op_pallas_strict":
        inv = Invocation(
            op_id="quantized_matmul",
            args=(x, w_q, scales, zeros),
            kwargs={
                "transpose": False,
                "group_size": group_size,
                "bits": cfg.bits,
                "mode": cfg.mode,
                "axis": "row",
                "platform": "pallas",
            },
        )
        candidates = _pallas_tpu_candidate_cfgs(inv)[: max(1, cfg.pallas_max_candidates)]
        if not candidates:
            candidates = [QuantizedMatmulConfig(platform="pallas", backend="tpu", tpu_path="packed")]

        if len(candidates) == 1:
            best_cfg = candidates[0]
        else:
            best_probe: dict | None = None
            best_cfg = None
            probe_errors: list[str] = []
            for cand in candidates:
                try:
                    fn, args = _make_qmm_fn(cand)
                    compiled = jax.jit(fn)
                    probe = _bench_compiled(
                        compiled_fn=compiled,
                        args=args,
                        dev=dev,
                        warmup=cfg.pallas_probe_warmup,
                        iters=cfg.pallas_probe_iters,
                    )
                    if best_probe is None or float(probe["mean_s"]) < float(best_probe["mean_s"]):
                        best_probe = probe
                        best_cfg = cand
                except Exception as exc:
                    probe_errors.append(f"{cand.block_m}x{cand.block_n}x{cand.block_k}: {type(exc).__name__}: {exc}")

            if best_cfg is None:
                raise RuntimeError("All strict packed candidates failed. " + " | ".join(probe_errors[:3]))
        chosen_cfg = best_cfg
        fn, args = _make_qmm_fn(best_cfg)
        compiled = jax.jit(fn)
        bench = _bench_compiled(
            compiled_fn=compiled,
            args=args,
            dev=dev,
            warmup=cfg.warmup,
            iters=cfg.iters,
        )
    else:
        fn, args = _make_qmm_fn(None)
        compiled = jax.jit(fn)
        bench = _bench_compiled(
            compiled_fn=compiled,
            args=args,
            dev=dev,
            warmup=cfg.warmup,
            iters=cfg.iters,
        )

    mean_s = float(bench["mean_s"])
    result = {
        "status": "ok",
        "scenario": cfg.scenario,
        "shape": [cfg.m, cfg.k, cfg.n],
        "mode": cfg.mode,
        "bits": cfg.bits,
        "group_size": group_size,
        "strategy": strategy,
        "compile_s": float(bench["compile_s"]),
        "mean_ms": mean_s * 1e3,
        "min_ms": float(bench["min_s"]) * 1e3,
        "max_ms": float(bench["max_s"]) * 1e3,
        "std_ms": float(bench["std_s"]) * 1e3,
        "tokens_per_s": float(cfg.m) / mean_s if mean_s > 0 else 0.0,
        "effective_tops": (2.0 * float(cfg.m) * float(cfg.k) * float(cfg.n) / mean_s) / 1e12 if mean_s > 0 else 0.0,
        "max_bytes_in_use": int(bench["max_bytes_in_use"]),
        "max_peak_bytes_in_use": int(bench["max_peak_bytes_in_use"]),
        "delta_live_bytes": int(bench["delta_live_bytes"]),
        "delta_peak_bytes": int(bench["delta_peak_bytes"]),
    }
    if chosen_cfg is not None:
        result["chosen_cfg"] = {
            "block_m": int(chosen_cfg.block_m),
            "block_n": int(chosen_cfg.block_n),
            "block_k": int(chosen_cfg.block_k),
        }
    print(json.dumps(result, sort_keys=True))
    return 0


def _bytes_to_gib(v: int) -> float:
    return float(v) / (1024.0**3)


def _run_parent(args: argparse.Namespace) -> int:
    scenarios = _parse_scenarios(args.scenarios)
    mode_bits = _parse_mode_bits(args.mode_bits)
    strategies = _parse_strategies(args.strategies)

    rows: list[dict] = []
    failures: list[dict] = []
    total = len(scenarios) * len(mode_bits) * len(strategies)
    idx = 0

    for s_name, m, k, n in scenarios:
        for mode, bits in mode_bits:
            for strategy in strategies:
                idx += 1
                cmd = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--scenario",
                    s_name,
                    "--m",
                    str(m),
                    "--k",
                    str(k),
                    "--n",
                    str(n),
                    "--mode",
                    mode,
                    "--bits",
                    str(bits),
                    "--strategy",
                    strategy,
                    "--warmup",
                    str(args.warmup),
                    "--iters",
                    str(args.iters),
                    "--seed",
                    str(args.seed),
                    "--pallas-probe-warmup",
                    str(args.pallas_probe_warmup),
                    "--pallas-probe-iters",
                    str(args.pallas_probe_iters),
                    "--pallas-max-candidates",
                    str(args.pallas_max_candidates),
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ), check=False)
                if proc.returncode != 0:
                    failures.append(
                        {
                            "scenario": s_name,
                            "mode": mode,
                            "bits": bits,
                            "strategy": strategy,
                            "returncode": proc.returncode,
                            "stderr": proc.stderr[-1200:],
                        }
                    )
                    print(f"[{idx}/{total}] FAIL {s_name} {mode}:{bits} {strategy}")
                    continue
                lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
                if not lines:
                    failures.append(
                        {
                            "scenario": s_name,
                            "mode": mode,
                            "bits": bits,
                            "strategy": strategy,
                            "returncode": 1,
                            "stderr": "empty worker output",
                        }
                    )
                    print(f"[{idx}/{total}] FAIL {s_name} {mode}:{bits} {strategy} (empty output)")
                    continue
                row = json.loads(lines[-1])
                rows.append(row)
                print(f"[{idx}/{total}] OK   {s_name} {mode}:{bits} {strategy} -> {row['mean_ms']:.3f} ms")

    by_case: dict[tuple[str, str, int], list[dict]] = {}
    for r in rows:
        by_case.setdefault((str(r["scenario"]), str(r["mode"]), int(r["bits"])), []).append(r)

    print("\n=== Winners (fastest mean_ms per scenario/mode) ===")
    print("scenario\tmode\tbits\tbest_strategy\tmean_ms\ttokens/s\tTOPS\tlive_GiB")
    for key in sorted(by_case.keys()):
        bucket = sorted(by_case[key], key=lambda x: float(x["mean_ms"]))
        best = bucket[0]
        print(
            "\t".join(
                [
                    key[0],
                    key[1],
                    str(key[2]),
                    str(best["strategy"]),
                    f"{float(best['mean_ms']):.3f}",
                    f"{float(best['tokens_per_s']):.1f}",
                    f"{float(best['effective_tops']):.3f}",
                    f"{_bytes_to_gib(int(best['max_bytes_in_use'])):.3f}",
                ]
            )
        )

    # Aggregate average rank-ish summary by strategy.
    strategy_stats: dict[str, list[float]] = {}
    for _, bucket in by_case.items():
        ordered = sorted(bucket, key=lambda x: float(x["mean_ms"]))
        for rank, row in enumerate(ordered, start=1):
            strategy_stats.setdefault(str(row["strategy"]), []).append(float(rank))
    print("\n=== Strategy Mean Rank (lower is better) ===")
    print("strategy\tmean_rank\tcases")
    for strategy, ranks in sorted(strategy_stats.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        mean_rank = sum(ranks) / len(ranks)
        print(f"{strategy}\t{mean_rank:.2f}\t{len(ranks)}")

    if failures:
        print("\n=== Failures ===")
        for f in failures[:20]:
            print(
                f"{f['scenario']} {f['mode']}:{f['bits']} {f['strategy']} rc={f['returncode']} "
                f"err={str(f['stderr']).replace(chr(10), ' ')[:220]}"
            )
        if len(failures) > 20:
            print(f"... and {len(failures) - 20} more failures")

    if args.output_json:
        payload = {"rows": rows, "failures": failures}
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote JSON results to {args.output_json}")
    return 0 if rows else 1


def main() -> int:
    wp = _build_worker_parser()
    if "--worker" in sys.argv:
        ns = wp.parse_args()
        cfg = WorkerArgs(
            scenario=ns.scenario,
            m=ns.m,
            k=ns.k,
            n=ns.n,
            mode=ns.mode,
            bits=ns.bits,
            strategy=ns.strategy,
            warmup=ns.warmup,
            iters=ns.iters,
            seed=ns.seed,
            pallas_probe_warmup=ns.pallas_probe_warmup,
            pallas_probe_iters=ns.pallas_probe_iters,
            pallas_max_candidates=ns.pallas_max_candidates,
        )
        return _run_worker(cfg)

    p = argparse.ArgumentParser(description="TPU QMM strategy sweep benchmark.")
    p.add_argument("--scenarios", type=str, default=None, help="CSV of name:m:k:n")
    p.add_argument("--mode-bits", type=str, default=None, help="CSV of mode:bits")
    p.add_argument("--strategies", type=str, default=None, help="CSV of strategy names")
    p.add_argument("--warmup", type=int, default=1, help="Warmup iterations per timed benchmark")
    p.add_argument("--iters", type=int, default=6, help="Timed iterations per benchmark")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--pallas-probe-warmup", type=int, default=0, help="Warmup for strict-pallas candidate probing")
    p.add_argument("--pallas-probe-iters", type=int, default=1, help="Timed iters for strict-pallas candidate probing")
    p.add_argument("--pallas-max-candidates", type=int, default=4, help="Max strict-pallas candidates to probe")
    p.add_argument("--output-json", type=str, default=None, help="Optional output JSON filepath")
    args = p.parse_args()
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
