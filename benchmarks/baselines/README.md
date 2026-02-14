# Benchmark Baselines

This directory is the default home for performance baseline JSON artifacts.

Notes:
- `benchmark_outputs/` is gitignored and intended for ephemeral local runs.
- Baselines under `benchmarks/baselines/` are intended to be long-lived and can be committed when appropriate.

## QMM LLM Gate

Default baseline path (used by `benchmarks/qmm_perf_gate.py`):
- `benchmarks/baselines/qmm_llm_gpu_xla_strict.json`

Generate or refresh the baseline:

```bash
.venv/bin/python benchmarks/qmm_perf_gate.py --write-new-baseline
```

## Quant/Dequant Microbench Gate

Default baseline paths (used by `benchmarks/quant_perf_gate.py`):
- `benchmarks/baselines/quantize_gpu.json`
- `benchmarks/baselines/dequantize_gpu.json`

Generate or refresh the baselines:

```bash
.venv/bin/python benchmarks/quant_perf_gate.py --write-new-baseline
```
