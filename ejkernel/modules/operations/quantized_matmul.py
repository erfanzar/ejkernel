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

"""Quantized matrix multiplication operation with automatic optimization.

This module implements fused quantized matrix multiplication, performing
dequantization and matmul in a single kernel pass for maximum throughput.
It supports multiple quantization formats and automatically selects the
optimal backend (Triton, Pallas, CUDA, CuTe, or XLA) based on the target
hardware and input characteristics.

Supported Quantization Modes:
    - affine: Standard asymmetric quantization with scales and zero-points
    - nf4: NormalFloat4 (QLoRA-style 4-bit quantization)
    - mxfp4: Microscaling FP4
    - mxfp8: Microscaling FP8
    - nvfp4: NVIDIA FP4
    - nvfp8: NVIDIA FP8

Key Features:
    - Fused dequantize + matmul in a single kernel pass
    - Automatic platform selection (Triton/Pallas/CUDA/CuTe/XLA)
    - Configurable block sizes with autotuning support
    - Split-K for improved parallelism on tall-skinny matrices
    - GEMV specialization for M=1 workloads
    - Custom VJP for backward pass compatibility
    - TPU Pallas support with packed path optimization

Mathematical Formulation:
    output = x @ dequantize(w, scales, zeros)
    output = x @ dequantize(w, scales, zeros).T  (when transpose=True)

Performance Characteristics:
    - Eliminates memory bandwidth overhead of separate dequantization
    - Automatic split-K selection for small batch sizes
    - Hardware-specific heuristics for block size selection
    - Persistent configuration caching across runs

References:
    - QLoRA: https://arxiv.org/abs/2305.14314
    - GPTQ: https://arxiv.org/abs/2210.17323
    - Microscaling: https://arxiv.org/abs/2310.10537
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ejkernel.kernels._registry import Platform, kernel_registry
from ejkernel.ops import (
    AutotunePolicy,
    ConfigCache,
    ConfigSelectorChain,
    Executor,
    Invocation,
    Kernel,
    Tuner,
    policy_override,
)
from ejkernel.ops.config.persistent import PersistentCache
from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    QuantizationMode,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    select_qmm_kernel_family,
    validate_packed_quantized_matmul_layout,
)

from ..base import detect_platform
from .configs import QuantizedMatmulConfig


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve quantization parameters from mode, group_size, and bits.

    Args:
        mode: Quantization mode string (e.g., "affine", "nf4").
        group_size: Optional group size override.
        bits: Optional bit-width override.

    Returns:
        Tuple of (group_size, bits) with defaults applied based on mode.
    """
    _, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return group_size, bits


def _static_bool(value, name: str) -> bool:
    """Extract a concrete boolean value, raising if it is a JAX tracer.

    Args:
        value: The value to concretize.
        name: Parameter name for the error message.

    Returns:
        The concrete boolean value.
    """
    return jax.core.concrete_or_error(bool, value, f"{name} must be static.")


def _static_int(value, name: str) -> int:
    """Extract a concrete integer value, raising if it is a JAX tracer.

    Args:
        value: The value to concretize.
        name: Parameter name for the error message.

    Returns:
        The concrete integer value.
    """
    return jax.core.concrete_or_error(int, value, f"{name} must be static.")


def _lcm(a: int, b: int) -> int:
    """Compute the least common multiple of two integers.

    Args:
        a: First integer. If <= 0, returns b.
        b: Second integer. If <= 0, returns a.

    Returns:
        The least common multiple of a and b.
    """
    if a <= 0:
        return int(b)
    if b <= 0:
        return int(a)
    return abs(a * b) // math.gcd(a, b)


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division of a by b.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        The smallest integer >= a/b.
    """
    return (a + b - 1) // b


def _nearest_choices(value: int, choices: tuple[int, ...], count: int = 2) -> list[int]:
    """Select the nearest choices to a target value from a set of options.

    Args:
        value: Target value to match.
        choices: Available choices to select from.
        count: Number of nearest choices to return.

    Returns:
        Sorted list of the `count` choices closest to `value`.
    """
    ranked = sorted(set(choices), key=lambda x: abs(x - value))
    return sorted(ranked[:count])


def _expand_choices(value: int, choices: tuple[int, ...]) -> list[int]:
    """Expand a value into a neighborhood of choices.

    Returns the value itself plus its immediate neighbors in the sorted
    choices list, providing a small search window for autotuning.

    Args:
        value: The base value to expand around.
        choices: Sorted tuple of available choices.

    Returns:
        Sorted list of up to 3 choices: the value and its neighbors.
    """
    choices = tuple(sorted(set(choices)))
    try:
        idx = choices.index(value)
    except ValueError:
        idx = 0
    out = {choices[idx]}
    if idx > 0:
        out.add(choices[idx - 1])
    if idx + 1 < len(choices):
        out.add(choices[idx + 1])
    return sorted(out)


def _ensure_aligned(choices: list[int], align: int, max_choice: int) -> list[int]:
    """Filter or round choices to ensure alignment.

    Returns only choices that are multiples of `align`. If none of the
    original choices are aligned, rounds up and filters by max_choice.

    Args:
        choices: List of candidate block sizes.
        align: Required alignment (e.g., group_size * values_per_word).
        max_choice: Maximum allowed value after rounding.

    Returns:
        List of aligned choices, or [align] as fallback.
    """
    if align <= 1:
        return choices
    aligned = [c for c in choices if c % align == 0]
    if aligned:
        return aligned
    rounded = [((c + align - 1) // align) * align for c in choices]
    aligned = [c for c in rounded if c <= max_choice]
    if aligned:
        return sorted(set(aligned))
    return [align]


def _inv_arg(inv: Invocation[QuantizedMatmulConfig, Array], name: str, index: int):
    """Resolve a positional-or-keyword argument from an Invocation.

    Args:
        inv: The kernel invocation containing args and kwargs.
        name: The keyword argument name to look up.
        index: The positional argument index to try first.

    Returns:
        The resolved argument value.
    """
    if len(inv.args) > index:
        return inv.args[index]
    return inv.kwargs[name]


def _infer_mkn(inv: Invocation[QuantizedMatmulConfig, Array], group_size: int) -> tuple[int, int, int, bool]:
    """Infer the M, K, N dimensions and transpose flag from an invocation.

    Extracts shape information from the input tensors (x, w, scales)
    to determine the effective matmul dimensions.

    Args:
        inv: The kernel invocation containing the input tensors.
        group_size: Quantization group size, used to compute N when
            transpose is False.

    Returns:
        Tuple of (M, K, N, transpose) where M is the batch dimension,
        K is the reduction dimension, N is the output dimension, and
        transpose indicates whether the weight is in transposed layout.
    """
    x = _inv_arg(inv, "x", 0)
    w = _inv_arg(inv, "w", 1)
    scales = _inv_arg(inv, "scales", 2)
    transpose = _static_bool(inv.kwargs.get("transpose", False), "transpose")
    M, K = x.shape
    if transpose:
        N = w.shape[0]
    else:
        N = scales.shape[1] * group_size
    return int(M), int(K), int(N), transpose


def _prefer_bf16(x: Array) -> bool:
    """Determine whether to prefer bfloat16 accumulation for the given input.

    Returns True only when the activation dtype is already bfloat16.

    On GPU, float16 is the typical fast-path compute type for quantized matmul
    and matches the default quant/dequant runtime config. Using bfloat16
    for float32 activations can introduce extra rounding error vs the reference
    dequantize+matmul path.

    Args:
        x: Input array to check dtype of.

    Returns:
        True if bfloat16 is preferred, False otherwise.
    """
    dt = getattr(x, "dtype", None)
    if dt is None:
        return True
    return dt == jnp.bfloat16


def _pick_split_k(m: int, k: int, block_k: int) -> int:
    """Select the split-K factor for improved parallelism on small M.

    When the M dimension is small and K is large, splitting the K
    reduction across multiple thread blocks improves GPU utilization.

    Args:
        m: M dimension (number of rows in the output).
        k: K dimension (reduction dimension).
        block_k: Block size along the K dimension.

    Returns:
        Split-K factor (1, 2, 4, or 8). Returns 1 for no split.
    """
    if block_k <= 0:
        return 1
    tiles = math.ceil(k / block_k)
    if tiles <= 1:
        return 1
    if m <= 128:
        if tiles >= 256:
            return 8
        if tiles >= 128:
            return 4
        if tiles >= 64:
            return 2
    return 1


def _xla_choices(hardware: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return block size choice tuples for XLA backend on the given hardware.

    Args:
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        Tuple of (block_m_choices, block_n_choices, block_k_choices) where
        each element is a tuple of valid block sizes for that dimension.
    """
    if hardware == "cpu":
        return (128, 256, 512), (128, 256, 512), (64, 128, 256)
    if hardware == "tpu":
        return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)
    return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)


def _tpu_predecode_fits_memory_model(k: int, n: int) -> bool:
    """Check whether a predecoded weight matrix fits the TPU memory budget.

    The predecode path materializes the full dequantized weight matrix
    (K x N in bfloat16). This function checks whether that temporary
    fits within the configurable memory cap.

    Args:
        k: K dimension of the weight matrix.
        n: N dimension of the weight matrix.

    Returns:
        True if the predecoded matrix fits within the memory cap.
    """
    cap_raw = os.getenv("EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES")
    if cap_raw is None:
        cap = 256 * 1024 * 1024
    else:
        try:
            cap = int(cap_raw)
        except ValueError:
            cap = 256 * 1024 * 1024
    if cap <= 0:
        return True
    return (k * n * 2) <= cap


def _prefer_packed_tpu_path(
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    mode: str,
    bits: int,
    group_size: int,
) -> bool:
    """Heuristic for preferring packed TPU path over predecode.

    Mirrors the kernel-side heuristic so higher-layer autotune can treat
    packed/predecode as separate candidate configurations.
    """
    if bits not in (4, 8):
        return False
    if mode != "nf4":
        return False
    enough_n = n >= max(512, 2 * block_n)
    enough_m = m >= max(64, block_m // 2)
    enough_k = k >= max(256, block_k)
    valid_grouping = group_size <= block_n
    return enough_n and enough_m and enough_k and valid_grouping


def _pallas_tpu_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for TPU Pallas quantized matmul.

    Selects block sizes based on matrix dimensions, quantization parameters,
    and TPU memory model constraints.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for TPU Pallas execution.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    m, k, n, _ = _infer_mkn(inv, group_size)
    values_per_word = 32 // bits if bits in (4, 8) else 1
    align_n = _lcm(128, _lcm(group_size, values_per_word))

    block_m = 256 if m >= 2048 else 128
    block_n = 256 if n >= 1024 else 128
    block_k = 256 if k >= 4096 else 128

    block_n = max(align_n, _ceil_div(block_n, align_n) * align_n)
    block_k = max(128, _ceil_div(block_k, 128) * 128)

    # If predecode would exceed the memory model cap, bias to smaller tiles to
    # reduce temporary pressure and make XLA fallback more likely.
    if not _tpu_predecode_fits_memory_model(k, n):
        block_m = 128
        block_n = max(128, align_n)
        block_k = 128

    packed_legal = False
    try:
        from ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core import (
            is_packed_tpu_legal_forward as _packed_legal_forward,
        )

        x = _inv_arg(inv, "x", 0)
        w = _inv_arg(inv, "w", 1)
        scales = _inv_arg(inv, "scales", 2)
        packed_legal = bool(
            _packed_legal_forward(
                x,
                w,
                scales,
                group_size=group_size,
                bits=bits,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
            )
        )
    except Exception:
        packed_legal = False

    predecode_ok = _tpu_predecode_fits_memory_model(k, n)
    if packed_legal and _prefer_packed_tpu_path(
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        mode=mode,
        bits=bits,
        group_size=group_size,
    ):
        tpu_path = "packed"
    elif predecode_ok:
        tpu_path = "predecode"
    elif packed_legal:
        tpu_path = "packed"
    else:
        tpu_path = "hybrid"

    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
        use_bf16=True,
        split_k=None,
        tpu_path=tpu_path,
        platform="pallas",
        backend="tpu",
    )


def _pallas_tpu_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
    """Generate candidate configurations for autotuning TPU Pallas quantized matmul.

    Creates a grid of block size combinations, filtering by alignment
    constraints and TPU memory model limits. Configurations that are legal
    for the packed TPU path are preferred.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        List of QuantizedMatmulConfig candidates for TPU autotuning.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    _m, k, n, _ = _infer_mkn(inv, group_size)

    values_per_word = 32 // bits if bits in (4, 8) else 1
    align_n = _lcm(128, _lcm(group_size, values_per_word))
    bm_opts = (128, 256)
    bn_seed = (128, 256, 512, 1024)
    bk_opts = (128, 256, 512)

    bn_opts = []
    for v in bn_seed:
        aligned = max(align_n, _ceil_div(v, align_n) * align_n)
        bn_opts.append(aligned)
    bn_opts = sorted(set(bn_opts))

    # Memory model: for very large KxN with predecode path, keep candidate set small.
    if not _tpu_predecode_fits_memory_model(k, n):
        bn_opts = [max(align_n, 128)]
        bk_opts = (128,)

    x = _inv_arg(inv, "x", 0)
    w = _inv_arg(inv, "w", 1)
    scales = _inv_arg(inv, "scales", 2)
    try:
        from ejkernel.kernels._pallas.tpu.quantized_matmul._pallas_impl_core import (
            is_packed_tpu_legal_forward as _packed_legal_forward,
        )
    except Exception:
        _packed_legal_forward = None

    configs: list[QuantizedMatmulConfig] = []
    predecode_ok = _tpu_predecode_fits_memory_model(k, n)

    for bm in bm_opts:
        for bn in bn_opts:
            for bk in bk_opts:
                packed_legal = False
                if _packed_legal_forward is not None and bits in (4, 8):
                    packed_legal = bool(
                        _packed_legal_forward(
                            x,
                            w,
                            scales,
                            group_size=group_size,
                            bits=bits,
                            block_m=bm,
                            block_n=bn,
                            block_k=bk,
                        )
                    )
                path_opts: list[str] = []
                if predecode_ok:
                    path_opts.append("predecode")
                if packed_legal:
                    path_opts.append("packed")
                if predecode_ok and packed_legal:
                    path_opts.append("hybrid")

                for path in path_opts:
                    cfg = QuantizedMatmulConfig(
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                        num_warps=4,
                        num_stages=2,
                        use_bf16=True,
                        split_k=None,
                        tpu_path=path,
                        platform="pallas",
                        backend="tpu",
                    )
                    configs.append(cfg)
                if not path_opts and _packed_legal_forward is not None and bits in (4, 8):
                    # Keep candidates that can at least execute via backend fallback.
                    if _packed_legal_forward(
                        x,
                        w,
                        scales,
                        group_size=group_size,
                        bits=bits,
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                    ):
                        configs.append(
                            QuantizedMatmulConfig(
                                block_m=bm,
                                block_n=bn,
                                block_k=bk,
                                num_warps=4,
                                num_stages=2,
                                use_bf16=True,
                                split_k=None,
                                tpu_path="packed",
                                platform="pallas",
                                backend="tpu",
                            )
                        )

    if not configs:
        configs.append(_pallas_tpu_heuristic_cfg(inv))
    return configs


def _xla_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> list[QuantizedMatmulConfig]:
    """Generate candidate configurations for autotuning XLA quantized matmul.

    Creates block size combinations based on matrix dimensions and hardware
    target, selecting nearby power-of-2 choices and ensuring alignment with
    quantization group size. Returns up to 6 candidates sorted by proximity
    to the actual matrix dimensions.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        List of up to 6 QuantizedMatmulConfig candidates for XLA autotuning.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, K, N, transpose = _infer_mkn(inv, group_size)
    values_per_word = 32 // bits if bits in (4, 8) else 1
    align = _lcm(group_size, values_per_word)

    bm_choices, bn_choices, bk_choices = _xla_choices(hardware)
    base_m = _nearest_choices(M, bm_choices, count=1)[0]
    base_n = _nearest_choices(N, bn_choices, count=1)[0]
    base_k = _nearest_choices(K, bk_choices, count=1)[0]

    bm_opts = _expand_choices(base_m, bm_choices)
    bn_opts = _expand_choices(base_n, bn_choices)
    bk_opts = _expand_choices(base_k, bk_choices)

    if transpose:
        bk_opts = _ensure_aligned(bk_opts, align, max(bk_choices))
    else:
        bn_opts = _ensure_aligned(bn_opts, align, max(bn_choices))

    use_bf16 = False if hardware == "cpu" else _prefer_bf16(_inv_arg(inv, "x", 0))
    configs = []
    for bm in bm_opts:
        for bn in bn_opts:
            for bk in bk_opts:
                configs.append(
                    QuantizedMatmulConfig(
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                        num_warps=4,
                        num_stages=3,
                        use_bf16=use_bf16,
                        split_k=None,
                        platform="xla",
                        backend="any",
                    )
                )

    def _score(cfg: QuantizedMatmulConfig) -> int:
        """Score a config by Manhattan distance from actual matrix dimensions."""
        return abs(cfg.block_m - M) + abs(cfg.block_n - N) + abs(cfg.block_k - K)

    configs.sort(key=_score)
    return configs[:6]


def _xla_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for XLA quantized matmul.

    Returns the top-ranked candidate from _xla_candidate_cfgs, or a
    minimal fallback configuration if no candidates are generated.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.
        hardware: Hardware target string ("cpu", "tpu", or "gpu").

    Returns:
        A QuantizedMatmulConfig tuned for XLA execution.
    """
    candidates = _xla_candidate_cfgs(inv, hardware)
    return candidates[0] if candidates else QuantizedMatmulConfig(platform="xla", backend="any")


def _triton_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for Triton GPU quantized matmul.

    Selects block sizes, warp counts, pipeline stages, and split-K factor
    based on matrix dimensions and quantization parameters. Includes
    shared memory usage estimation to avoid exceeding GPU limits.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for Triton GPU execution.
    """
    mode = str(inv.kwargs.get("mode", "affine"))
    group_size, bits = _resolve_qparams(mode, inv.kwargs.get("group_size"), inv.kwargs.get("bits"))
    M, K, N, _ = _infer_mkn(inv, group_size)

    block_m = 128 if M >= 128 else 64
    block_n = 128 if N >= 128 else 64
    if bits == 8 or group_size >= 128:
        block_k = 32
        num_stages = 2
    else:
        block_k = 64 if K >= 1024 else 32
        num_stages = 3 if block_k >= 64 else 2
    num_warps = 4 if (block_m >= 128 and block_n >= 128) else 2

    elem_size = 2
    smem_limit = 96 * 1024
    smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        num_stages = 2
        smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        block_k = 32
        num_stages = 2
        smem = (block_m * block_k + block_k * block_n) * elem_size * num_stages
    if smem > smem_limit:
        block_m = 64
        block_n = 64
        num_warps = 2
    split_k = _pick_split_k(M, K, block_k)

    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=split_k,
        platform="triton",
        backend="gpu",
    )


def _cuda_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for CUDA custom-call quantized matmul.

    Uses fixed 128x128x64 block sizes with 4 warps and 2 pipeline stages,
    which are well-suited for CUDA's custom-call codepath.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for CUDA custom-call execution.
    """
    return QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=64,
        num_warps=4,
        num_stages=2,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=None,
        platform="cuda",
        backend="gpu",
    )


def _cute_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
    """Generate a heuristic configuration for CuTe DSL quantized matmul.

    Uses fixed 128x128x64 block sizes with 4 warps and 2 pipeline stages,
    matching the CuTe DSL kernel's default tile shape.

    Args:
        inv: The kernel invocation containing input tensors and kwargs.

    Returns:
        A QuantizedMatmulConfig tuned for CuTe DSL execution.
    """
    return QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=64,
        num_warps=4,
        num_stages=2,
        use_bf16=_prefer_bf16(_inv_arg(inv, "x", 0)),
        split_k=None,
        platform="cute",
        backend="gpu",
    )


class QuantizedMatmul(Kernel[QuantizedMatmulConfig, Array]):
    """Quantized matrix multiplication kernel with configurable tiling and backend selection.

    This kernel performs matrix multiplication between a dense input matrix and a
    quantized weight matrix, supporting explicit quantization modes:
    affine, nf4, mxfp4, mxfp8, nvfp4, and nvfp8.

    The kernel automatically selects the optimal backend (Triton or XLA) based on
    the target platform and input characteristics. It supports autotuning for
    optimal block sizes and pipeline configurations.

    Attributes:
        op_id: Operation identifier ("quantized_matmul").

    Example:
        >>> kernel = QuantizedMatmul()
        >>> cfg = QuantizedMatmulConfig(block_m=128, block_n=128, block_k=64)
        >>> output = kernel.run(x, w_q, scales, zeros, cfg=cfg)
    """

    version = "1"

    def __init__(self) -> None:
        """Initialize the quantized matmul kernel."""
        super().__init__(op_id="quantized_matmul")

    def _resolve_inv_platform(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> Platform:
        """Resolve the target platform from invocation parameters.

        Args:
            inv: The kernel invocation containing configuration and arguments.

        Returns:
            The resolved Platform enum value.
        """
        platform = inv.kwargs.get("platform", None)
        if platform is None and inv.override_cfg is not None:
            platform = inv.override_cfg.platform
        return detect_platform(self.op_id, platform if platform is not None else "auto")

    def get_impl(self, cfg: QuantizedMatmulConfig):
        """Get the kernel implementation for the given configuration.

        Args:
            cfg: Kernel configuration specifying platform and backend.

        Returns:
            The registered kernel implementation function.
        """
        try:
            backend_name = jax.default_backend()
        except Exception:
            backend_name = "cpu"
        platform = detect_platform(
            self.op_id,
            cfg.platform,
            prefer_pallas=backend_name == "tpu",
            prefer_cuda=backend_name in ("gpu", "cuda"),
        )
        return kernel_registry.get(self.op_id, platform=platform, backend=cfg.backend)

    def run(
        self,
        x: Float[Array, "m k"],
        w: Array,
        scales: Array,
        zeros: Array | None = None,
        transpose: bool = False,
        group_size: int | None = None,
        bits: int | None = None,
        mode: QuantizationMode = "affine",
        axis: QuantizationAxis | None = None,
        gemv_mode: GemvMode = "auto",
        revsplit_k: RevSplitKMode = "auto",
        revsplit_k_parts: int | None = None,
        allow_dense_fallback: bool = True,
        _resolved_platform: str | None = None,
        platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
        *,
        cfg: QuantizedMatmulConfig,
    ) -> Float[Array, "m n"]:
        """Execute quantized matmul with the selected backend.

        Performs the computation:
            - ``output = x @ dequantize(w, scales, zeros)``
            - ``output = x @ dequantize(w, scales, zeros).T`` when ``transpose=True``

        Args:
            x: Input matrix of shape (M, K) in float dtype.
            w: Packed uint32 weights produced by quantize(). Shape depends on
                transpose and bits settings.
            scales: Per-group scales array. Shape is (N, K//group_size) for
                transpose=True or (K, N//group_size) for transpose=False.
            zeros: Per-group affine zero-points (canonical affine metadata).
                Required for affine mode and must be ``None`` for non-affine modes.
            transpose: If True, compute x @ w.T (weights stored in KxN layout).
                If False, compute x @ w (weights stored in KxN transposed layout).
            group_size: Group size used in quantization. If None, uses mode default.
            bits: Bit-width used in quantization. Honored for affine ({4,8});
                ignored for nf4/mxfp4/mxfp8/nvfp4/nvfp8.
            mode: Quantization mode. One of
                {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}.
            axis: Optional quantization axis convenience alias. "row" maps to
                transpose=False and "col" maps to transpose=True.
            platform: Platform override (triton/pallas/cuda/cute/xla/auto).
            cfg: Kernel configuration with block sizes and settings.

        Returns:
            Matrix multiplication result of shape (M, N). CUDA returns the same
            dtype as ``x``; other backends return float32.

        Notes:
            For best Triton performance, prepack weights in KxN layout using
            prepack_quantized_weights() and call with transpose=False.
            For affine mode, backend wrappers convert ``zeros`` to internal
            additive offsets right before kernel launch.
        """
        _ = _resolved_platform
        if platform is not None:
            cfg = QuantizedMatmulConfig(
                block_m=cfg.block_m,
                block_n=cfg.block_n,
                block_k=cfg.block_k,
                num_warps=cfg.num_warps,
                num_stages=cfg.num_stages,
                use_bf16=cfg.use_bf16,
                split_k=cfg.split_k,
                tpu_path=cfg.tpu_path,
                platform=platform,
                backend=cfg.backend,
            )

        impl = self.get_impl(cfg)
        resolved_platform = _resolved_platform
        if resolved_platform is None:
            try:
                backend_name = jax.default_backend()
            except Exception:
                backend_name = "cpu"
            resolved_platform = detect_platform(
                self.op_id,
                cfg.platform,
                prefer_pallas=backend_name == "tpu",
                prefer_cuda=backend_name in ("gpu", "cuda"),
            ).value

        impl_kwargs = dict(
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=axis,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            allow_dense_fallback=allow_dense_fallback,
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k,
            use_bf16=cfg.use_bf16,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            split_k=cfg.split_k,
        )
        if resolved_platform == Platform.PALLAS.value and cfg.tpu_path is not None:
            impl_kwargs["tpu_path"] = cfg.tpu_path

        return impl(
            x,
            w,
            scales,
            zeros,
            **impl_kwargs,
        )

    def heuristic_cfg(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return default heuristic configuration for any platform.

        Args:
            inv: The kernel invocation (unused for default heuristics).

        Returns:
            A default QuantizedMatmulConfig with balanced block sizes.
        """
        return _xla_heuristic_cfg(inv, "cpu")

    def candidate_cfgs(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for default candidates).

        Returns:
            List of QuantizedMatmulConfig candidates to try during autotuning.
        """
        return _xla_candidate_cfgs(inv, "cpu")

    def heuristic_cfg_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for GPU.

        Args:
            inv: The kernel invocation for platform resolution.

        Returns:
            A QuantizedMatmulConfig optimized for GPU execution.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.TRITON:
            return _triton_heuristic_cfg(inv)
        if resolved == Platform.CUDA:
            return _cuda_heuristic_cfg(inv)
        if resolved == Platform.CUTE:
            return _cute_heuristic_cfg(inv)
        return _xla_heuristic_cfg(inv, "gpu")

    def heuristic_cfg_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for CPU.

        Args:
            inv: The kernel invocation (unused for CPU heuristics).

        Returns:
            A QuantizedMatmulConfig optimized for CPU execution.
        """
        return _xla_heuristic_cfg(inv, "cpu")

    def heuristic_cfg_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
        """Return heuristic configuration optimized for TPU.

        Args:
            inv: The kernel invocation (unused for TPU heuristics).

        Returns:
            A QuantizedMatmulConfig optimized for TPU execution.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.PALLAS:
            return _pallas_tpu_heuristic_cfg(inv)
        return _xla_heuristic_cfg(inv, "tpu")

    def candidate_cfgs_gpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return GPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation for platform resolution.

        Returns:
            List of QuantizedMatmulConfig candidates optimized for GPU.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved in (Platform.TRITON, Platform.CUDA, Platform.CUTE):
            return []
        return _xla_candidate_cfgs(inv, "gpu")

    def candidate_cfgs_cpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return CPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for CPU candidates).

        Returns:
            List of QuantizedMatmulConfig candidates optimized for CPU.
        """
        return _xla_candidate_cfgs(inv, "cpu")

    def candidate_cfgs_tpu(self, inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
        """Return TPU-specific candidate configurations for autotuning.

        Args:
            inv: The kernel invocation (unused for TPU candidates).

        Returns:
            List of QuantizedMatmulConfig candidates optimized for TPU.
        """
        resolved = self._resolve_inv_platform(inv)
        if resolved == Platform.PALLAS:
            return _pallas_tpu_candidate_cfgs(inv)
        return _xla_candidate_cfgs(inv, "tpu")

    candidate_cfgs_shard_map_gpu = candidate_cfgs_gpu


_quantized_matmul_executor: Executor[QuantizedMatmulConfig, Array] = Executor(
    ConfigSelectorChain(
        cache=ConfigCache(),
        policy=AutotunePolicy(
            allow_autotune=True,
            cache_miss_fallback=os.getenv("EJKERNEL_AUTOTUNE_POLICY", "autotune"),
            validate_backward=False,
        ),
        tuner=Tuner(warmup=5, iters=100),
        persistent=PersistentCache("quantized-matmul", cfg_type=QuantizedMatmulConfig),
    )
)


def _quantized_matmul_impl(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    /,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    allow_dense_fallback: bool = True,
    platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Execute quantized matrix multiplication with normalized qparams.

    Internal implementation that normalizes quantization parameters (mode,
    group_size, bits, axis, transpose), resolves the kernel family (GEMM vs
    GEMV vs revsplit-K), validates zeros requirements, and dispatches to the
    appropriate platform executor.

    Args:
        x: Input matrix of shape (M, K) in float dtype.
        w: Packed quantized weights.
        scales: Per-group scale factors.
        zeros: Per-group zero-points (required for affine mode, None otherwise).
        transpose: If True, compute x @ dequantize(w).T.
        group_size: Quantization group size.
        bits: Quantization bit-width.
        mode: Quantization mode string.
        axis: Optional quantization axis convenience alias.
        gemv_mode: GEMV kernel selection mode ("auto", "on", "off").
        revsplit_k: Reverse split-K mode ("auto", "on", "off").
        revsplit_k_parts: Number of parts for reverse split-K.
        allow_dense_fallback: When dispatching to XLA, controls whether the
            XLA implementation may fall back to dequantize+matmul when blocked
            fusion preconditions are not met.
        platform: Platform override.
        cfg: Optional configuration override.

    Returns:
        Matrix multiplication result of shape (M, N).

    Raises:
        ValueError: If affine mode is used without zeros, or non-affine mode
            with zeros.
    """
    transpose = _static_bool(transpose, "transpose")
    if group_size is not None:
        group_size = _static_int(group_size, "group_size")
    if bits is not None:
        bits = _static_int(bits, "bits")
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    if axis is None:
        raise ValueError("_quantized_matmul_impl expects axis to be resolved (pass axis='row' or axis='col').")
    if axis not in {"row", "col"}:
        raise ValueError(f"_quantized_matmul_impl expected axis in {{'row','col'}}, got {axis!r}.")
    expected_transpose = axis == "col"
    if expected_transpose != bool(transpose):
        raise ValueError(
            "_quantized_matmul_impl received inconsistent axis/transpose: "
            f"axis={axis!r} requires transpose={expected_transpose}, got transpose={bool(transpose)}."
        )
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)
    allow_dense_fallback = _static_bool(allow_dense_fallback, "allow_dense_fallback")

    if int(x.shape[0]) == 1 and mode in {"mxfp4", "mxfp8"} and gemv_mode == "on":
        warnings.warn(
            "gemv_mode='on' with MX modes at M==1 is mapped to GEMM-SplitK for GemLite parity.",
            RuntimeWarning,
            stacklevel=2,
        )

    family, family_revsplit_parts = select_qmm_kernel_family(
        m=int(x.shape[0]),
        mode=mode,
        bits=bits,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if family == "gemv_revsplitk":
        revsplit_k_parts = family_revsplit_parts

    if mode == "affine" and zeros is None:
        raise ValueError("affine quantized_matmul requires `zeros`.")
    if mode != "affine" and zeros is not None:
        raise ValueError("zeros must be None for non-affine modes.")

    try:
        backend_name = jax.default_backend()
    except Exception:
        backend_name = "cpu"

    prefer_cuda = backend_name in ("gpu", "cuda") and axis != "col"
    # Prefer Triton for axis='col' (transpose=True) on CUDA backends by default.
    # CUDA supports transpose=True, but Triton is generally more competitive
    # for fused transpose workloads unless proven otherwise.
    prefer_triton = backend_name in ("gpu", "cuda") and axis == "col"
    resolved = detect_platform(
        "quantized_matmul",
        platform if platform is not None else (cfg.platform if cfg is not None else "auto"),
        prefer_pallas=backend_name == "tpu",
        prefer_cuda=prefer_cuda,
        prefer_triton=prefer_triton,
    )
    dispatch_platform = resolved.value
    extra_kwargs = {}
    if resolved == Platform.XLA:
        extra_kwargs["allow_dense_fallback"] = allow_dense_fallback

    if resolved in (Platform.TRITON, Platform.CUDA, Platform.CUTE):
        with policy_override(
            _quantized_matmul_executor.chooser,
            allow_autotune=False,
            cache_miss_fallback="heuristics",
        ):
            return _quantized_matmul_executor(
                QuantizedMatmul(),
                x=x,
                w=w,
                scales=scales,
                zeros=zeros,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                axis=axis,
                gemv_mode=gemv_mode,
                revsplit_k=revsplit_k,
                revsplit_k_parts=revsplit_k_parts,
                _resolved_platform=resolved.value,
                platform=dispatch_platform,
                _cfg=cfg,
                **extra_kwargs,
            )

    return _quantized_matmul_executor(
        QuantizedMatmul(),
        x=x,
        w=w,
        scales=scales,
        zeros=zeros,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        axis=axis,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        _resolved_platform=resolved.value,
        platform=dispatch_platform,
        _cfg=cfg,
        **extra_kwargs,
    )


def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    /,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    fuse: bool = True,
    strict_fuse: bool | None = None,
    platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication with fused dequantization and custom VJP.

    Performs output = x @ dequantize(w, scales, zeros) with automatic backend
    selection and a custom backward pass that dequantizes weights for the
    gradient computation. Supports affine, nf4, mxfp4, mxfp8, nvfp4, and
    nvfp8 quantization modes.

    Args:
        x: Input matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights produced by quantize().
        scales: Per-group scale factors.
        zeros: Per-group zero-points. Required for affine mode, must be
            None for non-affine modes.
        transpose: If True, compute x @ dequantize(w).T.
        group_size: Quantization group size. If None, uses mode default.
        bits: Quantization bit-width. Honored for affine ({1,2,4,8});
            ignored for nf4/mxfp4/mxfp8/nvfp4/nvfp8.
        mode: Quantization mode. One of
            {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}.
        axis: Optional quantization axis convenience alias. "row" maps to
            transpose=False; "col" maps to transpose=True.
        gemv_mode: GEMV kernel selection mode ("auto", "on", "off").
        revsplit_k: Reverse split-K mode ("auto", "on", "off").
        revsplit_k_parts: Number of parts for reverse split-K.
        fuse: If True, run platform fused quantized kernels. If False, force
            reference path (dequantize then matmul) using XLA/JAX ops.
        strict_fuse: If True, disallow dense dequantize+matmul fallbacks inside
            fused implementations (notably the XLA backend). When None, reads
            environment variable ``EJKERNEL_QMM_STRICT_FUSED``.
        platform: Platform override (triton/pallas/cuda/cute/xla/auto).
        cfg: Optional configuration override.

    Returns:
        Matrix multiplication result of shape (M, N).
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)

    if strict_fuse is None:
        env = os.getenv("EJKERNEL_QMM_STRICT_FUSED", "").strip().lower()
        strict_fuse = env in {"1", "true", "on", "yes"}
    strict_fuse_n = _static_bool(strict_fuse, "strict_fuse")

    fuse = _static_bool(fuse, "fuse")
    if strict_fuse_n and not fuse:
        raise ValueError("strict_fuse=True requires fuse=True.")

    if fuse and mode_n == "affine" and bits_n not in (4, 8):
        msg = "fuse=True with affine bits not in {4,8} is unsupported."
        if strict_fuse_n:
            raise ValueError(msg)
        warnings.warn(
            f"{msg} Falling back to reference dequantize+matmul path.",
            RuntimeWarning,
            stacklevel=2,
        )
        fuse = False
    if fuse and jax.default_backend() == "mps":
        msg = "fuse=True on MPS currently falls back to reference dequantize+matmul for stability."
        if strict_fuse_n:
            raise ValueError(msg)
        warnings.warn(
            msg,
            RuntimeWarning,
            stacklevel=2,
        )
        fuse = False
    runtime_axis, runtime_transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    validate_packed_quantized_matmul_layout(
        x,
        w,
        scales,
        zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=runtime_axis,
        transpose=runtime_transpose,
    )

    def _inner(xi, wi, si, zi):
        """Dispatch to _quantized_matmul_impl with captured quantization parameters."""
        if not fuse:
            from ejkernel.quantization._quants.quantizations import quantized_matmul as dense_quantized_matmul

            return dense_quantized_matmul(
                xi,
                wi,
                si,
                zi,
                transpose=runtime_transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                axis=runtime_axis,
            )

        return _quantized_matmul_impl(
            xi,
            wi,
            si,
            zi,
            transpose=runtime_transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=runtime_axis,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            allow_dense_fallback=not strict_fuse_n,
            platform=platform,
            cfg=cfg,
        )

    @jax.custom_vjp
    def _inner_vjp(xi, wi, si, zi):
        """Forward pass wrapper decorated with custom_vjp for backward compatibility."""
        return _inner(xi, wi, si, zi)

    def _inner_fwd(xi, wi, si, zi):
        """Custom VJP forward: compute output and save residuals (w, scales, zeros)."""
        y = _inner(xi, wi, si, zi)
        return y, (wi, si, zi)

    def _inner_bwd(res, g):
        """Custom VJP backward: dequantize weights and compute grad_x (grad_w/scales/zeros are zero)."""
        wi, si, zi = res
        from ejkernel.quantization._quants.quantizations import dequantize

        dequant_axis: QuantizationAxis = "col" if runtime_transpose else "row"
        w_f = dequantize(
            wi,
            si,
            zi,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=dequant_axis,
        )
        if runtime_transpose:
            grad_x = g @ w_f
        else:
            grad_x = g @ w_f.T

        grad_w = jnp.zeros_like(wi)
        grad_scales = jnp.zeros_like(si)
        grad_zeros = jnp.zeros_like(zi) if zi is not None else None
        return grad_x, grad_w, grad_scales, grad_zeros

    _inner_vjp.defvjp(_inner_fwd, _inner_bwd)

    return _inner_vjp(x, w, scales, zeros)
