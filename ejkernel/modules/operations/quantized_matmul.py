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

"""Quantized matrix multiplication operation with automatic optimization."""

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
)

from ..base import detect_platform
from .configs import QuantizedMatmulConfig


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    _, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return group_size, bits


def _fallback_platform_for_cuda_transpose() -> Platform:
    def _has_impl(cand: Platform, backend: str) -> bool:
        try:
            kernel_registry.get("quantized_matmul", platform=cand, backend=backend)
            return True
        except ValueError:
            return False

    try:
        backend = jax.default_backend()
    except Exception:
        backend = "cpu"

    if backend in ("gpu", "cuda"):
        for cand in (Platform.TRITON, Platform.CUTE):
            if _has_impl(cand, "gpu"):
                return cand
    if _has_impl(Platform.XLA, "any"):
        return Platform.XLA

    for cand, back in ((Platform.TRITON, "gpu"), (Platform.CUTE, "gpu"), (Platform.XLA, "any")):
        if _has_impl(cand, back):
            return cand
    return Platform.XLA


def _static_bool(value, name: str) -> bool:
    return jax.core.concrete_or_error(bool, value, f"{name} must be static.")


def _static_int(value, name: str) -> int:
    return jax.core.concrete_or_error(int, value, f"{name} must be static.")


def _lcm(a: int, b: int) -> int:
    if a <= 0:
        return int(b)
    if b <= 0:
        return int(a)
    return abs(a * b) // math.gcd(a, b)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _nearest_choices(value: int, choices: tuple[int, ...], count: int = 2) -> list[int]:
    ranked = sorted(set(choices), key=lambda x: abs(x - value))
    return sorted(ranked[:count])


def _expand_choices(value: int, choices: tuple[int, ...]) -> list[int]:
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
    if len(inv.args) > index:
        return inv.args[index]
    return inv.kwargs[name]


def _infer_mkn(inv: Invocation[QuantizedMatmulConfig, Array], group_size: int) -> tuple[int, int, int, bool]:
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
    dt = getattr(x, "dtype", None)
    if dt is None:
        return True
    return dt != jnp.float16


def _pick_split_k(m: int, k: int, block_k: int) -> int:
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
    if hardware == "cpu":
        return (128, 256, 512), (128, 256, 512), (64, 128, 256)
    if hardware == "tpu":
        return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)
    return (256, 512, 1024, 2048), (256, 512, 1024, 2048), (128, 256, 512)


def _tpu_predecode_fits_memory_model(k: int, n: int) -> bool:
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


def _pallas_tpu_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
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

    return QuantizedMatmulConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
        use_bf16=True,
        split_k=None,
        platform="pallas",
        backend="tpu",
    )


def _pallas_tpu_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array]) -> list[QuantizedMatmulConfig]:
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
    for bm in bm_opts:
        for bn in bn_opts:
            for bk in bk_opts:
                cfg = QuantizedMatmulConfig(
                    block_m=bm,
                    block_n=bn,
                    block_k=bk,
                    num_warps=4,
                    num_stages=2,
                    use_bf16=True,
                    split_k=None,
                    platform="pallas",
                    backend="tpu",
                )
                if _packed_legal_forward is not None and bits in (4, 8):
                    # Keep candidates that are legal for packed path, or that can
                    # still run through predecode memory model.
                    if _packed_legal_forward(
                        x,
                        w,
                        scales,
                        group_size=group_size,
                        bits=bits,
                        block_m=bm,
                        block_n=bn,
                        block_k=bk,
                    ) or _tpu_predecode_fits_memory_model(k, n):
                        configs.append(cfg)
                else:
                    configs.append(cfg)

    if not configs:
        configs.append(_pallas_tpu_heuristic_cfg(inv))
    return configs


def _xla_candidate_cfgs(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> list[QuantizedMatmulConfig]:
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
        return abs(cfg.block_m - M) + abs(cfg.block_n - N) + abs(cfg.block_k - K)

    configs.sort(key=_score)
    return configs[:6]


def _xla_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array], hardware: str) -> QuantizedMatmulConfig:
    candidates = _xla_candidate_cfgs(inv, hardware)
    return candidates[0] if candidates else QuantizedMatmulConfig(platform="xla", backend="any")


def _triton_heuristic_cfg(inv: Invocation[QuantizedMatmulConfig, Array]) -> QuantizedMatmulConfig:
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
    """Heuristic config for CUDA custom call path."""
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
    """Heuristic config for CuTe DSL path."""
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
        platform = detect_platform(self.op_id, cfg.platform, prefer_cuda=True)
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
                platform=platform,
                backend=cfg.backend,
            )

        impl = self.get_impl(cfg)
        return impl(
            x,
            w,
            scales,
            zeros,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            axis=axis,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            block_m=cfg.block_m,
            block_n=cfg.block_n,
            block_k=cfg.block_k,
            use_bf16=cfg.use_bf16,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            split_k=cfg.split_k,
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
    platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Execute quantized matrix multiplication with normalized qparams."""
    transpose = _static_bool(transpose, "transpose")
    if group_size is not None:
        group_size = _static_int(group_size, "group_size")
    if bits is not None:
        bits = _static_int(bits, "bits")
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    axis, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

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

    resolved = detect_platform(
        "quantized_matmul",
        platform if platform is not None else (cfg.platform if cfg is not None else "auto"),
        prefer_pallas=backend_name == "tpu",
        prefer_cuda=backend_name in ("gpu", "cuda"),
    )
    if resolved == Platform.CUDA and axis == "col":
        fallback = _fallback_platform_for_cuda_transpose()
        warnings.warn(
            (f"CUDA quantized_matmul does not support axis='col' (transpose=True); falling back to {fallback.value}."),
            RuntimeWarning,
            stacklevel=2,
        )
        resolved = fallback
    dispatch_platform = resolved.value

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
    platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    cfg: QuantizedMatmulConfig | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication with fused dequantization.

    See `_quantized_matmul_impl` for full documentation.
    """
    runtime_axis, runtime_transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)

    def _inner(xi, wi, si, zi):
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
            platform=platform,
            cfg=cfg,
        )

    @jax.custom_vjp
    def _inner_vjp(xi, wi, si, zi):
        return _inner(xi, wi, si, zi)

    def _inner_fwd(xi, wi, si, zi):
        y = _inner(xi, wi, si, zi)
        return y, (wi, si, zi)

    def _inner_bwd(res, g):
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
