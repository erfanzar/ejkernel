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

"""Shared quantization parameter normalization utilities.

This module centralizes normalization and validation for quantization mode,
bit-width, group-size, and quantization axis.
"""

from __future__ import annotations

from typing import Literal

from .grouping import _require_bits

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
QuantizationAxis = Literal["row", "col"]
BackendQuantizationMode = QuantizationMode
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]
KernelFamily = Literal["gemm", "gemm_splitk", "gemv_splitk", "gemv_revsplitk"]


def normalize_axis(axis: str | None, *, default: QuantizationAxis = "row") -> QuantizationAxis:
    """Normalize quantization axis name."""
    if axis is None:
        return default
    axis = axis.lower()
    if axis not in {"row", "col"}:
        raise ValueError("axis must be one of {'row','col'}.")
    return axis  # type: ignore[return-value]


def normalize_mode_and_bits(
    mode: str,
    bits: int | None,
) -> tuple[QuantizationMode, int | None, bool]:
    """Normalize explicit quantization mode and optional bits.

    Returns canonical (mode, bits, used_legacy_alias).
    """
    mode = mode.lower()
    if mode in {"mxfp", "nvfp"}:
        raise ValueError("Use explicit mode names: {'affine','nf4','mxfp4','mxfp8','nvfp4','nvfp8'}.")
    if mode not in {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}:
        raise ValueError("Unsupported quantization mode. Use one of {'affine','nf4','mxfp4','mxfp8','nvfp4','nvfp8'}.")

    if bits is not None:
        bits = int(bits)

    return mode, bits, False  # type: ignore[return-value]


def resolve_qparams(
    mode: str,
    group_size: int | None,
    bits: int | None,
) -> tuple[QuantizationMode, int, int, bool]:
    """Resolve explicit mode, bit-width, and group-size.

    Rules:
      - affine: bits in {4,8}, group_size in {32,64,128}
      - nf4: bits fixed to 4, group_size in {32,64,128}
      - mxfp4/mxfp8: group_size=32
      - nvfp4/nvfp8: group_size=16
    """
    mode, bits, used_legacy = normalize_mode_and_bits(mode, bits)

    if mode == "affine":
        bits = 4 if bits is None else _require_bits(bits, {4, 8})
        group_size = 64 if group_size is None else int(group_size)
        if group_size not in {32, 64, 128}:
            raise ValueError("affine mode supports group_size in {32,64,128}.")
        return mode, group_size, bits, used_legacy

    if mode == "nf4":
        bits = 4
        group_size = 64 if group_size is None else int(group_size)
        if group_size not in {32, 64, 128}:
            raise ValueError("nf4 mode supports group_size in {32,64,128}.")
        return mode, group_size, bits, used_legacy

    if mode in {"mxfp4", "mxfp8"}:
        bits = 4 if mode == "mxfp4" else 8
        group_size = 32 if group_size is None else int(group_size)
        if group_size != 32:
            raise ValueError(f"{mode} requires group_size=32.")
        return mode, group_size, bits, used_legacy

    bits = 4 if mode == "nvfp4" else 8
    group_size = 16 if group_size is None else int(group_size)
    if group_size != 16:
        raise ValueError(f"{mode} requires group_size=16.")
    return mode, group_size, bits, used_legacy


def to_backend_mode(mode: QuantizationMode, bits: int) -> BackendQuantizationMode:
    """Map explicit mode to backend mode key.

    The ``bits`` parameter is accepted for backward compatibility and ignored.
    """
    del bits
    return mode


def normalize_gemv_mode(mode: str | None) -> GemvMode:
    """Normalize GEMV dispatch override mode."""
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("gemv_mode must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_mode(mode: str | None) -> RevSplitKMode:
    """Normalize reverse split-K dispatch override mode."""
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("revsplit_k must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_parts(parts: int | None) -> int | None:
    """Normalize optional reverse split-K partition count.

    Only powers of two in {1,2,4,8,16} are supported.
    """
    if parts is None:
        return None
    parts = int(parts)
    if parts not in {1, 2, 4, 8, 16}:
        raise ValueError("revsplit_k_parts must be one of {1,2,4,8,16}.")
    return parts


def is_effective_4bit_mode(mode: QuantizationMode, bits: int) -> bool:
    """Return whether the effective runtime quantization is 4-bit."""
    if mode == "affine":
        return int(bits) == 4
    return mode in {"nf4", "mxfp4", "nvfp4"}


def select_qmm_kernel_family(
    *,
    m: int,
    mode: QuantizationMode,
    bits: int,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
) -> tuple[KernelFamily, int | None]:
    """Select QMM kernel family using GemLite-style policy.

    Policy:
      - M > 64 -> gemm
      - 1 < M <= 64 -> gemm_splitk
      - M == 1 -> GEMV family
        - 4-bit effective mode: gemv_revsplitk (auto)
        - 8-bit effective mode: gemv_splitk (auto)
    """
    m = int(m)
    if m <= 0:
        raise ValueError("Input activation leading dimension M must be >= 1.")

    gemv_mode_n = normalize_gemv_mode(gemv_mode)
    revsplit_k_n = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts_n = normalize_revsplitk_parts(revsplit_k_parts)
    is_4bit = is_effective_4bit_mode(mode, bits)

    if gemv_mode_n == "on" and m != 1:
        raise ValueError("gemv_mode='on' requires M == 1.")

    # GemLite parity: MX paths use GEMM-SplitK for M==1.
    if m == 1 and mode in {"mxfp4", "mxfp8"}:
        return "gemm_splitk", None

    use_gemv = (m == 1) if gemv_mode_n == "auto" else (gemv_mode_n == "on")
    if not use_gemv:
        return ("gemm" if m > 64 else "gemm_splitk"), None

    if revsplit_k_n == "on":
        if not is_4bit:
            raise ValueError("revsplit_k='on' requires an effective 4-bit mode.")
        parts = 2 if revsplit_k_parts_n is None else revsplit_k_parts_n
        if parts < 2:
            raise ValueError("revsplit_k='on' requires revsplit_k_parts >= 2.")
        return "gemv_revsplitk", parts

    if revsplit_k_n == "off":
        return "gemv_splitk", None

    # auto
    if is_4bit:
        if revsplit_k_parts_n is None:
            return "gemv_revsplitk", 2
        return "gemv_revsplitk", max(2, revsplit_k_parts_n)
    return "gemv_splitk", None


def resolve_runtime_axis_and_transpose(
    *,
    axis: str | None,
    transpose: bool,
) -> tuple[QuantizationAxis, bool]:
    """Resolve runtime axis and transpose consistency.

    Runtime mapping:
      - axis='row' -> transpose=False
      - axis='col' -> transpose=True

    If axis is omitted, transpose drives axis.
    """
    if axis is None:
        return ("col" if transpose else "row"), transpose

    axis_n = normalize_axis(axis)
    expected_transpose = axis_n == "col"
    if expected_transpose != bool(transpose):
        raise ValueError(
            "Inconsistent axis/transpose combination: "
            "axis='row' requires transpose=False, "
            "axis='col' requires transpose=True."
        )
    return axis_n, bool(transpose)


def resolve_prepack_axis(*, axis: str | None, transpose: bool) -> QuantizationAxis:
    """Resolve axis for prepack API.

    Backward-compatible mapping (when axis is omitted):
      - transpose=True  -> axis='row'
      - transpose=False -> axis='col'
    """
    if axis is not None:
        return normalize_axis(axis)
    return "row" if bool(transpose) else "col"
