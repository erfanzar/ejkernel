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
    """Normalize and validate the quantization axis name.

    Converts the axis string to lowercase and validates it against the
    allowed set of axis names.

    Args:
        axis: Axis name string, or None to use the default.
        default: Default axis to return when axis is None.

    Returns:
        Normalized axis name, either "row" or "col".

    Raises:
        ValueError: If axis is not one of {"row", "col"}.
    """
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
    """Normalize and validate explicit quantization mode and optional bit-width.

    Converts mode to lowercase and checks it against the set of supported
    modes. Ambiguous shorthand names ("mxfp", "nvfp") are rejected with
    a helpful error message.

    Args:
        mode: Quantization mode string (case-insensitive). Must be one of
            {"affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"}.
        bits: Optional bit-width override. Converted to int if provided.

    Returns:
        Tuple of (canonical_mode, bits, used_legacy_alias) where
        used_legacy_alias is always False for current modes.

    Raises:
        ValueError: If mode is ambiguous ("mxfp", "nvfp") or unsupported.
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
    """Resolve and validate the full set of quantization parameters.

    Applies mode-specific defaults and constraints for bit-width and
    group-size, raising clear errors when user-supplied values violate
    the rules for the chosen quantization mode.

    Mode-specific rules:
        - affine: bits in {4, 8} (default 4), group_size in {32, 64, 128} (default 64).
        - nf4: bits fixed to 4, group_size in {32, 64, 128} (default 64).
        - mxfp4 / mxfp8: bits fixed to 4 or 8 respectively, group_size must be 32.
        - nvfp4 / nvfp8: bits fixed to 4 or 8 respectively, group_size must be 16.

    Args:
        mode: Quantization mode string (passed through normalize_mode_and_bits).
        group_size: Number of elements per quantization group, or None for
            mode-specific default.
        bits: Bit-width for quantized values, or None for mode-specific default.

    Returns:
        Tuple of (mode, group_size, bits, used_legacy_alias).

    Raises:
        ValueError: If group_size or bits are invalid for the chosen mode.
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
    """Map a user-facing quantization mode to the backend mode key.

    In the current implementation, the backend mode is identical to the
    user-facing mode. The ``bits`` parameter is accepted for backward
    compatibility but is ignored.

    Args:
        mode: User-facing quantization mode.
        bits: Bit-width (ignored; kept for API compatibility).

    Returns:
        Backend quantization mode string, identical to the input mode.
    """
    del bits
    return mode


def normalize_gemv_mode(mode: str | None) -> GemvMode:
    """Normalize and validate the GEMV dispatch override mode.

    Args:
        mode: GEMV mode string ("auto", "on", "off"), or None for "auto".

    Returns:
        Normalized GEMV mode string.

    Raises:
        ValueError: If mode is not one of {"auto", "on", "off"}.
    """
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("gemv_mode must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_mode(mode: str | None) -> RevSplitKMode:
    """Normalize and validate the reverse split-K dispatch override mode.

    Args:
        mode: Reverse split-K mode string ("auto", "on", "off"), or None for "auto".

    Returns:
        Normalized reverse split-K mode string.

    Raises:
        ValueError: If mode is not one of {"auto", "on", "off"}.
    """
    if mode is None:
        return "auto"
    norm = str(mode).lower()
    if norm not in {"auto", "on", "off"}:
        raise ValueError("revsplit_k must be one of {'auto','on','off'}.")
    return norm  # type: ignore[return-value]


def normalize_revsplitk_parts(parts: int | None) -> int | None:
    """Normalize and validate the optional reverse split-K partition count.

    Only small powers of two are supported to ensure efficient GPU utilization.

    Args:
        parts: Number of split-K partitions, or None to leave unset.
            Must be one of {1, 2, 4, 8, 16} if provided.

    Returns:
        Validated partition count as int, or None if not provided.

    Raises:
        ValueError: If parts is not one of {1, 2, 4, 8, 16}.
    """
    if parts is None:
        return None
    parts = int(parts)
    if parts not in {1, 2, 4, 8, 16}:
        raise ValueError("revsplit_k_parts must be one of {1,2,4,8,16}.")
    return parts


def is_effective_4bit_mode(mode: QuantizationMode, bits: int) -> bool:
    """Check whether the effective runtime quantization is 4-bit.

    For affine mode, the bit-width is user-configurable and may be 4 or 8.
    For other modes, the effective bit-width is determined by the mode name
    (e.g., nf4, mxfp4, nvfp4 are all 4-bit).

    Args:
        mode: Quantization mode.
        bits: Bit-width parameter (only used for affine mode).

    Returns:
        True if the runtime quantization uses 4-bit values, False otherwise.
    """
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
    """Select the quantized matmul kernel family using a GemLite-style policy.

    The selection policy considers the activation batch dimension (M), the
    effective bit-width, and user overrides (gemv_mode, revsplit_k) to choose
    the best kernel family for the workload.

    Selection policy:
        - M > 64: "gemm" (standard batched GEMM).
        - 1 < M <= 64: "gemm_splitk" (split-K GEMM for moderate batch).
        - M == 1 with MX modes: "gemm_splitk" (MX paths use GEMM-SplitK for M=1).
        - M == 1, 4-bit effective: "gemv_revsplitk" (reverse split-K GEMV).
        - M == 1, 8-bit effective: "gemv_splitk" (split-K GEMV).

    Args:
        m: Activation leading dimension (batch size for matmul). Must be >= 1.
        mode: Quantization mode.
        bits: Bit-width parameter.
        gemv_mode: GEMV dispatch override ("auto", "on", "off").
        revsplit_k: Reverse split-K override ("auto", "on", "off").
        revsplit_k_parts: Number of reverse split-K partitions, or None.

    Returns:
        Tuple of (kernel_family, revsplitk_parts) where kernel_family is one
        of {"gemm", "gemm_splitk", "gemv_splitk", "gemv_revsplitk"} and
        revsplitk_parts is the number of partitions (or None if not applicable).

    Raises:
        ValueError: If m < 1, gemv_mode="on" with M != 1, or revsplit_k="on"
            with a non-4-bit mode.
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
    """Resolve runtime quantization axis and ensure transpose consistency.

    Enforces the bidirectional mapping between axis and transpose:
        - axis="row" requires transpose=False (no transpose needed).
        - axis="col" requires transpose=True (weight is transposed).

    When axis is omitted (None), the transpose flag determines the axis.

    Args:
        axis: Quantization axis string ("row" or "col"), or None to
            infer from the transpose flag.
        transpose: Whether the weight matrix is transposed at runtime.

    Returns:
        Tuple of (resolved_axis, resolved_transpose).

    Raises:
        ValueError: If axis and transpose are inconsistent (e.g.,
            axis="row" with transpose=True).
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
    """Resolve the quantization axis for the prepack API.

    When axis is explicitly provided, it is normalized and returned directly.
    When axis is omitted (None), the transpose flag provides backward-compatible
    axis inference:
        - transpose=True maps to axis="row" (legacy default).
        - transpose=False maps to axis="col".

    Args:
        axis: Explicit axis string ("row" or "col"), or None to infer
            from the transpose flag.
        transpose: Legacy transpose flag used for axis inference when
            axis is None.

    Returns:
        Resolved quantization axis, either "row" or "col".
    """
    if axis is not None:
        return normalize_axis(axis)
    return "row" if bool(transpose) else "col"
