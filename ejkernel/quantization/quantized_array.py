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

"""Container type for quantized tensors and metadata."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal

import jax

from ._quants.quantizations import dequantize, prepack_quantized_weights, quantize
from ._quants.quantizations import quantized_matmul as dense_quantized_matmul
from ._utils.qparams import (
    QuantizationAxis,
    QuantizationMode,
    normalize_axis,
    resolve_prepack_axis,
    resolve_qparams,
)
from .runtime import QuantRuntimeConfig


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class QuantizedArray:
    """Packed quantized tensor with all metadata required for runtime use."""

    data: jax.Array
    scales: jax.Array
    zeros: jax.Array | None
    mode: QuantizationMode
    group_size: int
    bits: int
    axis: QuantizationAxis
    runtime_config: QuantRuntimeConfig | None = None

    def tree_flatten(self):
        children = (self.data, self.scales, self.zeros)
        aux = (self.mode, int(self.group_size), int(self.bits), self.axis, self.runtime_config)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        mode, group_size, bits, axis, runtime_config = aux
        data, scales, zeros = children
        return cls(
            data=data,
            scales=scales,
            zeros=zeros,
            mode=mode,
            group_size=int(group_size),
            bits=int(bits),
            axis=axis,
            runtime_config=runtime_config,
        )

    @staticmethod
    def _shape_or_none(x: jax.Array | None) -> tuple[int, ...] | None:
        if x is None:
            return None
        return tuple(x.shape)

    @staticmethod
    def _numel(x: jax.Array | None) -> int:
        if x is None:
            return 0
        if x.ndim == 0:
            return 1
        return int(prod(int(dim) for dim in x.shape))

    @staticmethod
    def _storage_bits(x: jax.Array | None) -> int:
        if x is None:
            return 0
        return int(QuantizedArray._numel(x) * int(x.dtype.itemsize) * 8)

    @property
    def logical_num_values(self) -> int:
        """Number of dequantized scalar values represented by this payload."""
        if self.scales.ndim == 0:
            return int(self.group_size)
        prefix = 1 if self.scales.ndim == 1 else int(prod(int(dim) for dim in self.scales.shape[:-1]))
        n_groups = int(self.scales.shape[-1])
        return int(prefix * n_groups * int(self.group_size))

    @property
    def data_storage_bits(self) -> int:
        return self._storage_bits(self.data)

    @property
    def scales_storage_bits(self) -> int:
        return self._storage_bits(self.scales)

    @property
    def zeros_storage_bits(self) -> int:
        return self._storage_bits(self.zeros)

    @property
    def storage_bits(self) -> int:
        return self.data_storage_bits + self.scales_storage_bits + self.zeros_storage_bits

    @property
    def storage_bytes(self) -> int:
        return self.storage_bits // 8

    @property
    def storage_kib(self) -> float:
        return float(self.storage_bytes / 1024.0)

    @property
    def storage_mib(self) -> float:
        return float(self.storage_bytes / (1024.0 * 1024.0))

    @property
    def payload_bits_per_value(self) -> float:
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float(self.data_storage_bits / count)

    @property
    def metadata_bits_per_value(self) -> float:
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float((self.scales_storage_bits + self.zeros_storage_bits) / count)

    @property
    def actual_bits_per_value(self) -> float:
        """Effective bits/value including packed payload and metadata overhead."""
        count = self.logical_num_values
        if count == 0:
            return float("nan")
        return float(self.storage_bits / count)

    @property
    def effective_bits_per_value(self) -> float:
        """Alias for ``actual_bits_per_value``."""
        return self.actual_bits_per_value

    def __repr__(self) -> str:
        return (
            "QuantizedArray("
            f"data_shape={self._shape_or_none(self.data)}, "
            f"scales_shape={self._shape_or_none(self.scales)}, "
            f"zeros_shape={self._shape_or_none(self.zeros)}, "
            f"actual_bits_per_value={self.actual_bits_per_value:.3f}, "
            f"storage_bytes={self.storage_bytes}, "
            f"storage_kib={self.storage_kib:.3f}, "
            f"mode={self.mode!r}, "
            f"group_size={self.group_size}, "
            f"bits={self.bits}, "
            f"axis={self.axis!r}, "
            f"runtime_config={self.runtime_config!r}"
            ")"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_quantized(
        cls,
        data: jax.Array,
        scales: jax.Array,
        zeros: jax.Array | None = None,
        *,
        group_size: int | None = None,
        bits: int | None = None,
        mode: QuantizationMode = "affine",
        axis: QuantizationAxis = "row",
        runtime_config: QuantRuntimeConfig | None = None,
    ) -> QuantizedArray:
        """Construct a container from already-quantized buffers."""
        axis_n = normalize_axis(axis)
        mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)

        if mode_n == "affine" and zeros is None:
            raise ValueError("affine QuantizedArray requires `zeros`.")
        if mode_n != "affine" and zeros is not None:
            raise ValueError("zeros must be None for non-affine QuantizedArray modes.")

        return cls(
            data=data,
            scales=scales,
            zeros=zeros,
            mode=mode_n,
            group_size=group_size_n,
            bits=bits_n,
            axis=axis_n,
            runtime_config=runtime_config,
        )

    def as_tuple(
        self,
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, jax.Array]:
        """Return payload in legacy tuple form."""
        if self.mode == "affine":
            assert self.zeros is not None
            return self.data, self.scales, self.zeros
        return self.data, self.scales

    def dequantize(self, *, runtime_config: QuantRuntimeConfig | None = None) -> jax.Array:
        """Dequantize this container into floating-point weights."""
        runtime_cfg = self.runtime_config if runtime_config is None else runtime_config
        return dequantize(
            self.data,
            self.scales,
            self.zeros,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            axis=self.axis,
            runtime_config=runtime_cfg,
        )

    def matmul(
        self,
        x: jax.Array,
        *,
        fuse: bool = True,
        strict_fuse: bool | None = None,
        tpu_path: Literal["packed"] | None = None,
        allow_dense_fallback: bool | None = None,
        transpose: bool | None = None,
        axis: QuantizationAxis | None = None,
        platform: Literal["triton", "pallas", "cuda", "cute", "xla", "auto"] | None = None,
    ) -> jax.Array:
        """Run quantized matmul with either fused or reference execution."""
        axis_n = self.axis if axis is None else normalize_axis(axis)
        transpose_n = (axis_n == "col") if transpose is None else bool(transpose)
        if fuse:
            from ejkernel.modules.operations import quantized_matmul as fused_quantized_matmul

            return fused_quantized_matmul(
                x,
                self.data,
                self.scales,
                self.zeros,
                transpose=transpose_n,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                axis=axis_n,
                platform=platform,
                fuse=True,
                strict_fuse=strict_fuse,
                tpu_path=tpu_path,
                allow_dense_fallback=allow_dense_fallback,
            )

        return dense_quantized_matmul(
            x,
            self.data,
            self.scales,
            self.zeros,
            transpose=transpose_n,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            axis=axis_n,
        )


def quantize_array(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    axis: QuantizationAxis | None = None,
    runtime_config: QuantRuntimeConfig | None = None,
) -> QuantizedArray:
    """Quantize *w* and return a ``QuantizedArray`` container.

    If *axis* is omitted, a compatible axis is inferred from the input shape:
    - Prefer ``axis='row'`` when ``w.shape[-2]`` is divisible by ``group_size``.
    - Otherwise use ``axis='col'`` when ``w.shape[-1]`` is divisible by ``group_size``.
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    if axis is None:
        if w.ndim < 2:
            raise ValueError("quantize_array expects inputs with two or more dimensions.")
        if int(w.shape[-2]) % group_size_n == 0:
            axis_n: QuantizationAxis = "row"
        elif int(w.shape[-1]) % group_size_n == 0:
            axis_n = "col"
        else:
            raise ValueError(
                "group_size is incompatible with both possible grouping axes. "
                f"input_shape={tuple(w.shape)}, group_size={group_size_n}, "
                f"dim[-2]={int(w.shape[-2])}, dim[-1]={int(w.shape[-1])}. "
                "Pass axis='row' or axis='col' explicitly."
            )
    else:
        axis_n = normalize_axis(axis)
    out = quantize(
        w,
        group_size=group_size_n,
        bits=bits_n,
        mode=mode_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )
    if mode_n == "affine":
        data, scales, zeros = out
    else:
        data, scales = out
        zeros = None
    return QuantizedArray(
        data=data,
        scales=scales,
        zeros=zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )


def prepack_quantized_array(
    w: jax.Array,
    /,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    transpose: bool = True,
    axis: QuantizationAxis | None = None,
    runtime_config: QuantRuntimeConfig | None = None,
) -> QuantizedArray:
    """Prepack *w* and return a ``QuantizedArray`` container."""
    axis_n = resolve_prepack_axis(axis=axis, transpose=transpose)
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    out = prepack_quantized_weights(
        w,
        group_size=group_size_n,
        bits=bits_n,
        mode=mode_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )
    if mode_n == "affine":
        data, scales, zeros = out
    else:
        data, scales = out
        zeros = None
    return QuantizedArray(
        data=data,
        scales=scales,
        zeros=zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=axis_n,
        runtime_config=runtime_config,
    )


__all__ = ("QuantizedArray", "prepack_quantized_array", "quantize_array")
