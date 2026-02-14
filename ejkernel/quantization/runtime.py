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

"""Runtime policy controls for quantization fast paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuantRuntimeConfig:
    """Optional runtime policy for quantize/dequantize internals.

    These flags do not change quantization semantics. They select between
    compatible implementation strategies.
    """

    enable_u4_u8_fastpath: bool = True
    enable_threshold_codebook: bool = True
    enable_parity_fallback: bool = False
    strict_shape_alignment: bool = True
    prefer_compute_dtype: Literal["bf16", "fp16", "fp32"] = "fp32"
    affine_metadata_dtype: Literal["input", "bf16", "fp16", "fp32"] = "input"
    dequant_output_dtype: Literal["compute", "bf16", "fp16", "fp32"] = "fp32"
    dequant_unpack_policy: Literal["auto", "fast", "generic"] = "auto"
    minifloat_decode_policy: Literal["auto", "table", "arith"] = "auto"

    @classmethod
    def fastest_for_backend(
        cls,
        *,
        backend: str | None = None,
        keep_fp32_output: bool = False,
    ) -> "QuantRuntimeConfig":
        """Return an aggressive throughput profile tuned for the current backend."""
        if backend is None:
            import jax

            backend = jax.default_backend()
        backend = str(backend).lower()

        if backend == "tpu":
            compute = "bf16"
            meta = "bf16"
        else:
            # GPU/MPS: fp16 paths are generally the fastest for dequant pipelines.
            compute = "fp16"
            meta = "fp16"

        return cls(
            enable_u4_u8_fastpath=True,
            enable_threshold_codebook=True,
            enable_parity_fallback=False,
            strict_shape_alignment=True,
            prefer_compute_dtype=compute,
            affine_metadata_dtype=meta,
            dequant_output_dtype="fp32" if keep_fp32_output else "compute",
            dequant_unpack_policy="auto",
        )


def resolve_runtime_config(config: QuantRuntimeConfig | None) -> QuantRuntimeConfig:
    """Return *config* or a backend-tuned fast config when omitted."""
    if config is None:
        return QuantRuntimeConfig.fastest_for_backend()
    return config


__all__ = ("QuantRuntimeConfig", "resolve_runtime_config")
