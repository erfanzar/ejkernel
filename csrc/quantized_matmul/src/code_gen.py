"""Generate quantized matmul dequantization instantiation stubs.

This script emits per-mode CUDA wrapper files and a dispatch header.
It is safe to re-run and will only rewrite files when content changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP = {
    "f32": "float",
    "f16": "half",
    "bf16": "__nv_bfloat16",
}

DTYPE_SUFFIX = {
    "f32": "F32",
    "f16": "F16",
    "bf16": "BF16",
}

AFFINE_BITS = (4, 8)
AFFINE_DTYPES = ("f32", "f16", "bf16")

NF4_DTYPES = ("f32", "f16", "bf16")

MXFP_GROUP_SIZE = 32
NVFP_GROUP_SIZE = 16
GROUP_SIZES = list(range(8, 1025))

PRELUDE = """// Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
//
// Licensed under the Apache License, Version 2.0 (the \"License\");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an \"AS IS\" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// This file is auto-generated. See \"code_gen.py\".

"""

AFFINE_TEMPLATE = """#include \"qmm_dequant_kernels.h\"

void {func_name}({signature}) {{
  dequant_affine_int<{bits}, {ctype}, {ctype}><<<grid, block, 0, stream>>>(
      wq, scales, biases, out, K, N, n_words, group_size, n_groups);
}}

{gs_wrappers}
"""

NF4_TEMPLATE = """#include \"qmm_dequant_kernels.h\"

void {func_name}({signature}) {{
  dequant_nf4_int<4, {ctype}><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, group_size, n_groups);
}}

{gs_wrappers}
"""

MXFP_TEMPLATE = """#include \"qmm_dequant_kernels.h\"

void {func_name}({signature}) {{
  dequant_mxfp{bits}<{group_size}><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, n_groups);
}}
"""

NVFP_TEMPLATE = """#include \"qmm_dequant_kernels.h\"

void {func_name}({signature}) {{
  dequant_nvfp{bits}<{group_size}><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, n_groups);
}}
"""

DISPATCH_HEADER_PREAMBLE = """#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

"""


@dataclass(frozen=True)
class Kernel:
    kind: str
    bits: int | None
    dtype: str | None

    @property
    def ctype(self) -> str:
        if self.dtype is None:
            return ""
        return DTYPE_MAP[self.dtype]

    @property
    def func_name(self) -> str:
        if self.kind == "affine":
            return f"LaunchDequantAffineBits{self.bits}{DTYPE_SUFFIX[self.dtype]}"
        if self.kind == "nf4":
            return f"LaunchDequantNf4{DTYPE_SUFFIX[self.dtype]}"
        if self.kind == "mxfp4":
            return "LaunchDequantMxFp4"
        if self.kind == "mxfp8":
            return "LaunchDequantMxFp8"
        if self.kind == "nvfp4":
            return "LaunchDequantNvFp4"
        if self.kind == "nvfp8":
            return "LaunchDequantNvFp8"
        raise ValueError(f"Unknown kind: {self.kind}")

    @property
    def filename(self) -> str:
        if self.kind == "affine":
            return f"qmm_dequant_affine_bits{self.bits}_{self.dtype}.cu"
        if self.kind == "nf4":
            return f"qmm_dequant_nf4_{self.dtype}.cu"
        return f"qmm_dequant_{self.kind}.cu"

    def signature(self) -> str:
        if self.kind == "affine":
            return (
                f"const uint32_t *wq, const {self.ctype} *scales, "
                f"const {self.ctype} *biases, half *out, int64_t K, int64_t N, "
                f"int64_t n_words, int64_t group_size, int64_t n_groups, "
                f"dim3 grid, dim3 block, cudaStream_t stream"
            )
        if self.kind == "nf4":
            return (
                f"const uint32_t *wq, const {self.ctype} *scales, half *out, "
                f"int64_t K, int64_t N, int64_t n_words, int64_t group_size, "
                f"int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream"
            )
        return (
            "const uint32_t *wq, const uint8_t *scales, half *out, "
            "int64_t K, int64_t N, int64_t n_words, int64_t n_groups, "
            "dim3 grid, dim3 block, cudaStream_t stream"
        )

    def gs_func_name(self, group_size: int) -> str:
        if self.kind == "affine":
            return (
                f"LaunchDequantAffineBits{self.bits}"
                f"{DTYPE_SUFFIX[self.dtype]}Gs{group_size}"
            )
        if self.kind == "nf4":
            return f"LaunchDequantNf4{DTYPE_SUFFIX[self.dtype]}Gs{group_size}"
        raise ValueError(f"Group size specialization not supported: {self.kind}")

    def render(self) -> str:
        if self.kind == "affine":
            gs_wrappers = []
            for group_size in GROUP_SIZES:
                gs_wrappers.append(
                    "void {func_name}({signature}) {{\n"
                    "  (void)group_size;\n"
                    "  dequant_affine_int_gs<{bits}, {group_size}, {ctype}, {ctype}>"
                    "<<<grid, block, 0, stream>>>(wq, scales, biases, out, K, N, "
                    "n_words, n_groups);\n"
                    "}}\n".format(
                        func_name=self.gs_func_name(group_size),
                        signature=self.signature(),
                        bits=self.bits,
                        group_size=group_size,
                        ctype=self.ctype,
                    )
                )
            return AFFINE_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                bits=self.bits,
                ctype=self.ctype,
                gs_wrappers="".join(gs_wrappers),
            )
        if self.kind == "nf4":
            gs_wrappers = []
            for group_size in GROUP_SIZES:
                gs_wrappers.append(
                    "void {func_name}({signature}) {{\n"
                    "  (void)group_size;\n"
                    "  dequant_nf4_int_gs<4, {group_size}, {ctype}>"
                    "<<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, "
                    "n_groups);\n"
                    "}}\n".format(
                        func_name=self.gs_func_name(group_size),
                        signature=self.signature(),
                        group_size=group_size,
                        ctype=self.ctype,
                    )
                )
            return NF4_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                ctype=self.ctype,
                gs_wrappers="".join(gs_wrappers),
            )
        if self.kind == "mxfp4":
            return MXFP_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                bits=4,
                group_size=MXFP_GROUP_SIZE,
            )
        if self.kind == "mxfp8":
            return MXFP_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                bits=8,
                group_size=MXFP_GROUP_SIZE,
            )
        if self.kind == "nvfp4":
            return NVFP_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                bits=4,
                group_size=NVFP_GROUP_SIZE,
            )
        if self.kind == "nvfp8":
            return NVFP_TEMPLATE.format(
                func_name=self.func_name,
                signature=self.signature(),
                bits=8,
                group_size=NVFP_GROUP_SIZE,
            )
        raise ValueError(f"Unknown kind: {self.kind}")


def iter_kernels() -> list[Kernel]:
    kernels: list[Kernel] = []
    for bits in AFFINE_BITS:
        for dtype in AFFINE_DTYPES:
            kernels.append(Kernel(kind="affine", bits=bits, dtype=dtype))
    for dtype in NF4_DTYPES:
        kernels.append(Kernel(kind="nf4", bits=4, dtype=dtype))
    kernels.append(Kernel(kind="mxfp4", bits=4, dtype=None))
    kernels.append(Kernel(kind="mxfp8", bits=8, dtype=None))
    kernels.append(Kernel(kind="nvfp4", bits=4, dtype=None))
    kernels.append(Kernel(kind="nvfp8", bits=8, dtype=None))
    return kernels


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text() == content:
        return False
    path.write_text(content)
    return True


def _render_dispatch_header(kernels: list[Kernel]) -> str:
    lines = [PRELUDE, DISPATCH_HEADER_PREAMBLE]
    for kernel in kernels:
        lines.append(f"void {kernel.func_name}({kernel.signature()});\n")
        if kernel.kind in ("affine", "nf4"):
            for group_size in GROUP_SIZES:
                lines.append(
                    f"void {kernel.gs_func_name(group_size)}("
                    f"{kernel.signature()});\n"
                )
    lines.append("\n")
    lines.append("using DequantAffineF32Fn = void (*)(")
    lines.append(
        "const uint32_t *, const float *, const float *, half *, int64_t, int64_t,"
        " int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("using DequantAffineF16Fn = void (*)(")
    lines.append(
        "const uint32_t *, const half *, const half *, half *, int64_t, int64_t,"
        " int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("using DequantAffineBF16Fn = void (*)(")
    lines.append(
        "const uint32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *, half *,"
        " int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("using DequantNf4F32Fn = void (*)(")
    lines.append(
        "const uint32_t *, const float *, half *, int64_t, int64_t, int64_t,"
        " int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("using DequantNf4F16Fn = void (*)(")
    lines.append(
        "const uint32_t *, const half *, half *, int64_t, int64_t, int64_t,"
        " int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("using DequantNf4BF16Fn = void (*)(")
    lines.append(
        "const uint32_t *, const __nv_bfloat16 *, half *, int64_t, int64_t,"
        " int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);\n"
    )
    lines.append("\n")

    for kernel in kernels:
        if kernel.kind not in ("affine", "nf4"):
            continue
        if kernel.kind == "affine":
            if kernel.dtype == "f32":
                ret = "DequantAffineF32Fn"
            elif kernel.dtype == "f16":
                ret = "DequantAffineF16Fn"
            else:
                ret = "DequantAffineBF16Fn"
        else:
            if kernel.dtype == "f32":
                ret = "DequantNf4F32Fn"
            elif kernel.dtype == "f16":
                ret = "DequantNf4F16Fn"
            else:
                ret = "DequantNf4BF16Fn"
        lines.append(
            f"inline {ret} Resolve{kernel.func_name}(int64_t group_size) {{\n"
        )
        lines.append("  switch (group_size) {\n")
        for group_size in GROUP_SIZES:
            lines.append(
                f"  case {group_size}: return &{kernel.gs_func_name(group_size)};\n"
            )
        lines.append(f"  default: return &{kernel.func_name};\n")
        lines.append("  }\n")
        lines.append("}\n\n")
    return "".join(lines)


def generate(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    kernels = iter_kernels()
    count = 0
    for kernel in kernels:
        content = PRELUDE + kernel.render()
        if _write_if_changed(output_dir / kernel.filename, content):
            count += 1
    header_content = _render_dispatch_header(kernels)
    if _write_if_changed(output_dir / "qmm_dequant_dispatch.h", header_content):
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate quantized matmul dequantization instantiation stubs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output directory for generated files (defaults to this script's directory).",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    generated = generate(out_dir)
    print(f"Generated {generated} files in {out_dir}")


if __name__ == "__main__":
    main()
