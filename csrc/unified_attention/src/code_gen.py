"""Generate unified attention kernel instantiation stubs.

This script emits per-head-dim CUDA instantiation files to speed up compilation.
It is safe to re-run and will only rewrite files when content changes.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP = {
    "fp16": "half",
    "bf16": "__nv_bfloat16",
}

DEFAULT_SMS = (80, 90, 100, 110, 120)
HEAD_DIMS = (32, 64, 96, 128, 192, 256)

PRELUDE = """// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "ua_launch_template.h"

namespace ua {{

template<>
void run_unified_attention_<{DTYPE}, {HEAD_DIM}>(UaParams &params, cudaStream_t stream) {{
    run_unified_attention_hdim<{DTYPE}, {HEAD_DIM}>(params, stream);
}}

}}  // namespace ua
"""


@dataclass(frozen=True)
class Kernel:
    sm: int
    dtype: str
    head_dim: int

    def render(self) -> str:
        return PRELUDE.format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
        )

    @property
    def filename(self) -> str:
        return f"ua_fwd_hdim{self.head_dim}_{self.dtype}_sm{self.sm}.cu"


def iter_kernels(sms: tuple[int, ...]) -> list[Kernel]:
    return [
        Kernel(sm=sm, dtype=dtype, head_dim=head_dim)
        for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMS, sms)
    ]


def _write_if_changed(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def generate(output_dir: Path, sms: tuple[int, ...]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for kernel in iter_kernels(sms):
        _write_if_changed(output_dir / kernel.filename, kernel.render())
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate unified attention kernel instantiation stubs.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Output directory for generated .cu files (defaults to this script's directory).",
    )
    parser.add_argument(
        "--sms",
        default="80,90,100,110,120",
        help="Comma-separated list of SM versions to emit (e.g., 80,90,100,110,120).",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    sms = tuple(int(item.strip()) for item in args.sms.split(",") if item.strip())
    if not sms:
        sms = DEFAULT_SMS
    generated = generate(out_dir, sms)
    print(f"Generated {generated} files for SMs={sms} in {out_dir}")


if __name__ == "__main__":
    main()
