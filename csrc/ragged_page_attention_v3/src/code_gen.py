import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

DTYPE_MAP = {
    "fp16": "half",
    "bf16": "__nv_bfloat16",
    "fp32": "float",
}

SM = [80, 90, 100, 110, 120]
HEAD_DIMENSIONS = [32, 64, 96, 128, 192, 256]


def get_template() -> str:
    return """// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "rpa_v3_launch_template.h"

namespace rpa_v3 {{

template<>
void run_rpa_v3_update_kv_<{DTYPE}, {HEAD_DIM}>(RpaV3Params &params, cudaStream_t stream) {{
    run_rpa_v3_update_kv_hdim<{DTYPE}, {HEAD_DIM}>(params, stream);
}}

template<>
void run_rpa_v3_attention_<{DTYPE}, {HEAD_DIM}>(RpaV3Params &params, cudaStream_t stream) {{
    run_rpa_v3_attention_hdim<{DTYPE}, {HEAD_DIM}>(params, stream);
}}

}} // namespace rpa_v3
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int

    @property
    def template(self) -> str:
        return get_template().format(
            DTYPE=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
        )

    @property
    def filename(self) -> str:
        return f"rpa_v3_fwd_hdim{self.head_dim}_{self.dtype}_sm{self.sm}.cu"


def get_all_kernels():
    for dtype, head_dim, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, SM):
        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)


def main(output_dir: str | None) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="code_gen",
        description="Generate the ragged_page_attention_v3 kernel instantiations",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels (defaults to current directory)",
    )
    args = parser.parse_args()
    main(args.output_dir)
