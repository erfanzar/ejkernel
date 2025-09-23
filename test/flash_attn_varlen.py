# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import math
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import torch
import triton
import triton.testing
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
from jax import numpy as jnp

from ejkernel import flash_attn_varlen

to_jnp = lambda x: jnp.asarray(x.detach().cpu().numpy())  # noqa
to_torch = lambda x: torch.from_numpy(np.asarray(x))  # noqa


@pytest.mark.parametrize("causal", [False, True])
def test_flash_attention_varlen_comparison(causal):
    """
    Tests the output of ejgpu.flash_attention_varlen against flash_attn_varlen_func.
    Uses the exact setup and comparison logic from the original script.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(42)
    q = torch.randn(
        1000,
        32,
        128,
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    k = torch.randn(
        1000,
        16,
        128,
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    v = torch.randn(
        1000,
        16,
        128,
        dtype=torch.float16,
        device="cuda",
        requires_grad=True,
    )
    cu_seqlens_q = torch.Tensor([0, 100, 384, 1000]).cuda().to(torch.int32)
    cu_seqlens_k = torch.Tensor([0, 100, 384, 1000]).cuda().to(torch.int32)
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max()
    print(f"\nTesting causal={causal}: max_seqlen_q={max_seqlen_q}, max_seqlen_k={max_seqlen_k}")

    o = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q.item(),
        max_seqlen_k=max_seqlen_k.item(),
        causal=causal,
        softmax_scale=0.1,
    )

    torch.manual_seed(42)
    q1 = q.clone().detach().requires_grad_()
    k1 = k.clone().detach().requires_grad_()
    v1 = v.clone().detach().requires_grad_()
    cu_seqlens_q1 = cu_seqlens_q.clone().detach()
    cu_seqlens_k1 = cu_seqlens_k.clone().detach()
    max_seqlen_q1 = (cu_seqlens_q1[1:] - cu_seqlens_q1[:-1]).max()
    max_seqlen_k1 = (cu_seqlens_k1[1:] - cu_seqlens_k1[:-1]).max()
    o1_jax = flash_attn_varlen(
        q=to_jnp(q1),
        k=to_jnp(k1),
        v=to_jnp(v1),
        cu_seqlens_q=to_jnp(cu_seqlens_q1),
        cu_seqlens_k=to_jnp(cu_seqlens_k1),
        max_seqlens_q=max_seqlen_q1.item(),
        max_seqlens_k=max_seqlen_k1.item(),
        causal=causal,
        layout="thd",
        sm_scale=0.1,
    )
    o1 = to_torch(o1_jax).cuda()
    print(f"Comparing outputs for causal={causal}")
    allclose = torch.allclose(o, o1, atol=0.125, rtol=0)
    max_err = (o - o1).abs().max().item()
    print(f"Output Check (causal={causal}): Same Output = {allclose}, Max Error = {max_err}")

    assert allclose, f"Outputs differ significantly for causal={causal}. Max Error: {max_err}"


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[1024 * 2**i for i in range(1, 7)],
        line_arg="provider",
        line_vals=["flash", "ejgpu"],
        line_names=["Flash", "EJgpu"],
        styles=[("green", "solid"), ("blue", "dashdot")],
        ylabel="ms",
        plot_name="flash_attention_varlen_forward",
        args={"H": 64, "D": 128},
    )
)
def benchmark(N, H, D, provider):
    q = torch.randn((N, H, D), device="cuda", dtype=torch.float16)
    k = torch.randn((N, H // 16, D), device="cuda", dtype=torch.float16)
    v = torch.randn((N, H // 16, D), device="cuda", dtype=torch.float16)
    cu_seqlens = torch.tensor([0, N], device="cuda", dtype=torch.int32)
    sm_scale = 1 / math.sqrt(D)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "flash":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: _flash_attn_varlen_forward(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                N,
                N,
                dropout_p=0.0,
                causal=False,
                softmax_scale=sm_scale,
            ),
            quantiles=quantiles,
        )
    if provider == "ejgpu":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: flash_attn_varlen(
                q=to_jnp(q),
                k=to_jnp(k),
                v=to_jnp(v),
                cu_seqlens_q=to_jnp(cu_seqlens),
                cu_seqlens_k=to_jnp(cu_seqlens),
                layout="thd",
                max_seqlens_q=N,
                max_seqlens_k=N,
                causal=False,
                sm_scale=sm_scale,
            ),
            quantiles=quantiles,
        )
    return ms


def run_benchmarks(save_path=".", print_data=True):
    """Runs the benchmark function."""
    print("\n" + "=" * 20 + " RUNNING BENCHMARKS " + "=" * 20)
    if not torch.cuda.is_available():
        print("Skipping benchmarks: CUDA not available")
        return
    benchmark.run(print_data=print_data, save_path=save_path)


if __name__ == "__main__":
    result = pytest.main(["-v", __file__])
    run_benchmarks(save_path=".")
