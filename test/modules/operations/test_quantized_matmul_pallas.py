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

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import quantized_matmul
from ejkernel.modules.operations.quantized_matmul import QuantizedMatmulConfig
from ejkernel.quantization import prepack_quantized_weights

from ._utils import assert_allclose


def _has_tpu() -> bool:
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="Pallas TPU tests require TPU backend")


@pytest.mark.parametrize(
    "mode,bits",
    [("affine", 4), ("nf4", 4), ("mxfp8", 8), ("nvfp4", 4)],
)
def test_quantized_matmul_operation_pallas_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(7 if mode == "affine" else 9)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 128

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, zeros = packed
    else:
        w_q, scales = packed
        zeros = None

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="pallas",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode=mode,
        bits=bits,
        platform="xla",
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_pallas, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_operation_pallas_large_n_matches_xla(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EJKERNEL_QMM_TPU_PATH", "hybrid")
    key = jax.random.PRNGKey(101)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="pallas",
    )
    out_xla = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="xla",
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_xla = jax.block_until_ready(out_xla)
    assert_allclose(out_pallas, out_xla, atol=6e-2, rtol=6e-2)


def test_quantized_matmul_operation_pallas_strict_fuse_repairs_illegal_block_n():
    key = jax.random.PRNGKey(909)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 32, 128, 256

    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.bfloat16)
    w_q, scales, zeros = prepack_quantized_weights(w, mode="affine", bits=4, group_size=64)

    # block_n=128 is illegal for packed forward when n=256 (4-bit packed words).
    illegal_cfg = QuantizedMatmulConfig(
        block_m=128,
        block_n=128,
        block_k=128,
        tpu_path="packed",
        platform="pallas",
        backend="tpu",
    )

    out_pallas = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="pallas",
        strict_fuse=True,
        tpu_path="packed",
        cfg=illegal_cfg,
    )
    out_ref = quantized_matmul(
        x,
        w_q,
        scales,
        zeros,
        transpose=False,
        mode="affine",
        bits=4,
        group_size=64,
        platform="xla",
        fuse=False,
    )

    out_pallas = jax.block_until_ready(out_pallas)
    out_ref = jax.block_until_ready(out_ref)
    assert_allclose(out_pallas, out_ref, atol=2e-1, rtol=8e-2)
