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

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._triton.quantized_matmul import quantized_matmul as triton_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

triton_qmm_bwd = importlib.import_module("ejkernel.kernels._triton.quantized_matmul._triton_impl_bwd")

pytestmark = pytest.mark.skipif(jax.devices()[0].platform != "gpu", reason="Triton tests require GPU backend")


def _device_put_all(dev, *arrays):
    return [jax.device_put(arr, dev) for arr in arrays]


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_triton_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    out_triton = triton_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
        bits=bits,
    )

    out_triton = jax.block_until_ready(out_triton)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_triton, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_triton_grad_input_matches_xla(mode: str, bits: int):
    key = jax.random.PRNGKey(11 if mode == "affine" else 13)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    def _loss_triton(x_in):
        y = triton_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    def _loss_xla(x_in):
        y = xla_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    g_triton = jax.block_until_ready(jax.grad(_loss_triton)(x))
    g_xla = jax.block_until_ready(jax.grad(_loss_xla)(x))

    np.testing.assert_allclose(
        np.asarray(g_triton, dtype=np.float32),
        np.asarray(g_xla, dtype=np.float32),
        rtol=7e-2,
        atol=7e-2,
    )


@pytest.mark.parametrize(
    "mode,bits",
    [
        ("affine", 4),
        ("affine", 8),
        ("nf4", 4),
        ("mxfp4", 4),
        ("mxfp8", 8),
        ("nvfp4", 4),
        ("nvfp8", 8),
    ],
)
def test_quantized_matmul_triton_grad_input_same_kernel_path(monkeypatch: pytest.MonkeyPatch, mode: str, bits: int):
    key = jax.random.PRNGKey(17 if mode == "affine" else 19)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode, bits=bits)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    def _forbidden_dequant(*args, **kwargs):
        raise AssertionError(f"Unexpected dequant fallback in Triton grad path for mode={mode}.")

    monkeypatch.setattr(triton_qmm_bwd, "quantized_matmul_dequant_triton", _forbidden_dequant)

    def _loss(x_in):
        y = triton_quantized_matmul(
            x_in,
            w_q,
            scales,
            biases,
            transpose=False,
            mode=mode,
            bits=bits,
        )
        return jnp.mean(y)

    gx = jax.block_until_ready(jax.grad(_loss)(x))
    assert gx.shape == (m, k)
