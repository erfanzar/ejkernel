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

import shutil

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ejkernel.kernels._cuda.quantized_matmul import quantized_matmul as cuda_quantized_matmul
from ejkernel.kernels._xla.quantized_matmul import quantized_matmul as xla_quantized_matmul
from ejkernel.quantization import prepack_quantized_weights

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu" or shutil.which("nvcc") is None,
    reason="CUDA quantized_matmul tests require GPU backend and nvcc",
)

try:
    from ejkernel.kernels._cuda.quantized_matmul._build import build_cuda_lib

    build_cuda_lib()
except RuntimeError as exc:
    pytest.skip(f"CUDA quantized_matmul build failed: {exc}", allow_module_level=True)


def _device_put_all(dev, *arrays):
    return [jax.device_put(arr, dev) for arr in arrays]


@pytest.mark.parametrize("mode", ["affine", "nf4", "mxfp4"])
def test_quantized_matmul_cuda_matches_xla(mode: str):
    key = jax.random.PRNGKey(0 if mode == "affine" else 1)
    kx, kw = jax.random.split(key, 2)
    m, k, n = 16, 64, 64

    x = jax.random.normal(kx, (m, k), dtype=jnp.float16)
    w = jax.random.normal(kw, (n, k), dtype=jnp.float16)

    packed = prepack_quantized_weights(w, mode=mode)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None

    dev = jax.devices("gpu")[0]
    x, w_q, scales = _device_put_all(dev, x, w_q, scales)
    if biases is not None:
        biases = jax.device_put(biases, dev)

    out_cuda = cuda_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
    )
    out_xla = xla_quantized_matmul(
        x,
        w_q,
        scales,
        biases,
        transpose=False,
        mode=mode,
    )

    out_cuda = jax.block_until_ready(out_cuda)
    out_xla = jax.block_until_ready(out_xla)

    np.testing.assert_allclose(
        np.asarray(out_cuda, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=6e-2,
        atol=6e-2,
    )
