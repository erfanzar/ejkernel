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

from ejkernel.kernels._triton.flash_mla._interface import flash_mla_attention, flash_mla_attention_call

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "gpu",
    reason="Triton tests require GPU backend",
)


def test_flash_mla_raises_not_implemented():
    q = jnp.zeros((1, 2, 8, 16), dtype=jnp.float16)
    k = jnp.zeros((1, 2, 8, 16), dtype=jnp.float16)
    v = jnp.zeros((1, 2, 8, 16), dtype=jnp.float16)
    latent_key = jnp.zeros((16, 4), dtype=jnp.float16)
    latent_value = jnp.zeros((16, 4), dtype=jnp.float16)

    with pytest.raises(NotImplementedError):
        flash_mla_attention_call(q, k, v, latent_key=latent_key, latent_value=latent_value)

    with pytest.raises(NotImplementedError):
        flash_mla_attention(q, k, v, latent_key=latent_key, latent_value=latent_value)

