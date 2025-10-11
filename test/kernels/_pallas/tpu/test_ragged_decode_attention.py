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

from ejkernel.kernels._pallas.tpu.ragged_decode_attention import ragged_decode_attention as ragged_decode_tpu
from ejkernel.kernels._xla.ragged_decode_attention import ragged_decode_attention

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="Pallas TPU tests require TPU backend",
)


def _has_tpu():
    try:
        return len(jax.devices("tpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_tpu(), reason="TPU/Pallas required")


class TestRaggedDecodeAttentionTPU:
    def test_forward_shape_and_finite(self):
        B, S, HQ, HKV, D = 2, 512, 8, 2, 128
        kq, kk, kv = jax.random.split(jax.random.PRNGKey(0), 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([400, 480], dtype=jnp.int32)

        out = ragged_decode_tpu(q, k, v, sequence_start=starts, sequence_end=ends, softmax_scale=1.0, block_size=128)
        assert out.shape == (B, HQ, D)
        assert jnp.isfinite(out).all()

    def test_tail_block_and_gqa(self):
        B, S, HQ, HKV, D = 1, 1024, 32, 4, 128
        kq, kk, kv = jax.random.split(jax.random.PRNGKey(3), 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([5], dtype=jnp.int32)
        ends = jnp.array([407], dtype=jnp.int32)

        out_tpu = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=1.0, block_size=128)
        out_ref = ragged_decode_attention(q, k, v, starts, ends, softmax_scale=1.0)
        assert jnp.allclose(out_tpu, out_ref, rtol=0, atol=0.125)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
