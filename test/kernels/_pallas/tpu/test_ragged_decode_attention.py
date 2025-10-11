# Copyright 2025 ...
# SPDX-License-Identifier: Apache-2.0

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

    # @pytest.mark.parametrize(
    #     "sliding_window,logit_soft_cap",
    #     [
    #         (None, None),
    #         # ((128, 0), None),
    #         # ((256, 64), None),
    #         # (None, 10.0),
    #         # ((128, 64), 20.0),
    #     ],
    # )
    # def test_against_reference(self, sliding_window, logit_soft_cap):
    #     B, S, HQ, HKV, D = 2, 384, 16, 2, 128
    #     kq, kk, kv = jax.random.split(jax.random.PRNGKey(1), 3)

    #     q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
    #     k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
    #     v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

    #     starts = jnp.array([0, 50], dtype=jnp.int32)
    #     ends = jnp.array([350, 320], dtype=jnp.int32)

    #     out_tpu = ragged_decode_tpu(
    #         q,
    #         k,
    #         v,
    #         sequence_start=starts,
    #         sequence_end=ends,
    #         softmax_scale=1.0,
    #         block_size=128,
    #         sliding_window=sliding_window,
    #         logit_soft_cap=logit_soft_cap,
    #         softmax_aux=None,
    #     )
    #     out_ref = ragged_decode_attention(
    #         q,
    #         k,
    #         v,
    #         sequence_start=starts,
    #         sequence_end=ends,
    #         softmax_scale=1.0,
    #         sliding_window=sliding_window,
    #         logit_soft_cap=logit_soft_cap,
    #         softmax_aux=None,
    #     )

    #     assert out_tpu.shape == out_ref.shape
    #     assert jnp.allclose(out_tpu, out_ref, rtol=0, atol=1e-2)

    # def test_with_sinks(self):
    #     B, S, HQ, HKV, D = 2, 256, 16, 4, 128
    #     kq, kk, kv, _ka = jax.random.split(jax.random.PRNGKey(2), 4)

    #     q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
    #     k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
    #     v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

    #     starts = jnp.array([10, 0], dtype=jnp.int32)
    #     ends = jnp.array([200, 220], dtype=jnp.int32)

    #     NS = 3
    #     aux_shared = jnp.linspace(-0.5, 0.5, NS, dtype=jnp.float32)

    #     out_tpu = ragged_decode_tpu(
    #         q,
    #         k,
    #         v,
    #         starts,
    #         ends,
    #         softmax_scale=0.75,
    #         block_size=64,
    #         sliding_window=(128, 16),
    #         logit_soft_cap=25.0,
    #         softmax_aux=aux_shared,
    #     )
    #     out_ref = ragged_decode_attention(
    #         q,
    #         k,
    #         v,
    #         starts,
    #         ends,
    #         softmax_scale=0.75,
    #         sliding_window=(128, 16),
    #         logit_soft_cap=25.0,
    #         softmax_aux=aux_shared,
    #     )
    #     assert jnp.allclose(out_tpu, out_ref, rtol=2e-4, atol=2e-5)

    def test_tail_block_and_gqa(self):
        # Tail block: S not divisible by block_size; GQA heads grouping
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
