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
        """Test basic forward pass shape and finiteness."""
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
        """Test with non-aligned block sizes and GQA."""
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

    def test_variable_sequence_lengths(self):
        """Test with varying sequence lengths."""
        B, S, HQ, HKV, D = 3, 1024, 16, 4, 128
        key = jax.random.PRNGKey(42)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        # Different lengths: short, medium, long
        starts = jnp.array([0, 50, 200], dtype=jnp.int32)
        ends = jnp.array([128, 300, 800], dtype=jnp.int32)

        out_tpu = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=1.0, block_size=256)
        out_ref = ragged_decode_attention(q, k, v, starts, ends, softmax_scale=1.0)

        assert out_tpu.shape == (B, HQ, D)
        assert jnp.isfinite(out_tpu).all()
        assert jnp.allclose(out_tpu, out_ref, rtol=0, atol=0.15)

    def test_different_block_sizes(self):
        """Test with different block sizes."""
        B, S, HQ, HKV, D = 2, 512, 8, 2, 64
        key = jax.random.PRNGKey(123)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([256, 400], dtype=jnp.int32)

        for block_size in [128, 256]:
            out = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=1.0, block_size=block_size)
            assert out.shape == (B, HQ, D)
            assert jnp.isfinite(out).all()

    def test_single_batch_element(self):
        """Test with single batch element."""
        B, S, HQ, HKV, D = 1, 256, 8, 2, 128
        key = jax.random.PRNGKey(456)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([10], dtype=jnp.int32)
        ends = jnp.array([200], dtype=jnp.int32)

        out_tpu = ragged_decode_tpu(q, k, v, starts, ends, block_size=128)
        out_ref = ragged_decode_attention(q, k, v, starts, ends)

        assert out_tpu.shape == (B, HQ, D)
        assert jnp.allclose(out_tpu, out_ref, rtol=0, atol=0.125)

    def test_mqa_vs_gqa(self):
        """Test Multi-Query Attention vs Grouped Query Attention."""
        B, S, D = 2, 512, 128
        key = jax.random.PRNGKey(789)

        # MQA: 1 KV head, 8 Q heads
        HQ_mqa, HKV_mqa = 8, 1
        kq, kk, kv = jax.random.split(key, 3)
        q_mqa = jax.random.normal(kq, (B, HQ_mqa, D), dtype=jnp.float32)
        k_mqa = jax.random.normal(kk, (B, S, HKV_mqa, D), dtype=jnp.float32)
        v_mqa = jax.random.normal(kv, (B, S, HKV_mqa, D), dtype=jnp.float32)

        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([256, 400], dtype=jnp.int32)

        out_mqa = ragged_decode_tpu(q_mqa, k_mqa, v_mqa, starts, ends, block_size=128)
        assert out_mqa.shape == (B, HQ_mqa, D)
        assert jnp.isfinite(out_mqa).all()

        # GQA: 4 KV heads, 16 Q heads
        HQ_gqa, HKV_gqa = 16, 4
        key2 = jax.random.PRNGKey(790)
        kq2, kk2, kv2 = jax.random.split(key2, 3)
        q_gqa = jax.random.normal(kq2, (B, HQ_gqa, D), dtype=jnp.float32)
        k_gqa = jax.random.normal(kk2, (B, S, HKV_gqa, D), dtype=jnp.float32)
        v_gqa = jax.random.normal(kv2, (B, S, HKV_gqa, D), dtype=jnp.float32)

        out_gqa = ragged_decode_tpu(q_gqa, k_gqa, v_gqa, starts, ends, block_size=128)
        assert out_gqa.shape == (B, HQ_gqa, D)
        assert jnp.isfinite(out_gqa).all()

    def test_dtypes(self):
        """Test with different dtypes."""
        B, S, HQ, HKV, D = 2, 512, 8, 2, 128
        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([256, 400], dtype=jnp.int32)

        for dtype in [jnp.float32, jnp.bfloat16]:
            key = jax.random.PRNGKey(999)
            kq, kk, kv = jax.random.split(key, 3)

            q = jax.random.normal(kq, (B, HQ, D), dtype=dtype)
            k = jax.random.normal(kk, (B, S, HKV, D), dtype=dtype)
            v = jax.random.normal(kv, (B, S, HKV, D), dtype=dtype)

            out = ragged_decode_tpu(q, k, v, starts, ends, block_size=128)
            assert out.shape == (B, HQ, D)
            assert out.dtype == dtype
            assert jnp.isfinite(out).all()

    def test_different_head_dimensions(self):
        """Test with different head dimensions."""
        B, S, HQ, HKV = 2, 512, 8, 2
        starts = jnp.array([0, 100], dtype=jnp.int32)
        ends = jnp.array([256, 400], dtype=jnp.int32)

        for head_dim in [64, 128, 256]:
            key = jax.random.PRNGKey(1000 + head_dim)
            kq, kk, kv = jax.random.split(key, 3)

            q = jax.random.normal(kq, (B, HQ, head_dim), dtype=jnp.float32)
            k = jax.random.normal(kk, (B, S, HKV, head_dim), dtype=jnp.float32)
            v = jax.random.normal(kv, (B, S, HKV, head_dim), dtype=jnp.float32)

            out = ragged_decode_tpu(q, k, v, starts, ends, block_size=128)
            assert out.shape == (B, HQ, head_dim)
            assert jnp.isfinite(out).all()

    def test_correctness_vs_reference(self):
        """Test numerical correctness against XLA reference."""
        B, S, HQ, HKV, D = 2, 512, 16, 4, 128
        key = jax.random.PRNGKey(2025)
        kq, kk, kv = jax.random.split(key, 3)

        q = jax.random.normal(kq, (B, HQ, D), dtype=jnp.float32)
        k = jax.random.normal(kk, (B, S, HKV, D), dtype=jnp.float32)
        v = jax.random.normal(kv, (B, S, HKV, D), dtype=jnp.float32)

        starts = jnp.array([10, 150], dtype=jnp.int32)
        ends = jnp.array([300, 450], dtype=jnp.int32)

        out_tpu = ragged_decode_tpu(q, k, v, starts, ends, softmax_scale=1.0, block_size=128)
        out_ref = ragged_decode_attention(q, k, v, starts, ends, softmax_scale=1.0)

        # Check shape
        assert out_tpu.shape == out_ref.shape

        # Check numerical accuracy
        max_diff = jnp.max(jnp.abs(out_tpu - out_ref))
        assert max_diff < 0.15, f"Max difference {max_diff} exceeds tolerance"

        # Check relative error
        assert jnp.allclose(out_tpu, out_ref, rtol=0, atol=0.125)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
