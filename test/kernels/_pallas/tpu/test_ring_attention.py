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


"""Tests for Pallas TPU ring attention implementation."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec

from ejkernel.kernels import pallas
from ejkernel.utils import numeric_gen

pytestmark = pytest.mark.skipif(
    jax.devices()[0].platform != "tpu",
    reason="Pallas TPU tests require TPU backend",
)


class TestRingAttentionPallasFwd:
    """Test forward pass of Pallas TPU ring attention."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert out.dtype == jnp.float32
        assert not jnp.any(jnp.isnan(out))
        assert not jnp.any(jnp.isinf(out))

    def test_softmax_aux_1d(self):
        """Test with 1D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_softmax_aux_2d(self):
        """Test with 2D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_logit_soft_cap(self):
        """Test with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            logit_soft_cap=30.0,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_symmetric(self):
        """Test with symmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_asymmetric(self):
        """Test with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=(32, 96),
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_small(self):
        """Test with very small sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=0,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_large(self):
        """Test with sliding window larger than sequence."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=512,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_with_softmax_aux(self):
        """Test sliding window combined with softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            softmax_aux=softmax_aux,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_sliding_window_with_causal(self):
        """Test sliding window with causal masking."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            causal_block_size=32,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_attention_sinks(self):
        """Test with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=64,
            attention_sink_size=8,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_causal_mask(self):
        """Test with causal masking."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            causal_block_size=64,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_combined_features(self):
        """Test with all features combined."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logit_soft_cap=30.0,
            sliding_window=(32, 96),
            attention_sink_size=8,
            causal_block_size=64,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))


class TestRingAttentionPallasBwd:
    """Test backward pass (gradients) of Pallas TPU ring attention."""

    def test_basic_gradient(self):
        """Test basic gradient computation."""
        batch, seq_len, num_heads, head_dim = 2, 128, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert grad_q.shape == q.shape
        assert grad_k.shape == k.shape
        assert grad_v.shape == v.shape
        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))
        assert not jnp.isnan(loss)
        assert loss > 0

    def test_gradient_with_softmax_aux(self):
        """Test gradient with softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 2
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                softmax_aux=softmax_aux,
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_logit_soft_cap(self):
        """Test gradient with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                logit_soft_cap=30.0,
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_sliding_window_symmetric(self):
        """Test gradient with symmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                sliding_window=64,
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_with_sliding_window_asymmetric(self):
        """Test gradient with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                sliding_window=(32, 96),
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        _loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))

    def test_gradient_combined_features(self):
        """Test gradient with combined features."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        num_sinks = 2
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        def loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q,
                k,
                v,
                softmax_aux=softmax_aux,
                logit_soft_cap=30.0,
                sliding_window=64,
                attention_sink_size=4,
                query_chunk_size=128,
                key_chunk_size=128,
            )
            return jnp.mean(out**2)

        loss, grads = jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        grad_q, grad_k, grad_v = grads

        assert not jnp.any(jnp.isnan(grad_q))
        assert not jnp.any(jnp.isnan(grad_k))
        assert not jnp.any(jnp.isnan(grad_v))
        assert loss > 0


class TestRingAttentionPallasDistributed:
    """Test distributed execution of Pallas TPU ring attention."""

    def test_distributed_shard_map(self):
        """Test distributed execution with shard_map - this is how Pallas should be used on TPU."""
        pytest.importorskip("eformer.escale")
        from eformer.escale import create_mesh

        try:
            mesh = create_mesh()
        except RuntimeError:
            pytest.skip("Mesh creation failed")

        batch, seq_len, num_heads, head_dim = 4, 1024, 32, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        softmax_aux = numeric_gen(num_heads, dtype="f4") * -1.0

        out = shard_map(
            partial(pallas.tpu.ring_attention, axis_name="sp", query_chunk_size=128, key_chunk_size=128),
            in_specs=(
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                None,
                None,
                None,
                PartitionSpec("tp"),
            ),
            out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            mesh=mesh,
            check_vma=False,
        )(q, k, v, None, None, None, softmax_aux)

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))


class TestRingAttentionPallasEdgeCases:
    """Test edge cases for Pallas TPU ring attention."""

    def test_small_sequence(self):
        """Test with small sequence length."""
        batch, seq_len, num_heads, head_dim = 1, 64, 4, 32
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=32,
            key_chunk_size=32,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_single_head(self):
        """Test with single attention head."""
        batch, seq_len, num_heads, head_dim = 2, 128, 1, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_different_chunk_sizes(self):
        """Test with different query and key chunk sizes."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=256,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_large_head_dim(self):
        """Test with large head dimension."""
        batch, seq_len, num_heads, head_dim = 2, 256, 4, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        out = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        assert out.shape == (batch, seq_len, num_heads, head_dim)
        assert not jnp.any(jnp.isnan(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
