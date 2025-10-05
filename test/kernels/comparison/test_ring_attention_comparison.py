#!/usr/bin/env python3
"""Comparison test for ring attention implementations across backends."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec

from ejkernel.kernels import pallas, xla
from ejkernel.utils import numeric_gen


class TestRingAttentionComparison:
    """Test ring attention implementations across XLA and Pallas backends."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Create mesh for distributed testing (if available)
        try:
            from eformer.escale import create_mesh

            self.mesh = create_mesh()
            self.has_mesh = True
        except (ImportError, RuntimeError):
            self.mesh = None
            self.has_mesh = False

        # Set tolerances for comparison
        self.rtol = 1e-3
        self.atol = 1e-3

    def test_basic_attention_comparison(self):
        """Test basic ring attention between XLA and Pallas implementations."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Basic ring attention outputs don't match between XLA and Pallas",
        )

    def test_with_softmax_aux_1d(self):
        """Test ring attention with 1D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create 1D softmax_aux
        num_sinks = 4
        softmax_aux_1d = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux_1d,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux_1d,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with 1D softmax_aux outputs don't match",
        )

    def test_with_softmax_aux_2d(self):
        """Test ring attention with 2D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create 2D softmax_aux
        num_sinks = 4
        softmax_aux_2d = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux_2d,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux_2d,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with 2D softmax_aux outputs don't match",
        )

    def test_with_logit_soft_cap(self):
        """Test ring attention with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        logit_soft_cap = 30.0

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            logit_soft_cap=logit_soft_cap,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            logit_soft_cap=logit_soft_cap,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with logit_soft_cap outputs don't match",
        )

    def test_with_sliding_window_symmetric(self):
        """Test ring attention with symmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Test symmetric window
        sliding_window = 128

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with symmetric sliding_window outputs don't match",
        )

    def test_with_sliding_window_asymmetric(self):
        """Test ring attention with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Test asymmetric window (left_window, right_window)
        sliding_window = (64, 192)

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with asymmetric sliding_window outputs don't match",
        )

    def test_with_sliding_window_small(self):
        """Test ring attention with very small sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Test very small window (only attend to self and immediate neighbors)
        sliding_window = 1

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with small sliding_window outputs don't match",
        )

    def test_with_sliding_window_and_softmax_aux(self):
        """Test ring attention with sliding window and softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        sliding_window = 128
        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            softmax_aux=softmax_aux,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with sliding_window and softmax_aux outputs don't match",
        )

    def test_with_attention_sinks(self):
        """Test ring attention with attention sinks."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        sliding_window = 128
        attention_sink_size = 8

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            attention_sink_size=attention_sink_size,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            sliding_window=sliding_window,
            attention_sink_size=attention_sink_size,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with attention_sinks outputs don't match",
        )

    def test_combined_features(self):
        """Test ring attention with all features combined."""
        batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create softmax_aux
        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        # Combined features
        logit_soft_cap = 30.0
        sliding_window = (64, 128)  # Asymmetric
        attention_sink_size = 8
        causal_block_size = 64

        # XLA implementation
        out_xla = xla.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logit_soft_cap=logit_soft_cap,
            sliding_window=sliding_window,
            attention_sink_size=attention_sink_size,
            causal_block_size=causal_block_size,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Pallas implementation
        out_pallas = pallas.tpu.ring_attention(
            q,
            k,
            v,
            softmax_aux=softmax_aux,
            logit_soft_cap=logit_soft_cap,
            sliding_window=sliding_window,
            attention_sink_size=attention_sink_size,
            causal_block_size=causal_block_size,
            query_chunk_size=128,
            key_chunk_size=128,
        )

        # Compare outputs
        np.testing.assert_allclose(
            out_xla,
            out_pallas,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Ring attention with combined features outputs don't match",
        )

    @pytest.mark.skipif(not jax.devices()[0].platform == "tpu", reason="Requires TPU")
    def test_distributed_with_shard_map(self):
        """Test ring attention with distributed execution using shard_map for Pallas."""
        if not self.has_mesh:
            pytest.skip("Mesh creation not available")

        batch, seq_len, num_heads, head_dim = 4, 1024, 32, 128
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create softmax_aux
        softmax_aux = numeric_gen(num_heads, dtype="f4") * -1.0

        # Pallas implementation with shard_map (as it should be used)
        out_pallas = shard_map(
            partial(pallas.tpu.ring_attention, axis_name="sp"),
            in_specs=(
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                None,  # attn_bias
                None,  # q_segment_ids
                None,  # kv_segment_ids
                PartitionSpec("tp"),  # softmax_aux
            ),
            out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            mesh=self.mesh,
            check_vma=False,
        )(q, k, v, None, None, None, softmax_aux)

        # XLA implementation with shard_map for comparison
        out_xla = shard_map(
            partial(xla.ring_attention, axis_name="sp"),
            in_specs=(
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
                None,  # attn_bias
                None,  # q_segment_ids
                None,  # kv_segment_ids
                PartitionSpec("tp"),  # softmax_aux
            ),
            out_specs=PartitionSpec(("dp", "fsdp"), "sp", "tp", None),
            mesh=self.mesh,
            check_vma=False,
        )(q, k, v, None, None, None, softmax_aux)

        # Compare outputs
        assert out_pallas.shape == out_xla.shape

        # Full comparison with tolerance
        np.testing.assert_allclose(
            out_pallas,
            out_xla,
            rtol=self.rtol,
            atol=self.atol,
            err_msg="Distributed ring attention outputs don't match between XLA and Pallas",
        )


class TestRingAttentionGradientComparison:
    """Test gradient computation comparison between XLA and Pallas implementations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        # Set tolerances for gradient comparison
        self.grad_rtol = 5e-3  # Slightly higher tolerance for gradients
        self.grad_atol = 5e-3

    def test_basic_gradient_comparison(self):
        """Test basic gradient computation between XLA and Pallas."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(q, k, v, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(q, k, v, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Compute gradients for XLA
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        # Compute gradients for Pallas
        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla,
            dq_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Query gradients don't match between XLA and Pallas",
        )
        np.testing.assert_allclose(
            dk_xla,
            dk_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Key gradients don't match between XLA and Pallas",
        )
        np.testing.assert_allclose(
            dv_xla,
            dv_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Value gradients don't match between XLA and Pallas",
        )

    def test_gradient_with_softmax_aux_1d(self):
        """Test gradient computation with 1D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create 1D softmax_aux
        num_sinks = 4
        softmax_aux = jnp.ones((num_sinks,), dtype=jnp.float32) * -2.0

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(q, k, v, softmax_aux=softmax_aux, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(q, k, v, softmax_aux=softmax_aux, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla,
            dq_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Query gradients with 1D softmax_aux don't match",
        )
        np.testing.assert_allclose(
            dk_xla,
            dk_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Key gradients with 1D softmax_aux don't match",
        )
        np.testing.assert_allclose(
            dv_xla,
            dv_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Value gradients with 1D softmax_aux don't match",
        )

    def test_gradient_with_softmax_aux_2d(self):
        """Test gradient computation with 2D softmax_aux."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create 2D softmax_aux
        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(q, k, v, softmax_aux=softmax_aux, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(q, k, v, softmax_aux=softmax_aux, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla,
            dq_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Query gradients with 2D softmax_aux don't match",
        )
        np.testing.assert_allclose(
            dk_xla,
            dk_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Key gradients with 2D softmax_aux don't match",
        )
        np.testing.assert_allclose(
            dv_xla,
            dv_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Value gradients with 2D softmax_aux don't match",
        )

    def test_gradient_with_logit_soft_cap(self):
        """Test gradient computation with logit soft cap."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        logit_soft_cap = 30.0

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(q, k, v, logit_soft_cap=logit_soft_cap, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q, k, v, logit_soft_cap=logit_soft_cap, query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out**2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla,
            dq_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Query gradients with logit_soft_cap don't match",
        )
        np.testing.assert_allclose(
            dk_xla,
            dk_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Key gradients with logit_soft_cap don't match",
        )
        np.testing.assert_allclose(
            dv_xla,
            dv_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Value gradients with logit_soft_cap don't match",
        )

    def test_gradient_with_combined_features(self):
        """Test gradient computation with all features combined."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create softmax_aux
        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0
        logit_soft_cap = 30.0

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(
                q, k, v, softmax_aux=softmax_aux, logit_soft_cap=logit_soft_cap, query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out**2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q, k, v, softmax_aux=softmax_aux, logit_soft_cap=logit_soft_cap, query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out**2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla,
            dq_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Query gradients with combined features don't match",
        )
        np.testing.assert_allclose(
            dk_xla,
            dk_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Key gradients with combined features don't match",
        )
        np.testing.assert_allclose(
            dv_xla,
            dv_pallas,
            rtol=self.grad_rtol,
            atol=self.grad_atol,
            err_msg="Value gradients with combined features don't match",
        )

    def test_gradient_with_sliding_window(self):
        """Test gradient computation with sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        sliding_window = 64

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(
                q, k, v, sliding_window=sliding_window,
                query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out ** 2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q, k, v, sliding_window=sliding_window,
                query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out ** 2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla, dq_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Query gradients with sliding_window don't match"
        )
        np.testing.assert_allclose(
            dk_xla, dk_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Key gradients with sliding_window don't match"
        )
        np.testing.assert_allclose(
            dv_xla, dv_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Value gradients with sliding_window don't match"
        )

    def test_gradient_with_sliding_window_asymmetric(self):
        """Test gradient computation with asymmetric sliding window."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        sliding_window = (32, 96)  # Asymmetric window

        # Define loss function for XLA
        def xla_loss_fn(q, k, v):
            out = xla.ring_attention(
                q, k, v, sliding_window=sliding_window,
                query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out ** 2)

        # Define loss function for Pallas
        def pallas_loss_fn(q, k, v):
            out = pallas.tpu.ring_attention(
                q, k, v, sliding_window=sliding_window,
                query_chunk_size=128, key_chunk_size=128
            )
            return jnp.mean(out ** 2)

        # Compute gradients
        xla_grad_fn = jax.grad(xla_loss_fn, argnums=(0, 1, 2))
        dq_xla, dk_xla, dv_xla = xla_grad_fn(q, k, v)

        pallas_grad_fn = jax.grad(pallas_loss_fn, argnums=(0, 1, 2))
        dq_pallas, dk_pallas, dv_pallas = pallas_grad_fn(q, k, v)

        # Compare gradients
        np.testing.assert_allclose(
            dq_xla, dq_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Query gradients with asymmetric sliding_window don't match"
        )
        np.testing.assert_allclose(
            dk_xla, dk_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Key gradients with asymmetric sliding_window don't match"
        )
        np.testing.assert_allclose(
            dv_xla, dv_pallas, rtol=self.grad_rtol, atol=self.grad_atol,
            err_msg="Value gradients with asymmetric sliding_window don't match"
        )

    def test_xla_attention_vs_ring_attention_gradients(self):
        """Test gradients between XLA standard attention and ring attention."""
        batch, seq_len, num_heads, head_dim = 2, 256, 8, 64
        q, k, v = [numeric_gen(batch, seq_len, num_heads, head_dim, dtype="f4") for _ in range(3)]

        # Create softmax_aux for testing
        num_sinks = 4
        softmax_aux = jnp.ones((num_heads, num_sinks), dtype=jnp.float32) * -2.0

        # Define loss function for XLA standard attention
        def xla_attention_loss_fn(q, k, v):
            out, _ = xla.attention(
                q, k, v, attention_mask=None, bias=None, softmax_aux=softmax_aux, softmax_scale=None, deterministic=True
            )
            return jnp.mean(out**2)

        # Define loss function for XLA ring attention
        def xla_ring_loss_fn(q, k, v):
            out = xla.ring_attention(q, k, v, softmax_aux=softmax_aux, query_chunk_size=128, key_chunk_size=128)
            return jnp.mean(out**2)

        # Compute gradients
        xla_attention_grad_fn = jax.grad(xla_attention_loss_fn, argnums=(0, 1, 2))
        dq_att, dk_att, dv_att = xla_attention_grad_fn(q, k, v)

        xla_ring_grad_fn = jax.grad(xla_ring_loss_fn, argnums=(0, 1, 2))
        dq_ring, dk_ring, dv_ring = xla_ring_grad_fn(q, k, v)

        # Compare gradients (should be close when using same features)
        np.testing.assert_allclose(
            dq_att,
            dq_ring,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Query gradients between XLA attention and ring attention don't match",
        )
        np.testing.assert_allclose(
            dk_att,
            dk_ring,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Key gradients between XLA attention and ring attention don't match",
        )
        np.testing.assert_allclose(
            dv_att,
            dv_ring,
            rtol=1e-2,
            atol=1e-2,
            err_msg="Value gradients between XLA attention and ring attention don't match",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
