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


"""Comparison tests between XLA and Triton flash attention implementations."""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.flash_attention import flash_attention as flash_attention_triton
from ejkernel.kernels._xla.flash_attention import flash_attention as flash_attention_xla


class TestBasicComparison:
    """Compare basic functionality between XLA and Triton."""

    def test_basic_attention(self):
        """Test that basic attention produces similar results."""
        batch, seq_len, num_heads, head_dim = 2, 32, 4, 64
        key = jax.random.PRNGKey(0)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)

        out_xla = flash_attention_xla(q_f32, k_f32, v_f32)
        out_triton = flash_attention_triton(q_f16, k_f16, v_f16)

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=1e-2, atol=1e-3)


class TestSoftCapComparison:
    """Compare logits_soft_cap between XLA and Triton."""

    def test_soft_cap_agreement(self):
        """Test that soft cap produces similar results."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        key = jax.random.PRNGKey(42)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)

        logits_soft_cap = 20.0

        out_xla = flash_attention_xla(q_f32, k_f32, v_f32, logits_soft_cap=logits_soft_cap)
        out_triton = flash_attention_triton(q_f16, k_f16, v_f16, logits_soft_cap=logits_soft_cap)

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=2e-2, atol=1e-3)

    def test_soft_cap_gradient_agreement(self):
        """Test that gradients with soft cap are similar."""
        batch, seq_len, num_heads, head_dim = 1, 8, 2, 16
        key = jax.random.PRNGKey(0)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)

        def loss_xla(q, k, v):
            out = flash_attention_xla(q, k, v, logits_soft_cap=20.0)
            return jnp.sum(out**2)

        def loss_triton(q, k, v):
            out = flash_attention_triton(q, k, v, logits_soft_cap=20.0)
            return jnp.sum(out.astype(jnp.float32) ** 2)

        grads_xla = jax.grad(loss_xla, argnums=(0, 1, 2))(q_f32, k_f32, v_f32)
        grads_triton = jax.grad(loss_triton, argnums=(0, 1, 2))(q_f16, k_f16, v_f16)

        for g_xla, g_triton in zip(grads_xla, grads_triton, strict=False):
            assert jnp.allclose(g_xla, g_triton.astype(jnp.float32), rtol=5e-2, atol=1e-2)


class TestSoftmaxAuxComparison:
    """Compare softmax_aux (attention sinks) between XLA and Triton."""

    def test_sinks_agreement(self):
        """Test that attention sinks produce similar results."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        num_sinks = 4
        key = jax.random.PRNGKey(123)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks_f32 = jax.random.normal(key, (num_heads, num_sinks)) * 0.1

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)
        sinks_f16 = sinks_f32.astype(jnp.float16)

        out_xla = flash_attention_xla(q_f32, k_f32, v_f32, softmax_aux=sinks_f32)
        out_triton = flash_attention_triton(q_f16, k_f16, v_f16, softmax_aux=sinks_f16)

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=2e-2, atol=1e-3)

    def test_sinks_broadcast_agreement(self):
        """Test that 1D sink broadcasting works similarly."""
        batch, seq_len, num_heads, head_dim = 1, 8, 4, 16
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks_1d_f32 = jax.random.normal(key, (num_sinks,))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)
        sinks_1d_f16 = sinks_1d_f32.astype(jnp.float16)

        out_xla = flash_attention_xla(q_f32, k_f32, v_f32, softmax_aux=sinks_1d_f32)
        out_triton = flash_attention_triton(q_f16, k_f16, v_f16, softmax_aux=sinks_1d_f16)

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=2e-2, atol=1e-3)


class TestCombinedFeaturesComparison:
    """Compare combined features between XLA and Triton."""

    def test_soft_cap_and_sinks_together(self):
        """Test that both features work together consistently."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        num_sinks = 4
        key = jax.random.PRNGKey(999)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks_f32 = jax.random.normal(key, (num_heads, num_sinks))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)
        sinks_f16 = sinks_f32.astype(jnp.float16)

        out_xla = flash_attention_xla(q_f32, k_f32, v_f32, logits_soft_cap=20.0, softmax_aux=sinks_f32)
        out_triton = flash_attention_triton(q_f16, k_f16, v_f16, logits_soft_cap=20.0, softmax_aux=sinks_f16)

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=3e-2, atol=1e-3)

    def test_all_features_with_causal(self):
        """Test all features with causal masking."""
        batch, seq_len, num_heads, head_dim = 1, 16, 2, 32
        num_sinks = 2
        key = jax.random.PRNGKey(0)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        k_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))
        sinks_f32 = jax.random.normal(key, (num_heads, num_sinks))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)
        sinks_f16 = sinks_f32.astype(jnp.float16)

        out_xla = flash_attention_xla(
            q_f32,
            k_f32,
            v_f32,
            sliding_window=(seq_len - 1, 0),
            logits_soft_cap=20.0,
            softmax_aux=sinks_f32,
        )
        out_triton = flash_attention_triton(
            q_f16,
            k_f16,
            v_f16,
            causal=True,
            logits_soft_cap=20.0,
            softmax_aux=sinks_f16,
        )

        assert jnp.allclose(out_xla, out_triton.astype(jnp.float32), rtol=3e-2, atol=1e-3)


class TestNumericalStability:
    """Test numerical stability across implementations."""

    def test_large_logits_with_soft_cap(self):
        """Test that soft cap handles large values well."""
        batch, seq_len, num_heads, head_dim = 1, 8, 2, 16
        key = jax.random.PRNGKey(0)

        q_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim)) * 3
        k_f32 = q_f32
        v_f32 = jax.random.normal(key, (batch, seq_len, num_heads, head_dim))

        q_f16 = q_f32.astype(jnp.float16)
        k_f16 = k_f32.astype(jnp.float16)
        v_f16 = v_f32.astype(jnp.float16)

        out_xla_no_cap = flash_attention_xla(q_f32, k_f32, v_f32)
        out_triton_no_cap = flash_attention_triton(q_f16, k_f16, v_f16)

        out_xla_cap = flash_attention_xla(q_f32, k_f32, v_f32, logits_soft_cap=20.0)
        out_triton_cap = flash_attention_triton(q_f16, k_f16, v_f16, logits_soft_cap=20.0)

        assert jnp.all(jnp.isfinite(out_xla_no_cap))
        assert jnp.all(jnp.isfinite(out_triton_no_cap))
        assert jnp.all(jnp.isfinite(out_xla_cap))
        assert jnp.all(jnp.isfinite(out_triton_cap))

        _diff_no_cap = jnp.abs(out_xla_no_cap - out_triton_no_cap.astype(jnp.float32)).mean()
        diff_cap = jnp.abs(out_xla_cap - out_triton_cap.astype(jnp.float32)).mean()

        assert jnp.isfinite(diff_cap)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
