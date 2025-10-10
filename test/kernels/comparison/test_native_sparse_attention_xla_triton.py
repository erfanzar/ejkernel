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


"""Comparison tests between XLA and Triton native_sparse_attention implementations.

Notes:
- Triton path runs in fp16/bf16, XLA path in fp32; tolerances account for precision differences.
- Per-token sparsity format is used: block_indices shape [B, T, H_kv, S], block_counts int or [B, T, H_kv].
- GQA is enforced for Triton: (num_q_heads / num_kv_heads) must be a multiple of 16.
"""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.kernels._triton.native_sparse_attention import apply_native_sparse_attention as apply_nsa_triton
from ejkernel.kernels._triton.native_sparse_attention import native_sparse_attention as nsa_triton
from ejkernel.kernels._xla.native_sparse_attention import apply_native_sparse_attention as apply_nsa_xla
from ejkernel.kernels._xla.native_sparse_attention import native_sparse_attention as nsa_xla


def _cos_rel_l2(a: jnp.ndarray, b: jnp.ndarray):
    a32 = a.astype(jnp.float32)
    b32 = b.astype(jnp.float32)
    dot = jnp.vdot(a32, b32)
    denom = (jnp.linalg.norm(a32) * jnp.linalg.norm(b32)) + 1e-6
    cos = dot / denom
    rel_l2 = jnp.linalg.norm(a32 - b32) / (jnp.linalg.norm(a32) + 1e-6)
    return float(cos), float(rel_l2)


class TestApplyNativeSparseAttentionComparison:
    """Compare apply_native_sparse_attention between XLA and Triton."""

    def test_basic_sparse_attention(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        S = 4

        key = jax.random.PRNGKey(0)
        kq, kk, kv = jax.random.split(key, 3)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        block_indices = jnp.tile(
            jnp.arange(S, dtype=jnp.int32)[None, None, None, :],
            (batch, seq_len, num_kv_heads, 1),
        )

        block_counts = jnp.full((batch, seq_len, num_kv_heads), S, dtype=jnp.int32)

        softmax_scale = float(head_dim**-0.5)

        out_xla = apply_nsa_xla(q_f32, k_f32, v_f32, block_indices, block_counts, block_size, softmax_scale)
        out_tri = apply_nsa_triton(q_f16, k_f16, v_f16, block_indices, block_counts, block_size, softmax_scale)

        assert out_xla.shape == (batch, seq_len, num_q_heads, head_dim)
        assert out_tri.shape == (batch, seq_len, num_q_heads, head_dim)
        assert jnp.isfinite(out_xla).all() and jnp.isfinite(out_tri).all()

        assert jnp.allclose(out_xla, out_tri.astype(jnp.float32), rtol=1.5e-1, atol=1.5e-2)

    def test_uniform_block_counts_int(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 32, 2, 32
        block_size = 64
        S = 2

        key = jax.random.PRNGKey(42)
        kq, kk, kv = jax.random.split(key, 3)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        block_indices = jnp.zeros((batch, seq_len, num_kv_heads, S), dtype=jnp.int32)
        for t in range(seq_len):
            cur = t // block_size
            inds = jnp.arange(max(0, cur - S + 1), cur + 1, dtype=jnp.int32)
            if inds.shape[0] < S:
                inds = jnp.pad(inds, (0, S - inds.shape[0]), constant_values=0)
            block_indices = block_indices.at[:, t, :, :].set(inds)

        softmax_scale = float(head_dim**-0.5)

        out_xla = apply_nsa_xla(q_f32, k_f32, v_f32, block_indices, S, block_size, softmax_scale)
        out_tri = apply_nsa_triton(q_f16, k_f16, v_f16, block_indices, S, block_size, softmax_scale)

        assert jnp.allclose(out_xla, out_tri.astype(jnp.float32), rtol=5e-2, atol=5e-3)


class TestNativeSparseAttentionComparison:
    """Compare full native_sparse_attention (with compression) between XLA and Triton."""

    def test_with_precomputed_indices(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        S = 4

        key = jax.random.PRNGKey(123)
        kq, kk, kv = jax.random.split(key, 3)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        block_indices = jnp.tile(
            jnp.arange(S, dtype=jnp.int32)[None, None, None, :],
            (batch, seq_len, num_kv_heads, 1),
        )

        out_xla = nsa_xla(q_f32, k_f32, v_f32, block_indices=block_indices, block_counts=S, block_size=block_size)
        out_tri = nsa_triton(q_f16, k_f16, v_f16, block_indices=block_indices, block_counts=S, block_size=block_size)

        assert jnp.allclose(out_xla, out_tri.astype(jnp.float32), rtol=5e-2, atol=5e-3)

    def test_with_compression_and_selection(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 2, 256, 32, 2, 64
        block_size = 64
        S = 4

        key = jax.random.PRNGKey(456)
        kq, kk, kv, kg = jax.random.split(key, 4)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        g_cmp_f32 = jax.random.uniform(kg, (batch, seq_len, num_q_heads), dtype=jnp.float32)
        g_cmp_f16 = g_cmp_f32.astype(jnp.float16)

        out_xla = nsa_xla(q_f32, k_f32, v_f32, g_cmp=g_cmp_f32, block_counts=S, block_size=block_size)
        out_tri = nsa_triton(q_f16, k_f16, v_f16, g_cmp=g_cmp_f16, block_counts=S, block_size=block_size)

        assert jnp.allclose(out_xla, out_tri.astype(jnp.float32), rtol=1.5e-1, atol=1.5e-2)

    def test_with_both_gates(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 32, 2, 32
        block_size = 64
        S = 2

        key = jax.random.PRNGKey(789)
        kq, kk, kv, kg1, kg2 = jax.random.split(key, 5)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        g_cmp_f32 = jax.random.uniform(kg1, (batch, seq_len, num_q_heads), dtype=jnp.float32)
        g_slc_f32 = jax.random.uniform(kg2, (batch, seq_len, num_q_heads), dtype=jnp.float32)
        g_cmp_f16, g_slc_f16 = g_cmp_f32.astype(jnp.float16), g_slc_f32.astype(jnp.float16)

        out_xla = nsa_xla(q_f32, k_f32, v_f32, g_cmp=g_cmp_f32, g_slc=g_slc_f32, block_counts=S, block_size=block_size)
        out_tri = nsa_triton(
            q_f16, k_f16, v_f16, g_cmp=g_cmp_f16, g_slc=g_slc_f16, block_counts=S, block_size=block_size
        )

        assert jnp.allclose(out_xla, out_tri.astype(jnp.float32), rtol=1.5e-1, atol=1.5e-2)


class TestGradientComparison:
    """Compare gradients between XLA and Triton implementations."""

    def test_gradient_agreement(self):
        batch, seq_len, num_q_heads, num_kv_heads, head_dim = 1, 128, 32, 2, 32
        block_size = 64
        S = 2

        key = jax.random.PRNGKey(999)
        kq, kk, kv = jax.random.split(key, 3)

        q_f32 = jax.random.normal(kq, (batch, seq_len, num_q_heads, head_dim), dtype=jnp.float32)
        k_f32 = jax.random.normal(kk, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)
        v_f32 = jax.random.normal(kv, (batch, seq_len, num_kv_heads, head_dim), dtype=jnp.float32)

        q_f16, k_f16, v_f16 = q_f32.astype(jnp.float16), k_f32.astype(jnp.float16), v_f32.astype(jnp.float16)

        block_indices = jnp.tile(
            jnp.arange(S, dtype=jnp.int32)[None, None, None, :],
            (batch, seq_len, num_kv_heads, 1),
        )
        softmax_scale = float(head_dim**-0.5)

        def loss_xla(q, k, v):
            out = apply_nsa_xla(q, k, v, block_indices, S, block_size, softmax_scale)
            return jnp.sum(out**2)

        def loss_tri(q, k, v):
            out = apply_nsa_triton(q, k, v, block_indices, S, block_size, softmax_scale)
            return jnp.sum(out**2)

        gx = jax.grad(loss_xla, argnums=(0, 1, 2))(q_f32, k_f32, v_f32)
        gy = jax.grad(loss_tri, argnums=(0, 1, 2))(q_f16, k_f16, v_f16)

        for g_xla, g_tri in zip(gx, gy, strict=False):
            ok_elem = jnp.allclose(g_xla, g_tri.astype(jnp.float32), rtol=2e-1, atol=5e-2)

            cos, rel_l2 = _cos_rel_l2(g_xla, g_tri)
            assert ok_elem or (cos > 0.98 and rel_l2 < 0.2), f"grad mismatch: cos={cos:.4f}, rel_l2={rel_l2:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
