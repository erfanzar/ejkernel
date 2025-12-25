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


"""Comprehensive tests for all ejkernel module operations.

This test suite covers all operations in ejkernel.modules with various scenarios:
- Basic functionality (simple shapes, forward pass)
- Different sequence lengths
- Different batch sizes
- Different head counts (including MQA/GQA scenarios)
- Gradient computation (backward pass)
- Platform-specific features
- Edge cases (small/large dimensions)

The tests verify that operations execute without errors across different
configurations, but do not compare outputs to reference implementations.
"""

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules import (
    attention,
    blocksparse_attention,
    flash_attention,
    gla_attention,
    grouped_matmul,
    lightning_attention,
    mean_pooling,
    native_sparse_attention,
    page_attention,
    ragged_decode_attention,
    ragged_page_attention_v2,
    recurrent_attention,
    ring_attention,
    scaled_dot_product_attention,
    unified_attention,
)
from ejkernel.ops import get_device_platform
from ejkernel.types import MaskInfo


def rand_tensors(B, Nq, Nk, Hq, Hkv, D, dtype=jnp.float16, key=0):
    """Generate random Q, K, V tensors."""
    rng = jax.random.PRNGKey(key)
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (B, Nq, Hq, D), dtype=dtype)
    k = jax.random.normal(k2, (B, Nk, Hkv, D), dtype=dtype)
    v = jax.random.normal(k3, (B, Nk, Hkv, D), dtype=dtype)
    return q, k, v


class TestFlashAttention:
    """Test suite for FlashAttention operation."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [128, 512, 2048])
    @pytest.mark.parametrize("num_heads", [8, 16])
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_flash_attention_basic(self, batch_size, seq_len, num_heads, head_dim):
        """Test basic flash attention with various shapes."""
        q, k, v = rand_tensors(batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v)
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_flash_attention_causal(self):
        """Test flash attention with causal masking."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v, causal=True)
        assert output.shape == (B, N, H, D)

    def test_flash_attention_with_scale(self):
        """Test flash attention with custom softmax scale."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        scale = D**-0.5
        output = flash_attention(q, k, v, softmax_scale=scale)
        assert output.shape == (B, N, H, D)

    def test_flash_attention_with_bias(self):
        """Test flash attention with attention bias."""
        B, N, H, D = 2, 256, 8, 64
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        bias = jax.random.normal(jax.random.PRNGKey(42), (B, H, N, N), dtype=jnp.bfloat16)
        output = flash_attention(q, k, v, bias)
        assert output.shape == (B, N, H, D)

    def test_flash_attention_sliding_window(self):
        """Test flash attention with sliding window."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v, sliding_window=(256, 256))
        assert output.shape == (B, N, H, D)

    def test_flash_attention_logits_soft_cap(self):
        """Test flash attention with logits soft capping."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v, logits_soft_cap=30.0)
        assert output.shape == (B, N, H, D)

    def test_flash_attention_gqa(self):
        """Test flash attention with grouped-query attention (GQA)."""
        B, N, Hq, Hkv, D = 2, 512, 16, 4, 128
        q, k, v = rand_tensors(B, N, N, Hq, Hkv, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v)
        assert output.shape == (B, N, Hq, D)

    def test_flash_attention_mqa(self):
        """Test flash attention with multi-query attention (MQA)."""
        B, N, Hq, Hkv, D = 2, 512, 16, 1, 128
        q, k, v = rand_tensors(B, N, N, Hq, Hkv, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v)
        assert output.shape == (B, N, Hq, D)

    def test_flash_attention_cross_attention(self):
        """Test flash attention with different query and key sequence lengths."""
        B, Nq, Nk, H, D = 2, 128, 512, 8, 128
        q, k, v = rand_tensors(B, Nq, Nk, H, H, D, dtype=jnp.bfloat16)
        output = flash_attention(q, k, v)
        assert output.shape == (B, Nq, H, D)

    def test_flash_attention_gradient(self):
        """Test flash attention gradient computation."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        def loss_fn(q, k, v):
            output = flash_attention(q, k, v, causal=True)
            return jnp.mean(output)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert len(grads) == 3
        assert grads[0].shape == q.shape
        assert grads[1].shape == k.shape
        assert grads[2].shape == v.shape

    def test_flash_attention_packed_segments_xla_matches_dense(self):
        """Test packed/multi-sequence segment IDs on XLA."""
        B, T, Hq, Hkv, D = 2, 64, 8, 4, 32
        q, k, v = rand_tensors(B, T, T, Hq, Hkv, D, dtype=jnp.bfloat16, key=0)

        seg = jnp.array(
            [
                [0] * 24 + [1] * 16 + [2] * 8 + [-1] * 16,
                [0] * 16 + [1] * 16 + [-1] * 32,
            ],
            dtype=jnp.int32,
        )
        mask_info = MaskInfo.from_segments(q_segment_ids=seg)

        out = flash_attention(q, k, v, mask_info=mask_info, platform="xla")

        reps = Hq // Hkv
        k_rep = jnp.repeat(k, reps, axis=2)
        v_rep = jnp.repeat(v, reps, axis=2)
        scale = D**-0.5

        scores = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k_rep.astype(jnp.float32)) * scale
        q_ids = seg[:, None, :, None]
        kv_ids = seg[:, None, None, :]
        mask = (q_ids == kv_ids) & (q_ids >= 0)
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1).astype(jnp.float32)
        ref = jnp.einsum("bhqk,bkhd->bqhd", weights, v_rep.astype(jnp.float32)).astype(q.dtype)
        ref = jnp.where((seg >= 0)[:, :, None, None], ref, 0)

        assert jnp.allclose(out, ref, atol=0.2, rtol=0.0)

    @pytest.mark.skipif(get_device_platform() != "gpu", reason="requires GPU")
    def test_flash_attention_packed_segments_triton_matches_xla(self):
        """Test packed/multi-sequence segment IDs on Triton vs XLA."""
        B, T, Hq, Hkv, D = 2, 128, 8, 4, 64
        q, k, v = rand_tensors(B, T, T, Hq, Hkv, D, dtype=jnp.bfloat16, key=1)
        seg = jnp.array(
            [
                [0] * 48 + [1] * 32 + [2] * 16 + [-1] * 32,
                [0] * 64 + [1] * 32 + [-1] * 32,
            ],
            dtype=jnp.int32,
        )
        mask_info = MaskInfo.from_segments(q_segment_ids=seg)
        out_triton = flash_attention(q, k, v, mask_info=mask_info, platform="triton")
        out_xla = flash_attention(q, k, v, mask_info=mask_info, platform="xla")
        assert jnp.allclose(out_triton, out_xla, atol=0.25, rtol=0.0)


class TestAttention:
    """Test suite for standard Attention operation."""

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("num_heads", [8, 16])
    def test_attention_basic(self, batch_size, seq_len, num_heads):
        """Test basic attention with various shapes."""
        head_dim = 128
        q, k, v = rand_tensors(batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, dtype=jnp.bfloat16)
        output, _ = attention(q, k, v)
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_attention_with_mask(self):
        """Test attention with attention mask."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        mask = jax.random.randint(jax.random.PRNGKey(0), (B, 1, N, N), 0, 2).astype(bool)
        mask_info = MaskInfo.from_attention_mask(mask)
        output, _ = attention(q, k, v, mask_info=mask_info)
        assert output.shape == (B, N, H, D)

    def test_attention_with_bias(self):
        """Test attention with attention bias."""
        B, N, H, D = 2, 256, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        bias = jax.random.normal(jax.random.PRNGKey(42), (B, H, N, N), dtype=jnp.bfloat16)
        output, _ = attention(q, k, v, bias)
        assert output.shape == (B, N, H, D)

    def test_attention_sliding_window(self):
        """Test attention with sliding window."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output, _ = attention(q, k, v, sliding_window=(256, 256))
        assert output.shape == (B, N, H, D)

    def test_attention_gradient(self):
        """Test attention gradient computation."""
        B, N, H, D = 2, 256, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        def loss_fn(q, k, v):
            output, _ = attention(q, k, v)
            return jnp.mean(output)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert len(grads) == 3


class TestBlockSparseAttention:
    """Test suite for BlockSparseAttention operation."""

    @pytest.mark.parametrize("batch_size", [1, 2])
    @pytest.mark.parametrize("seq_len", [512, 1024])
    @pytest.mark.parametrize("num_heads", [8, 16])
    def test_blocksparse_attention_basic(self, batch_size, seq_len, num_heads):
        """Test basic block sparse attention."""
        head_dim = 128
        q, k, v = rand_tensors(batch_size, seq_len, seq_len, num_heads, num_heads, head_dim, dtype=jnp.bfloat16)

        q_t = q.transpose(0, 2, 1, 3)
        k_t = k.transpose(0, 2, 1, 3)
        v_t = v.transpose(0, 2, 1, 3)
        output = blocksparse_attention(q_t, k_t, v_t)
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    def test_blocksparse_attention_causal(self):
        """Test block sparse attention with causal masking."""
        B, N, H, D = 2, 1024, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        q_t, k_t, v_t = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        output = blocksparse_attention(q_t, k_t, v_t, causal=True)
        assert output.shape == (B, H, N, D)

    def test_blocksparse_attention_with_mask(self):
        """Test block sparse attention with attention mask."""
        B, N, H, D = 2, 1024, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        q_t, k_t, v_t = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        mask = jax.random.randint(jax.random.PRNGKey(0), (B, 1, N, N), 0, 4) > 2
        mask_info = MaskInfo.from_attention_mask(mask)
        output = blocksparse_attention(q_t, k_t, v_t, mask_info=mask_info)
        assert output.shape == (B, H, N, D)

    def test_blocksparse_attention_sliding_window(self):
        """Test block sparse attention with sliding window."""
        B, N, H, D = 2, 1024, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        q_t, k_t, v_t = q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3)
        output = blocksparse_attention(q_t, k_t, v_t, sliding_window=(256, 256))
        assert output.shape == (B, H, N, D)


class TestPageAttention:
    """Test suite for PageAttention operation."""

    def test_page_attention_basic(self):
        """Test basic page attention."""
        num_seqs, H, D = 2, 8, 128
        page_size = 16
        num_pages = 8

        q = jax.random.normal(jax.random.PRNGKey(0), (num_seqs, H, D), dtype=jnp.bfloat16)

        k_cache = jax.random.normal(jax.random.PRNGKey(1), (num_pages, H, page_size, D), dtype=jnp.bfloat16)
        v_cache = jax.random.normal(jax.random.PRNGKey(2), (num_pages, H, page_size, D), dtype=jnp.bfloat16)
        block_tables = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=jnp.int32)
        context_lens = jnp.array([48, 64], dtype=jnp.int32)

        output = page_attention(q, k_cache, v_cache, context_lens, block_tables)
        assert output.shape == (num_seqs, H, D)

    def test_page_attention_variable_context(self):
        """Test page attention with variable context lengths."""
        num_seqs, H, D = 4, 8, 128
        page_size = 16
        num_pages = 16

        q = jax.random.normal(jax.random.PRNGKey(0), (num_seqs, H, D), dtype=jnp.bfloat16)
        k_cache = jax.random.normal(jax.random.PRNGKey(1), (num_pages, H, page_size, D), dtype=jnp.bfloat16)
        v_cache = jax.random.normal(jax.random.PRNGKey(2), (num_pages, H, page_size, D), dtype=jnp.bfloat16)
        block_tables = jnp.array([[0, 1, 2, 3], [4, 5, -1, -1], [6, 7, 8, -1], [9, 10, 11, 12]], dtype=jnp.int32)
        context_lens = jnp.array([64, 32, 48, 64], dtype=jnp.int32)

        output = page_attention(q, k_cache, v_cache, context_lens, block_tables)
        assert output.shape == (num_seqs, H, D)


class TestRaggedPageAttention:
    """Test suite for RaggedPageAttentionv2 operation."""

    def test_ragged_page_attention_basic(self):
        """Test basic ragged page attention."""
        total_q_len = 256
        H, D = 8, 128
        page_size = 16
        num_pages = 16

        q = jax.random.normal(jax.random.PRNGKey(0), (total_q_len, H, D), dtype=jnp.bfloat16)

        k_cache = jax.random.normal(jax.random.PRNGKey(1), (num_pages, H, page_size, D), dtype=jnp.bfloat16)
        v_cache = jax.random.normal(jax.random.PRNGKey(2), (num_pages, H, page_size, D), dtype=jnp.bfloat16)

        cu_seqlens_q = jnp.array([0, 64, 192, 256], dtype=jnp.int32)
        block_tables = jnp.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], dtype=jnp.int32)
        context_lens = jnp.array([48, 96, 64], dtype=jnp.int32)

        k_pages = jnp.transpose(k_cache, (0, 2, 1, 3))
        v_pages = jnp.transpose(v_cache, (0, 2, 1, 3))
        kv_pages = jnp.stack([k_pages, v_pages], axis=3).reshape(num_pages, page_size, H * 2, D)

        output = ragged_page_attention_v2(q, kv_pages, context_lens, block_tables, cu_seqlens_q, 3)
        assert output.shape == (total_q_len, H, D)


class TestRaggedDecodeAttention:
    """Test suite for RaggedDecodeAttention operation."""

    def test_ragged_decode_attention_basic(self):
        """Test basic ragged decode attention."""
        H, D = 8, 128
        max_kv_len = 512
        num_seqs = 4

        q = jax.random.normal(jax.random.PRNGKey(0), (num_seqs, H, D), dtype=jnp.bfloat16)

        k = jax.random.normal(jax.random.PRNGKey(1), (num_seqs, max_kv_len, H, D), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.PRNGKey(2), (num_seqs, max_kv_len, H, D), dtype=jnp.bfloat16)

        kv_lengths = jnp.array([256, 384, 128, 512], dtype=jnp.int32)

        sequence_start = jnp.zeros((num_seqs,), dtype=jnp.int32)
        sequence_end = kv_lengths

        output = ragged_decode_attention(q, k, v, sequence_start, sequence_end)
        assert output.shape == (num_seqs, H, D)


class TestUnifiedAttention:
    """Test suite for UnifiedAttention operation."""

    def test_unified_attention_basic(self):
        """Test unified attention forward shape."""
        num_seqs = 3
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 32
        block_size = 8

        kv_lens = [32, 17, 9]
        q_lens = [8, 1, 4]
        max_kv = max(kv_lens)
        max_blocks_per_seq = (max_kv + block_size - 1) // block_size
        num_blocks_total = num_seqs * max_blocks_per_seq

        block_tables = jnp.arange(num_blocks_total, dtype=jnp.int32).reshape(num_seqs, max_blocks_per_seq)
        kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)

        cu = [0]
        for q in q_lens:
            cu.append(cu[-1] + int(q))
        query_start_loc = jnp.array(cu, dtype=jnp.int32)
        total_tokens = int(query_start_loc[-1])

        q = jax.random.normal(jax.random.PRNGKey(0), (total_tokens, num_q_heads, head_dim), dtype=jnp.float32).astype(
            jnp.bfloat16
        )
        k_cache = jax.random.normal(
            jax.random.PRNGKey(1), (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
        ).astype(jnp.bfloat16)
        v_cache = jax.random.normal(
            jax.random.PRNGKey(2), (num_blocks_total, block_size, num_kv_heads, head_dim), dtype=jnp.float32
        ).astype(jnp.bfloat16)

        out = unified_attention(q, k_cache, v_cache, kv_lens_arr, block_tables, query_start_loc)
        assert out.shape == (total_tokens, num_q_heads, head_dim)


class TestRingAttention:
    """Test suite for RingAttention operation."""

    @pytest.mark.skip(reason="RingAttention requires JAX pmap distributed context with axis_name")
    def test_ring_attention_basic(self):
        """Test basic ring attention."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output = ring_attention(q, k, v, axis_name="batch")
        assert output.shape == (B, N, H, D)

    @pytest.mark.skip(reason="RingAttention requires JAX pmap distributed context")
    def test_ring_attention_causal(self):
        """Test ring attention with causal masking."""
        B, N, H, D = 2, 1024, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        output = ring_attention(q, k, v, axis_name="batch", chunk_size=512, causal=True)
        assert output.shape == (B, N, H, D)


class TestRecurrentAttention:
    """Test suite for RecurrentAttention operation."""

    def test_recurrent_attention_basic(self):
        """Test basic recurrent attention."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        output = recurrent_attention(q, k, v)
        assert output.shape == (B, N, H, D)

    def test_recurrent_attention_with_scale(self):
        """Test recurrent attention with custom scale."""
        B, N, H, D = 2, 256, 8, 64
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        scale = D**-0.5
        output = recurrent_attention(q, k, v, softmax_scale=scale)
        assert output.shape == (B, N, H, D)


class TestGLAttention:
    """Test suite for Gated Linear Attention operation."""

    def test_gla_attention_basic(self):
        """Test basic gated linear attention."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        g = jax.random.normal(jax.random.PRNGKey(42), (B, N, H, D), dtype=jnp.bfloat16)
        output = gla_attention(q, k, v, g)
        assert output.shape == (B, N, H, D)

    def test_gla_attention_with_scale(self):
        """Test GLA with custom scale."""
        B, N, H, D = 2, 256, 8, 64
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)
        g = jax.random.normal(jax.random.PRNGKey(42), (B, N, H, D), dtype=jnp.bfloat16)
        scale = D**-0.5
        output = gla_attention(q, k, v, g, softmax_scale=scale)
        assert output.shape == (B, N, H, D)


class TestLightningAttention:
    """Test suite for LightningAttention operation."""

    def test_lightning_attention_basic(self):
        """Test basic lightning attention."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        output = lightning_attention(q, k, v, layer_idx=0, num_layers=1)
        assert output.shape == (B, N, H, D)

    def test_lightning_attention_with_scale(self):
        """Test lightning attention with custom scale."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        output = lightning_attention(q, k, v, layer_idx=5, num_layers=12, softmax_scale=0.125)
        assert output.shape == (B, N, H, D)


class TestNativeSparseAttention:
    """Test suite for NativeSparseAttention operation."""

    def test_native_sparse_attention_basic(self):
        """Test basic native sparse attention."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        block_size = 64
        num_blocks = (N + block_size - 1) // block_size
        block_counts = min(4, num_blocks)
        block_indices = jnp.tile(jnp.arange(block_counts, dtype=jnp.int32)[None, None, None, :], (B, N, H, 1))

        output = native_sparse_attention(q, k, v, None, None, block_indices, block_counts)
        assert output.shape == (B, N, H, D)

    def test_native_sparse_attention_with_indices(self):
        """Test native sparse attention with explicit block indices."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        block_size = 64
        num_blocks = (N + block_size - 1) // block_size
        block_counts = min(4, num_blocks)
        block_indices = jnp.tile(jnp.arange(block_counts, dtype=jnp.int32)[None, None, None, :], (B, N, H, 1))

        output = native_sparse_attention(q, k, v, None, None, block_indices, block_counts, softmax_scale=0.125)
        assert output.shape == (B, N, H, D)


class TestGroupedMatmul:
    """Test suite for GroupedMatmul operation."""

    def test_grouped_matmul_basic(self):
        """Test basic grouped matrix multiplication."""
        M, K, N = 256, 128, 64
        num_groups = 4
        lhs = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
        rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, K, N), dtype=jnp.bfloat16)
        group_sizes = jnp.array([64, 64, 64, 64], dtype=jnp.int32)

        output = grouped_matmul(lhs, rhs, group_sizes)
        assert output.shape == (M, N)

    def test_grouped_matmul_variable_sizes(self):
        """Test grouped matmul with variable group sizes."""
        M, K, N = 256, 128, 64
        num_groups = 3
        lhs = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
        rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, K, N), dtype=jnp.bfloat16)
        group_sizes = jnp.array([100, 100, 56], dtype=jnp.int32)

        output = grouped_matmul(lhs, rhs, group_sizes)
        assert output.shape == (M, N)

    def test_grouped_matmul_transposed_rhs(self):
        """Test grouped matmul with transposed RHS."""
        M, K, N = 256, 128, 64
        num_groups = 4
        lhs = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
        rhs = jax.random.normal(jax.random.PRNGKey(1), (num_groups, N, K), dtype=jnp.bfloat16)
        group_sizes = jnp.array([64, 64, 64, 64], dtype=jnp.int32)

        output = grouped_matmul(lhs, rhs, group_sizes, transpose_rhs=True)
        assert output.shape == (M, N)


class TestMeanPooling:
    """Test suite for MeanPooling operation."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("hidden_dim", [256, 768])
    def test_mean_pooling_basic(self, batch_size, seq_len, hidden_dim):
        """Test basic mean pooling."""
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_dim), dtype=jnp.bfloat16)
        output = mean_pooling(x)
        assert output.shape == (batch_size, hidden_dim)

    @pytest.mark.skip(reason="MeanPooling with cu_seqlens has JAX tracer issues in current implementation")
    def test_mean_pooling_variable_length(self):
        """Test mean pooling with variable sequence lengths."""
        batch_size, seq_len, hidden_dim = 4, 512, 768
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, 8, hidden_dim), dtype=jnp.bfloat16)

        cu_seqlens = jnp.array([0, 128, 256, 384, 512], dtype=jnp.int32)
        output = mean_pooling(x, cu_seqlens=cu_seqlens)
        assert output.shape == (batch_size, hidden_dim)

    def test_mean_pooling_custom_chunk(self):
        """Test mean pooling with custom chunk size."""
        batch_size, seq_len, hidden_dim = 4, 512, 768
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, hidden_dim), dtype=jnp.bfloat16)
        output = mean_pooling(x, chunk_size=64)
        assert output.shape == (batch_size, hidden_dim)

    @pytest.mark.skip(reason="MeanPooling gradient has JAX transformation issues with compiled functions")
    def test_mean_pooling_gradient(self):
        """Test mean pooling gradient computation."""
        batch_size, seq_len, hidden_dim = 2, 256, 512
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, 8, hidden_dim), dtype=jnp.bfloat16)

        def loss_fn(x):
            output = mean_pooling(x)
            return jnp.mean(output)

        grad = jax.grad(loss_fn)(x)
        assert grad.shape == x.shape


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_attention_with_pooling(self):
        """Test attention followed by mean pooling."""
        B, N, H, D = 2, 512, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        attn_output = flash_attention(q, k, v, causal=True)
        assert attn_output.shape == (B, N, H, D)

        pooled = attn_output.reshape(B, N, H * D)
        output = mean_pooling(pooled)
        assert output.shape == (B, H * D)

    def test_multiple_attention_variants(self):
        """Test that different attention variants produce valid outputs."""
        B, N, H, D = 2, 256, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        flash_out = flash_attention(q, k, v, causal=True)
        sdpa_out = scaled_dot_product_attention(q, k, v, causal=True)

        assert flash_out.shape == (B, N, H, D)
        assert sdpa_out.shape == (B, N, H, D)

    @pytest.mark.skip(reason="Gradient through mean_pooling has JAX transformation issues")
    def test_gradient_through_multiple_ops(self):
        """Test gradient computation through multiple operations."""
        B, N, H, D = 2, 256, 8, 128
        q, k, v = rand_tensors(B, N, N, H, H, D, dtype=jnp.bfloat16)

        def loss_fn(q, k, v):
            attn_out = flash_attention(q, k, v, causal=True)
            pooled = attn_out.reshape(B, N, H * D)
            pooled_out = mean_pooling(pooled)
            return jnp.mean(pooled_out)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        assert len(grads) == 3
        assert all(g.shape == t.shape for g, t in zip(grads, [q, k, v], strict=False))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
