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

from ejkernel.kernels._pallas.tpu.ragged_page_attention import ragged_page_attention
from ejkernel.kernels._xla.ragged_page_attention import ragged_page_attention as ragged_page_attention_xla

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


class TestRaggedPageAttentionTPU:
    """Test suite for TPU Pallas ragged page attention implementation."""

    def test_basic_forward_shape_and_finite(self):
        """Test basic forward pass with simple configuration."""
        total_tokens = 256
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_seqs = 4
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(42)
        key, k1, k2, k3, k4 = jax.random.split(key, 5)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)
        context_lens = jax.random.randint(k3, (num_seqs,), 32, 128, dtype=jnp.int32)
        block_tables = jax.random.randint(k4, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)

        tokens_per_seq = total_tokens // num_seqs
        query_start_loc = jnp.arange(0, (num_seqs + 1) * tokens_per_seq, tokens_per_seq, dtype=jnp.int32)

        output = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=32,
        )

        assert output.shape == (total_tokens, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()
        assert output.dtype == queries.dtype

    def test_mixed_prefill_and_decode(self):
        """Test mixed prefill and decode scenarios."""

        total_tokens = 200
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 64
        pages_per_seq = 16
        num_seqs = 4
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(123)
        key, k1, k2, _k3, k4 = jax.random.split(key, 5)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([128, 64, 192, 96], dtype=jnp.int32)
        block_tables = jax.random.randint(k4, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)

        query_start_loc = jnp.array([0, 100, 150, 190, 200], dtype=jnp.int32)

        output = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=32,
        )

        assert output.shape == (total_tokens, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_single_sequence(self):
        """Test with a single sequence."""
        total_tokens = 128
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 16
        pages_per_seq = 8
        num_seqs = 1
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(456)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([100], dtype=jnp.int32)
        block_tables = jnp.arange(pages_per_seq, dtype=jnp.int32).reshape(1, -1)
        query_start_loc = jnp.array([0, total_tokens], dtype=jnp.int32)

        output = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=32,
        )

        assert output.shape == (total_tokens, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_gqa_with_different_ratios(self):
        """Test Grouped Query Attention with various head ratios."""
        total_tokens = 256
        page_size = 16
        num_pages = 32
        pages_per_seq = 4
        num_seqs = 2
        head_dim = 128

        configs = [
            (8, 2),
            (16, 4),
            (32, 8),
        ]

        for num_q_heads, num_kv_heads in configs:
            num_combined_kv_heads = num_kv_heads * 2
            key = jax.random.PRNGKey(789 + num_q_heads)
            key, k1, k2 = jax.random.split(key, 3)

            queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
            kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

            context_lens = jnp.array([48, 60], dtype=jnp.int32)
            block_tables = jnp.tile(jnp.arange(pages_per_seq, dtype=jnp.int32), (num_seqs, 1))
            query_start_loc = jnp.array([0, 128, 256], dtype=jnp.int32)

            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                num_kv_pages_per_block=2,
                num_queries_per_block=32,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_different_page_sizes(self):
        """Test with different page sizes."""
        total_tokens = 256
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        num_seqs = 2
        num_combined_kv_heads = num_kv_heads * 2

        page_configs = [
            (8, 64, 16),
            (16, 32, 8),
            (32, 16, 4),
        ]

        for page_size, num_pages, pages_per_seq in page_configs:
            key = jax.random.PRNGKey(1000 + page_size)
            key, k1, k2 = jax.random.split(key, 3)

            queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
            kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

            context_lens = jnp.array([64, 96], dtype=jnp.int32)
            block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
            query_start_loc = jnp.array([0, 128, 256], dtype=jnp.int32)

            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                num_kv_pages_per_block=2,
                num_queries_per_block=32,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_dtypes(self):
        """Test with different dtypes."""
        total_tokens = 256
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_seqs = 2
        num_combined_kv_heads = num_kv_heads * 2

        for dtype in [jnp.float32, jnp.bfloat16]:
            key = jax.random.PRNGKey(2000)
            key, k1, k2 = jax.random.split(key, 3)

            queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=dtype)
            kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=dtype)

            context_lens = jnp.array([64, 96], dtype=jnp.int32)
            block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
            query_start_loc = jnp.array([0, 128, 256], dtype=jnp.int32)

            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                num_kv_pages_per_block=2,
                num_queries_per_block=32,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert output.dtype == dtype
            assert jnp.isfinite(output).all()

    def test_custom_softmax_scale(self):
        """Test with custom softmax scale."""
        total_tokens = 256
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_seqs = 2
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(3000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([64, 96], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
        query_start_loc = jnp.array([0, 128, 256], dtype=jnp.int32)

        for scale in [0.1, 0.5, 1.0]:
            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                softmax_scale=scale,
                num_kv_pages_per_block=2,
                num_queries_per_block=32,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_different_block_configurations(self):
        """Test with different kernel block configurations."""
        total_tokens = 256
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_seqs = 2
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(4000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([64, 96], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
        query_start_loc = jnp.array([0, 128, 256], dtype=jnp.int32)

        block_configs = [
            (2, 32),
            (4, 64),
            (1, 16),
        ]

        for num_kv_pages_per_block, num_queries_per_block in block_configs:
            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                num_kv_pages_per_block=num_kv_pages_per_block,
                num_queries_per_block=num_queries_per_block,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_varying_sequence_counts(self):
        """Test with different numbers of sequences."""
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 64
        pages_per_seq = 8
        num_combined_kv_heads = num_kv_heads * 2

        for num_seqs in [1, 2, 4, 8]:
            total_tokens = num_seqs * 64

            key = jax.random.PRNGKey(5000 + num_seqs)
            key, k1, k2 = jax.random.split(key, 3)

            queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
            kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

            context_lens = jax.random.randint(key, (num_seqs,), 32, 100, dtype=jnp.int32)
            block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
            query_start_loc = jnp.arange(0, total_tokens + 1, 64, dtype=jnp.int32)

            output = ragged_page_attention(
                queries=queries,
                kv_pages=kv_pages,
                context_lens=context_lens,
                block_tables=block_tables,
                query_start_loc=query_start_loc,
                num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
                num_kv_pages_per_block=2,
                num_queries_per_block=32,
            )

            assert output.shape == (total_tokens, num_q_heads, head_dim)
            assert jnp.isfinite(output).all()

    def test_decode_only_scenario(self):
        """Test decode-only scenario (1 token per sequence)."""
        num_seqs = 8
        total_tokens = num_seqs
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(6000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([64, 80, 96, 48, 72, 88, 56, 100], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)

        query_start_loc = jnp.arange(0, num_seqs + 1, dtype=jnp.int32)

        output = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=4,
        )

        assert output.shape == (total_tokens, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_prefill_only_scenario(self):
        """Test prefill-only scenario (many tokens per sequence)."""
        num_seqs = 2
        tokens_per_seq = 256
        total_tokens = num_seqs * tokens_per_seq
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 64
        pages_per_seq = 16
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(7000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([tokens_per_seq, tokens_per_seq], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
        query_start_loc = jnp.array([0, tokens_per_seq, total_tokens], dtype=jnp.int32)

        output = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=4,
            num_queries_per_block=64,
        )

        assert output.shape == (total_tokens, num_q_heads, head_dim)
        assert jnp.isfinite(output).all()

    def test_numerical_correctness_vs_xla_simple(self):
        """Test numerical correctness against XLA reference implementation."""
        total_tokens = 128
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 32
        pages_per_seq = 8
        num_seqs = 2
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(8000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([64, 96], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)
        query_start_loc = jnp.array([0, 64, 128], dtype=jnp.int32)

        output_tpu = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=32,
        )

        output_xla = ragged_page_attention_xla(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-2, atol=1e-2)

    def test_numerical_correctness_vs_xla_mixed(self):
        """Test numerical correctness against XLA for mixed prefill/decode."""
        total_tokens = 200
        num_q_heads = 8
        num_kv_heads = 2
        head_dim = 128
        page_size = 16
        num_pages = 64
        pages_per_seq = 16
        num_seqs = 4
        num_combined_kv_heads = num_kv_heads * 2

        key = jax.random.PRNGKey(9000)
        key, k1, k2 = jax.random.split(key, 3)

        queries = jax.random.normal(k1, (total_tokens, num_q_heads, head_dim), dtype=jnp.float32)
        kv_pages = jax.random.normal(k2, (num_pages, page_size, num_combined_kv_heads, head_dim), dtype=jnp.float32)

        context_lens = jnp.array([128, 64, 192, 96], dtype=jnp.int32)
        block_tables = jax.random.randint(key, (num_seqs, pages_per_seq), 0, num_pages, dtype=jnp.int32)

        query_start_loc = jnp.array([0, 100, 150, 190, 200], dtype=jnp.int32)

        output_tpu = ragged_page_attention(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
            num_kv_pages_per_block=2,
            num_queries_per_block=32,
        )

        output_xla = ragged_page_attention_xla(
            queries=queries,
            kv_pages=kv_pages,
            context_lens=context_lens,
            block_tables=block_tables,
            query_start_loc=query_start_loc,
            num_seqs=jnp.array([num_seqs], dtype=jnp.int32),
        )

        assert output_tpu.shape == output_xla.shape
        assert jnp.allclose(output_tpu, output_xla, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
