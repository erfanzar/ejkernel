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

from __future__ import annotations

import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if importlib.util.find_spec("cutlass") is None:
    pytest.skip("CUTLASS CuTe DSL not installed", allow_module_level=True)
if jax.devices()[0].platform != "gpu":
    pytest.skip("CUTE tests require GPU backend", allow_module_level=True)

from ejkernel.kernels._cute.ragged_page_attention_v3 import ragged_page_attention_v3 as cute_ragged_page_attention_v3
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3 as xla_ragged_page_attention_v3


def _kv_cache_shape(
    *,
    num_pages: int,
    page_size: int,
    num_kv_heads: int,
    head_dim_padded: int,
    dtype: jnp.dtype,
) -> tuple[int, ...]:
    pack = 2 if jnp.dtype(dtype).itemsize == 2 else 1
    if pack == 1:
        return (num_pages, page_size, num_kv_heads * 2, 1, head_dim_padded)
    return (num_pages, page_size, num_kv_heads, 2, head_dim_padded)


def test_ragged_page_attention_v3_cute_matches_xla():
    key = jax.random.PRNGKey(0)
    kq, kk, kv, kc = jax.random.split(key, 4)

    num_seqs = 3
    max_num_seqs = 3
    pages_per_seq = 4
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 4
    head_dim = 64
    head_dim_padded = 128
    dtype = jnp.float16

    kv_lens = jnp.array([40, 32, 48], dtype=jnp.int32)
    q_lens = jnp.array([8, 4, 6], dtype=jnp.int32)

    total_q = int(jnp.sum(q_lens))
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0)).astype(jnp.int32)

    queries = jax.random.normal(kq, (total_q, num_q_heads, head_dim), dtype=dtype)
    keys = jax.random.normal(kk, (total_q, num_kv_heads, head_dim), dtype=dtype)
    values = jax.random.normal(kv, (total_q, num_kv_heads, head_dim), dtype=dtype)

    total_pages = max_num_seqs * pages_per_seq
    kv_cache = jax.random.normal(
        kc,
        _kv_cache_shape(
            num_pages=total_pages,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim_padded=head_dim_padded,
            dtype=dtype,
        ),
        dtype=dtype,
    )

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    kv_cache_xla = kv_cache.copy()
    kv_cache_cute = kv_cache.copy()

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_xla,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=head_dim**-0.5,
    )
    out_cute, cache_cute = cute_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_cute,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_scale=head_dim**-0.5,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    cache_cute = jax.block_until_ready(cache_cute)
    cache_xla = jax.block_until_ready(cache_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=2e-2,
        atol=2e-2,
    )
    np.testing.assert_allclose(
        np.asarray(cache_cute),
        np.asarray(cache_xla),
        rtol=0.0,
        atol=0.0,
    )


def test_ragged_page_attention_v3_cute_matches_xla_with_optional_features():
    key = jax.random.PRNGKey(13)
    kq, kk, kv, kc, ka = jax.random.split(key, 5)

    num_seqs = 2
    max_num_seqs = 2
    pages_per_seq = 8
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    head_dim_padded = 128
    dtype = jnp.bfloat16

    kv_lens = jnp.array([80, 64], dtype=jnp.int32)
    q_lens = jnp.array([16, 8], dtype=jnp.int32)

    total_q = int(jnp.sum(q_lens))
    query_start_loc = jnp.pad(jnp.cumsum(q_lens), (1, 0)).astype(jnp.int32)

    queries = jax.random.normal(kq, (total_q, num_q_heads, head_dim), dtype=dtype)
    keys = jax.random.normal(kk, (total_q, num_kv_heads, head_dim), dtype=dtype)
    values = jax.random.normal(kv, (total_q, num_kv_heads, head_dim), dtype=dtype)

    total_pages = max_num_seqs * pages_per_seq
    kv_cache = jax.random.normal(
        kc,
        _kv_cache_shape(
            num_pages=total_pages,
            page_size=page_size,
            num_kv_heads=num_kv_heads,
            head_dim_padded=head_dim_padded,
            dtype=dtype,
        ),
        dtype=dtype,
    )

    block_tables = jnp.arange(total_pages, dtype=jnp.int32)
    distribution = jnp.array([0, 0, num_seqs], dtype=jnp.int32)
    softmax_aux = jax.random.normal(ka, (num_q_heads,), dtype=jnp.float32)
    kv_cache_xla = kv_cache.copy()
    kv_cache_cute = kv_cache.copy()

    kwargs = dict(
        softmax_scale=head_dim**-0.5,
        sliding_window=32,
        logits_soft_cap=20.0,
        q_scale=1.0,
        k_scale=0.5,
        v_scale=1.2,
    )

    out_xla, cache_xla = xla_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_xla,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        **kwargs,
    )
    out_cute, cache_cute = cute_ragged_page_attention_v3(
        queries,
        keys,
        values,
        kv_cache_cute,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        **kwargs,
    )

    out_cute = jax.block_until_ready(out_cute)
    out_xla = jax.block_until_ready(out_xla)
    cache_cute = jax.block_until_ready(cache_cute)
    cache_xla = jax.block_until_ready(cache_xla)

    np.testing.assert_allclose(
        np.asarray(out_cute, dtype=np.float32),
        np.asarray(out_xla, dtype=np.float32),
        rtol=4e-2,
        atol=4e-2,
    )
    np.testing.assert_allclose(
        np.asarray(cache_cute),
        np.asarray(cache_xla),
        rtol=0.0,
        atol=0.0,
    )
