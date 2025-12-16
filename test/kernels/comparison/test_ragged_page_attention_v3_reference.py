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

from ejkernel.kernels._pallas.tpu.ragged_page_attention_v3._pallas_impl_fwd import ref_ragged_paged_attention
from ejkernel.kernels._triton.ragged_page_attention_v3 import ragged_page_attention_v3 as triton_rpa_v3
from ejkernel.kernels._xla.ragged_page_attention_v3 import ragged_page_attention_v3 as xla_rpa_v3
from ejkernel.utils import make_dummy_rpa_inputs


def _has_gpu() -> bool:
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="Ragged page attention v3 Triton tests require GPU backend")


def _as_f32(x: jax.Array) -> jax.Array:
    return x.astype(jnp.float32) if x.dtype != jnp.float32 else x


def test_ragged_page_attention_v3_matches_reference():
    batch = make_dummy_rpa_inputs(
        rng_seed=0,
        num_seqs=8,
        pages_per_seq=8,
        page_size=16,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        q_dtype=None,
        kv_len_max=64,
        total_q=None,
        total_num_pages=None,
        decode_prefill_mixed=None,
    )

    softmax_scale = 1.0

    ref_out, ref_cache = ref_ragged_paged_attention(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=softmax_scale,
    )

    out_xla, cache_xla = xla_rpa_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=softmax_scale,
    )

    out_triton, cache_triton = triton_rpa_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=softmax_scale,
    )

    # Eager reference uses the same math but will differ slightly for bf16 outputs.
    assert jnp.allclose(_as_f32(out_xla), _as_f32(ref_out), rtol=0, atol=0.125)
    assert jnp.allclose(_as_f32(out_triton), _as_f32(ref_out), rtol=0, atol=0.125)

    assert jnp.allclose(cache_xla, ref_cache, rtol=0, atol=0)
    assert jnp.allclose(cache_triton, ref_cache, rtol=0, atol=0)


def test_ragged_page_attention_v3_xla_outer_jit_matches_reference():
    batch = make_dummy_rpa_inputs(
        rng_seed=1,
        num_seqs=8,
        pages_per_seq=8,
        page_size=16,
        num_q_heads=8,
        num_kv_heads=4,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        q_dtype=None,
        kv_len_max=64,
        total_q=None,
        total_num_pages=None,
        decode_prefill_mixed=None,
    )
    softmax_scale = 1.0

    ref_out, _ = ref_ragged_paged_attention(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"],
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        softmax_scale=softmax_scale,
    )

    @jax.jit
    def _wrapped(q, k, v, cache, kv_lens, block_tables, qsl, dist):
        return xla_rpa_v3(
            q,
            k,
            v,
            cache,
            kv_lens,
            block_tables,
            qsl,
            dist,
            softmax_scale=softmax_scale,
        )

    out_xla, _ = _wrapped(
        batch["queries"],
        batch["keys"],
        batch["values"],
        batch["kv_cache"].copy(),
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
    )

    assert jnp.allclose(_as_f32(out_xla), _as_f32(ref_out), rtol=0, atol=0.125)

