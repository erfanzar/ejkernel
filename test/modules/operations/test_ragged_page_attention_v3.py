from __future__ import annotations

import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import ragged_page_attention_v3
from ejkernel.utils import make_dummy_rpa_inputs

from ._utils import assert_allclose, device_platform


def test_ragged_page_attention_v3_shapes_and_attention_sink_runs():
    batch = make_dummy_rpa_inputs(
        rng_seed=0,
        num_seqs=2,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=8,
    )

    kv_cache0 = batch["kv_cache"].copy()
    out, kv_cache_out = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache0,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        None,
        softmax_scale=128**-0.5,
        platform="xla",
    )
    assert out.shape == batch["queries"].shape
    assert kv_cache_out.shape == batch["kv_cache"].shape

    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    out2, kv_cache_out2 = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_out,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="xla",
    )
    assert out2.shape == batch["queries"].shape
    assert kv_cache_out2.shape == batch["kv_cache"].shape


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_ragged_page_attention_v3_pallas_matches_xla_on_tpu():
    batch = make_dummy_rpa_inputs(
        rng_seed=1,
        num_seqs=2,
        pages_per_seq=2,
        page_size=16,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=128,
        kv_dtype=jnp.bfloat16,
        total_q=8,
    )
    sinks = jnp.zeros((batch["queries"].shape[1],), dtype=jnp.float32)
    kv_cache_xla = batch["kv_cache"].copy()
    kv_cache_pallas = batch["kv_cache"].copy()

    out_xla, cache_xla = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_xla,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="xla",
    )
    out_pallas, cache_pallas = ragged_page_attention_v3(
        batch["queries"],
        batch["keys"],
        batch["values"],
        kv_cache_pallas,
        batch["kv_lens"],
        batch["block_tables"],
        batch["query_start_loc"],
        batch["distribution"],
        sinks,
        softmax_scale=128**-0.5,
        sliding_window=16,
        logits_soft_cap=10.0,
        platform="pallas",
    )

    assert_allclose(out_pallas, out_xla, atol=0.35)
    assert_allclose(cache_pallas, cache_xla, atol=0.35)
