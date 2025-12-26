from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import flash_attention
from ejkernel.types import MaskInfo

from ._utils import assert_allclose, dense_attention_reference, device_platform, rand_qkv


def test_flash_attention_matches_dense_reference_with_bias():
    key = jax.random.PRNGKey(0)
    q, k, v = rand_qkv(key, batch=2, q_len=8, kv_len=10, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)
    bias = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 8, 10), dtype=jnp.float32).astype(jnp.bfloat16)

    out = flash_attention(q, k, v, bias, softmax_scale=32**-0.5, platform="xla")
    ref_out, _ = dense_attention_reference(q, k, v, bias=bias, softmax_scale=32**-0.5)
    assert out.shape == (2, 8, 4, 32)
    assert_allclose(out, ref_out, atol=0.15)


def test_flash_attention_segments_sliding_window_logits_soft_cap_and_softmax_aux_match_reference_xla():
    key = jax.random.PRNGKey(2)
    q, k, v = rand_qkv(key, batch=2, q_len=12, kv_len=12, q_heads=4, kv_heads=2, head_dim=16, dtype=jnp.bfloat16)

    seg = jnp.array(
        [
            [0] * 4 + [1] * 4 + [-1] * 4,
            [0] * 6 + [1] * 3 + [-1] * 3,
        ],
        dtype=jnp.int32,
    )
    mask_info = MaskInfo.from_segments(q_segment_ids=seg)

    softmax_aux = jax.random.normal(jax.random.PRNGKey(3), (2,), dtype=jnp.float32).astype(jnp.bfloat16)
    sliding_window = (5, 1)
    logits_soft_cap = 6.0
    scale = 16**-0.5

    out = flash_attention(
        q,
        k,
        v,
        None,
        None,
        None,
        softmax_aux,
        mask_info=mask_info,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        platform="xla",
    )

    seg_mask = (seg[:, None, :, None] == seg[:, None, None, :]) & (seg[:, None, :, None] >= 0)
    ref_out, _ = dense_attention_reference(
        q,
        k,
        v,
        attention_mask=seg_mask,
        causal=True,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_scale=scale,
        softmax_aux=softmax_aux,
    )

    assert_allclose(out, ref_out, atol=0.15)


def test_flash_attention_dropout_seed_changes_output():
    key = jax.random.PRNGKey(4)
    q, k, v = rand_qkv(key, batch=1, q_len=16, kv_len=16, q_heads=4, kv_heads=4, head_dim=32, dtype=jnp.bfloat16)

    out0 = flash_attention(q, k, v, dropout_prob=0.25, dropout_seed=0, platform="xla")
    out1 = flash_attention(q, k, v, dropout_prob=0.25, dropout_seed=1, platform="xla")

    assert out0.shape == q.shape
    assert out1.shape == q.shape
    assert not jnp.allclose(out0, out1)


@pytest.mark.skipif(device_platform() != "tpu", reason="TPU-only cross-backend comparison (pallas vs xla)")
def test_flash_attention_pallas_matches_xla_on_tpu_with_sliding_window_and_soft_cap():
    key = jax.random.PRNGKey(5)
    q, k, v = rand_qkv(key, batch=2, q_len=128, kv_len=128, q_heads=4, kv_heads=2, head_dim=32, dtype=jnp.bfloat16)

    out_xla = flash_attention(q, k, v, sliding_window=(64, 0), logits_soft_cap=10.0, causal=True, platform="xla")
    out_pallas = flash_attention(q, k, v, sliding_window=(64, 0), logits_soft_cap=10.0, causal=True, platform="pallas")

    assert_allclose(out_pallas, out_xla, atol=0.2)
