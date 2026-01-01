from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.modules.operations import flash_mla


def test_flash_mla_basic_runs_on_xla():
    b, t, q_heads, kv_heads, head_dim = 1, 8, 4, 2, 16
    kv_lora_rank = 8
    rope_dim = 4
    d_nope = head_dim - rope_dim
    key = jax.random.PRNGKey(0)
    key, kq, kkv, kwk, kwv, kbk = jax.random.split(key, 6)

    query = jax.random.normal(kq, (b, t, q_heads, head_dim), dtype=jnp.float32)
    key_value = jax.random.normal(kkv, (b, t, kv_lora_rank), dtype=jnp.float32)
    w_kc = jax.random.normal(kwk, (kv_lora_rank, kv_heads, d_nope), dtype=jnp.float32)
    w_vc = jax.random.normal(kwv, (kv_lora_rank, kv_heads, head_dim), dtype=jnp.float32)
    b_k = jax.random.normal(kbk, (b, t, rope_dim), dtype=jnp.float32)

    out = flash_mla(query, key_value, w_kc, w_vc, None, b_k, causal=True, platform="xla")
    assert out.shape == (b, t, q_heads, head_dim)
    assert jnp.isfinite(out).all()
