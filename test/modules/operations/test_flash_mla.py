from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ejkernel.modules.operations import mla_attention

from ._utils import device_platform, has_triton


@pytest.mark.skipif(device_platform() != "gpu" or not has_triton(), reason="FlashMLA is Triton/GPU only")
def test_flash_mla_basic_runs_on_gpu():
    b, t, q_heads, kv_heads, head_dim, kv_lora_rank = 1, 8, 4, 2, 16, 8
    query = jax.random.normal(jax.random.PRNGKey(0), (b, t, q_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    key_value = jax.random.normal(jax.random.PRNGKey(1), (b, t, kv_lora_rank), dtype=jnp.float32).astype(jnp.bfloat16)
    w_kc = jax.random.normal(jax.random.PRNGKey(2), (kv_lora_rank, kv_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)
    w_vc = jax.random.normal(jax.random.PRNGKey(3), (kv_lora_rank, kv_heads, head_dim), dtype=jnp.float32).astype(jnp.bfloat16)

    out = mla_attention(query, key_value, w_kc, w_vc, causal=True, platform="triton")
    assert out.shape == (b, t, q_heads, head_dim)

