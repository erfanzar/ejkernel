import os

os.environ["EJKERNEL_LOG_AUTOTUNE"] = "1"
import jax
from jax import numpy as jnp

from ejkernel.modules import ragged_decode_attention

total_tokens = 128
H, D = 8, 128
max_kv_len = 512
num_seqs = 4

q = jax.random.normal(jax.random.PRNGKey(0), (num_seqs, H, D), dtype=jnp.bfloat16)

k = jax.random.normal(jax.random.PRNGKey(1), (num_seqs, max_kv_len, H, D), dtype=jnp.bfloat16)
v = jax.random.normal(jax.random.PRNGKey(2), (num_seqs, max_kv_len, H, D), dtype=jnp.bfloat16)

cu_seqlens = jnp.array([0, 8, 18, 32], dtype=jnp.int32)
kv_lengths = cu_seqlens + 1

output = ragged_decode_attention(q, k, v, cu_seqlens, kv_lengths)
