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
