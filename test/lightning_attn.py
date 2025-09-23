# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


import jax.numpy as jnp
import pytest
from einops import rearrange

from ejkernel import lightning_attn
from ejkernel.utils import numeric_gen


def run_recurrent_lightning_test(
    batch_size=4,
    seq_len=2048,
    num_heads=4,
    key_dim=512,
    value_dim=512,
    atol=1e-3,
    rtol=1e-3,
):
    """
    Run a recurrent Lightning Attention test in both batched and flattened (cu_seqlens) mode.
    """
    print(f"\nRunning test: B={batch_size}, T={seq_len}, H={num_heads}, K={key_dim}, V={value_dim}")

    query = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    key = numeric_gen(batch_size, seq_len, num_heads, key_dim)
    value = numeric_gen(batch_size, seq_len, num_heads, value_dim)
    init_state = numeric_gen(batch_size, num_heads, key_dim, value_dim)
    output_batched, state_batched = lightning_attn(
        query,
        key,
        value,
        layer_idx=4,
        num_layers=8,
        initial_state=init_state,
    )
    query_flat, key_flat, value_flat = map(lambda x: rearrange(x, "b t h d -> 1 (b t) h d"), (query, key, value))
    cu_seqlens = jnp.arange(0, (batch_size + 1) * seq_len, seq_len, dtype=jnp.int32)
    output_var, state_var = lightning_attn(
        query_flat,
        key_flat,
        value_flat,
        layer_idx=4,
        num_layers=8,
        initial_state=init_state,
        cu_seqlens=cu_seqlens,
    )

    assert jnp.allclose(output_batched.ravel(), output_var.ravel(), rtol=rtol, atol=atol), "Output tensors differ"
    assert jnp.allclose(state_batched.ravel(), state_var.ravel(), rtol=rtol, atol=atol), "Final states differ"

    print("✅ Passed")


@pytest.mark.parametrize(
    "scenario",
    [
        dict(batch_size=8, seq_len=128, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=1024, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=2048, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=4096, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=8, seq_len=8192, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=4, seq_len=2048, num_heads=4, key_dim=512, value_dim=512),
        dict(batch_size=2, seq_len=1024, num_heads=8, key_dim=256, value_dim=256),
        dict(batch_size=1, seq_len=512, num_heads=2, key_dim=128, value_dim=64),
        dict(batch_size=8, seq_len=128, num_heads=4, key_dim=64, value_dim=64),
        dict(batch_size=2, seq_len=4096, num_heads=1, key_dim=32, value_dim=32),
    ],
)
def test_recurrent_lightning_variants(scenario):
    run_recurrent_lightning_test(**scenario)


if __name__ == "__main__":
    pytest.main([__file__])
