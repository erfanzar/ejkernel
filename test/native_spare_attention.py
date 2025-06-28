# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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

import jax
import jax.numpy as jnp

from ejgpu import native_spare_attention
from ejgpu.utils import cdiv, numeric_gen


def generate_block_indices(batch_size, sequence_length, kv_heads, num_blocks_per_token, block_size, seed=0):
    """
    Generate sparse attention block indices.
    """
    block_indices = jnp.full(
        (batch_size, sequence_length, kv_heads, num_blocks_per_token), sequence_length, dtype=jnp.int32
    )
    key = jax.random.PRNGKey(seed)

    for b in range(batch_size):
        for t in range(sequence_length):
            for h in range(kv_heads):
                key, subkey = jax.random.split(key)
                max_blocks = max(1, cdiv(t, block_size))
                selected_blocks = jax.random.permutation(subkey, max_blocks)[:num_blocks_per_token]
                block_indices = block_indices.at[b, t, h, : len(selected_blocks)].set(selected_blocks)

    return jnp.sort(block_indices, axis=-1)


def run_attention_test(
    batch_size=1,
    kv_heads=2,
    query_heads=64,
    sequence_length=256,
    head_dim=64,
    num_blocks_per_token=16,
    block_size=32,
    scale=0.1,
    dtype=jnp.float16,
    seed=42,
    verbose=True,
):
    """
    Run a native sparse attention test.
    """
    if verbose:
        print("\nRunning test with:")
        print(f"  Batch size:         {batch_size}")
        print(f"  KV heads:           {kv_heads}")
        print(f"  Query heads:        {query_heads}")
        print(f"  Sequence length:    {sequence_length}")
        print(f"  Head dimension:     {head_dim}")
        print(f"  Blocks per token:   {num_blocks_per_token}")
        print(f"  Block size:         {block_size}")
        print(f"  Scale:              {scale}")
        print(f"  Dtype:              {dtype}")

    query = numeric_gen(batch_size, sequence_length, query_heads, head_dim, dtype=dtype)
    key = numeric_gen(batch_size, sequence_length, kv_heads, head_dim, dtype=dtype)
    value = numeric_gen(batch_size, sequence_length, kv_heads, head_dim, dtype=dtype)
    g_cmp = None
    block_indices = generate_block_indices(
        batch_size,
        sequence_length,
        kv_heads,
        num_blocks_per_token,
        block_size,
        seed=seed,
    )

    output = native_spare_attention(
        q=query,
        k=key,
        v=value,
        block_indices=block_indices,
        block_size=block_size,
        scale=scale,
        g_cmp=g_cmp,
    )

    if verbose:
        print("  Output shape:", output.shape)
        print("  Output sample:", output[0, 0, 0, :5])

    return output


def main():
    """
    Run multiple test scenarios for sparse attention.
    """
    test_scenarios = [
        dict(
            batch_size=1,
            kv_heads=1,
            query_heads=16,
            sequence_length=32,
            head_dim=16,
            num_blocks_per_token=2,
            block_size=8,
            scale=1.0,
        ),
        dict(
            batch_size=2,
            kv_heads=2,
            query_heads=64,
            sequence_length=128,
            head_dim=32,
            num_blocks_per_token=8,
            block_size=16,
            scale=0.2,
        ),
        dict(
            batch_size=1,
            kv_heads=4,
            query_heads=128,
            sequence_length=64,
            head_dim=64,
            num_blocks_per_token=4,
            block_size=16,
            scale=0.15,
        ),
        dict(
            batch_size=2,
            kv_heads=2,
            query_heads=64,
            sequence_length=512,
            head_dim=128,
            num_blocks_per_token=32,
            block_size=64,
            scale=0.05,
        ),
        dict(
            batch_size=1,
            kv_heads=8,
            query_heads=128,
            sequence_length=128,
            head_dim=32,
            num_blocks_per_token=8,
            block_size=32,
            scale=0.1,
        ),
        dict(
            batch_size=1,
            kv_heads=4,
            query_heads=64,
            sequence_length=64,
            head_dim=256,
            num_blocks_per_token=4,
            block_size=16,
            scale=0.2,
        ),
        dict(
            batch_size=1,
            kv_heads=4,
            query_heads=64,
            sequence_length=256,
            head_dim=64,
            num_blocks_per_token=64,
            block_size=32,
            scale=0.05,
        ),
        dict(
            batch_size=8,
            kv_heads=2,
            query_heads=32,
            sequence_length=64,
            head_dim=32,
            num_blocks_per_token=8,
            block_size=16,
            scale=0.3,
        ),
        dict(
            batch_size=1,
            kv_heads=1,
            query_heads=16,
            sequence_length=16,
            head_dim=8,
            num_blocks_per_token=1,
            block_size=8,
            scale=0.9,
        ),
        dict(
            batch_size=2,
            kv_heads=2,
            query_heads=64,
            sequence_length=1024,
            head_dim=64,
            num_blocks_per_token=32,
            block_size=64,
            scale=0.1,
        ),
    ]

    for i, scenario in enumerate(test_scenarios):
        print(f"\n=== Test Scenario {i + 1} ===")
        try:
            run_attention_test(**scenario)
        except Exception as e:
            print(f"❌ Scenario {i + 1} failed: {e}")


if __name__ == "__main__":
    main()
