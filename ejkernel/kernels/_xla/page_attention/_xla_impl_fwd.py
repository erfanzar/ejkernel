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
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit


@ejit(static_argnums=(6,))
def _page_attention_fwd(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float,
    block_size: int,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """
    Forward pass for page attention using JAX/XLA.

    This implements paged attention where KV cache is stored in blocks (pages).
    Each sequence has a block table that maps logical positions to physical blocks.

    Args:
        query: Query tensor [num_seqs, num_heads, head_dim]
        key_cache: Paged key cache [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: Paged value cache [num_blocks, num_kv_heads, block_size, head_dim]
        context_lens: Length of context for each sequence [num_seqs]
        block_tables: Block table mapping [num_seqs, max_blocks]
        attn_scale: Attention scaling factor
        block_size: Size of each block/page

    Returns:
        Attention output [num_seqs, num_heads, head_dim]
    """
    num_seqs, num_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]
    max_blocks = block_tables.shape[1]

    q_heads_per_kv_head = num_heads // num_kv_heads

    query = query.reshape(num_seqs, num_kv_heads, q_heads_per_kv_head, head_dim)

    query = query * attn_scale

    def attend_sequence(seq_idx):
        """Compute attention for a single sequence."""
        q = query[seq_idx]
        context_len = context_lens[seq_idx]
        blocks = block_tables[seq_idx]

        def attend_block(block_idx):
            """Attend to a single block."""
            physical_block = blocks[block_idx]

            k_block = key_cache[physical_block]
            v_block = value_cache[physical_block]

            scores = jnp.einsum("ihd,ikd->ihk", q, k_block)

            block_start = block_idx * block_size
            token_indices = jnp.arange(block_size) + block_start
            valid_mask = token_indices < context_len
            scores = jnp.where(valid_mask[None, None, :], scores, -1e9)

            return scores, v_block

        all_scores, all_values = jax.vmap(attend_block)(jnp.arange(max_blocks))

        all_scores = all_scores.transpose(1, 2, 0, 3).reshape(num_kv_heads, q_heads_per_kv_head, max_blocks * block_size)

        all_values = all_values.transpose(1, 0, 2, 3).reshape(num_kv_heads, max_blocks * block_size, head_dim)

        attn_weights = jax.nn.softmax(all_scores, axis=-1)

        output = jnp.einsum("ihk,ikd->ihd", attn_weights, all_values)

        return output.reshape(num_heads, head_dim)

    output = jax.vmap(attend_sequence)(jnp.arange(num_seqs))

    return output
