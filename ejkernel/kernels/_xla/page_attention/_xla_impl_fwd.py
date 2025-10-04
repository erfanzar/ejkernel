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

    # Handle GQA: num_heads might be > num_kv_heads
    q_heads_per_kv_head = num_heads // num_kv_heads

    # Reshape query for GQA: [num_seqs, num_kv_heads, q_heads_per_kv, head_dim]
    query = query.reshape(num_seqs, num_kv_heads, q_heads_per_kv_head, head_dim)

    # Scale query
    query = query * attn_scale

    def attend_sequence(seq_idx):
        """Compute attention for a single sequence."""
        q = query[seq_idx]  # [num_kv_heads, q_heads_per_kv, head_dim]
        context_len = context_lens[seq_idx]
        blocks = block_tables[seq_idx]  # [max_blocks]

        def attend_block(block_idx):
            """Attend to a single block."""
            physical_block = blocks[block_idx]

            # Get K,V for this block: [num_kv_heads, block_size, head_dim]
            k_block = key_cache[physical_block]
            v_block = value_cache[physical_block]

            # Compute attention scores: [num_kv_heads, q_heads_per_kv, block_size]
            scores = jnp.einsum("ihd,ikd->ihk", q, k_block)

            # Create mask for valid tokens in this block
            block_start = block_idx * block_size
            token_indices = jnp.arange(block_size) + block_start
            valid_mask = token_indices < context_len
            scores = jnp.where(valid_mask[None, None, :], scores, -1e9)

            return scores, v_block

        # Attend to all blocks
        all_scores, all_values = jax.vmap(attend_block)(jnp.arange(max_blocks))
        # all_scores: [max_blocks, num_kv_heads, q_heads_per_kv, block_size]
        # all_values: [max_blocks, num_kv_heads, block_size, head_dim]

        # Reshape scores to merge blocks: [num_kv_heads, q_heads_per_kv, max_blocks * block_size]
        all_scores = all_scores.transpose(1, 2, 0, 3).reshape(num_kv_heads, q_heads_per_kv_head, max_blocks * block_size)

        # Reshape values: [num_kv_heads, max_blocks * block_size, head_dim]
        all_values = all_values.transpose(1, 0, 2, 3).reshape(num_kv_heads, max_blocks * block_size, head_dim)

        # Apply softmax
        attn_weights = jax.nn.softmax(all_scores, axis=-1)  # [num_kv_heads, q_heads_per_kv, max_blocks * block_size]

        # Apply attention weights to values
        output = jnp.einsum("ihk,ikd->ihd", attn_weights, all_values)  # [num_kv_heads, q_heads_per_kv, head_dim]

        # Reshape back to [num_heads, head_dim]
        return output.reshape(num_heads, head_dim)

    # Process all sequences
    output = jax.vmap(attend_sequence)(jnp.arange(num_seqs))

    return output
