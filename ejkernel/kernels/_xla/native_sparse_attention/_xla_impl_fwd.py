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


@ejit(static_argnums=(5,))
def _sparse_attention_fwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_key_blocks"],
    block_counts: Int[Array, "batch num_kv_heads num_query_blocks"],
    block_size: int,
    scale: float,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    Forward pass for sparse attention with block-based sparsity pattern.
    Supports Grouped Query Attention (GQA) where num_q_heads >= num_kv_heads.

    Args:
        q: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        v: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        block_indices: Sparsity pattern [batch, num_kv_heads, num_query_blocks, num_key_blocks]
            Specifies which key blocks each query block attends to
        block_counts: Number of key blocks to attend to per query block
            Can be int (uniform) or tensor [batch, num_kv_heads, num_query_blocks]
        block_size: Size of each block
        scale: Attention scaling factor

    Returns:
        Attention output [batch, seq_len, num_q_heads, head_dim]
    """
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group_size = num_q_heads // num_kv_heads
    num_query_blocks = (seq_len + block_size - 1) // block_size
    num_key_blocks = (seq_len + block_size - 1) // block_size

    # Pad sequences to be divisible by block_size
    pad_len = num_query_blocks * block_size - seq_len
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    # Reshape into blocks
    # q: [batch, num_query_blocks, block_size, num_q_heads, head_dim]
    # k,v: [batch, num_key_blocks, block_size, num_kv_heads, head_dim]
    q_blocks = q.reshape(batch, num_query_blocks, block_size, num_q_heads, head_dim)
    k_blocks = k.reshape(batch, num_key_blocks, block_size, num_kv_heads, head_dim)
    v_blocks = v.reshape(batch, num_key_blocks, block_size, num_kv_heads, head_dim)

    # Transpose to [batch, num_heads, num_blocks, block_size, head_dim]
    q_blocks = q_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_q_heads, num_query_blocks, block_size, head_dim]
    k_blocks = k_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_kv_heads, num_key_blocks, block_size, head_dim]
    v_blocks = v_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_kv_heads, num_key_blocks, block_size, head_dim]

    def attend_query_block(b_idx, q_h_idx, qb_idx):
        """Compute attention for a single query block."""
        # For GQA: map query head to corresponding KV head
        kv_h_idx = q_h_idx // group_size

        q_block = q_blocks[b_idx, q_h_idx, qb_idx]  # [block_size, head_dim]

        # Get which key blocks to attend to (indexed by kv head)
        num_blocks_for_this_query = block_counts[b_idx, kv_h_idx, qb_idx]
        all_key_block_indices = block_indices[b_idx, kv_h_idx, qb_idx]  # [max_num_key_blocks]

        def attend_key_block(kb_pos):
            """Attend to a single key block position."""
            kb_idx = all_key_block_indices[kb_pos]
            k_block = k_blocks[b_idx, kv_h_idx, kb_idx]  # [block_size, head_dim]
            v_block = v_blocks[b_idx, kv_h_idx, kb_idx]  # [block_size, head_dim]

            # Compute attention scores: [block_size, block_size]
            scores = jnp.einsum("qd,kd->qk", q_block, k_block) * scale

            # Mask out this key block if beyond num_blocks
            is_valid = kb_pos < num_blocks_for_this_query
            scores = jnp.where(is_valid, scores, -1e9)

            return scores, v_block

        # Process all potential key block positions
        max_num_key_blocks = all_key_block_indices.shape[0]
        scores_list, v_list = jax.vmap(attend_key_block)(jnp.arange(max_num_key_blocks))
        # scores_list: [max_num_key_blocks, block_size, block_size]
        # v_list: [max_num_key_blocks, block_size, head_dim]

        # Reshape to merge key blocks: [block_size, max_num_key_blocks * block_size]
        all_scores = scores_list.transpose(1, 0, 2).reshape(block_size, -1)
        all_values = v_list.transpose(1, 0, 2).reshape(-1, head_dim)  # [max_num_key_blocks * block_size, head_dim]

        # Apply softmax
        attn_weights = jax.nn.softmax(all_scores, axis=-1)  # [block_size, max_num_key_blocks * block_size]

        # Apply attention weights
        output = jnp.einsum("qk,kd->qd", attn_weights, all_values)  # [block_size, head_dim]

        return output

    # Process all query blocks
    def process_head(b_idx, q_h_idx):
        outputs = jax.vmap(lambda qb: attend_query_block(b_idx, q_h_idx, qb))(jnp.arange(num_query_blocks))
        return outputs  # [num_query_blocks, block_size, head_dim]

    def process_batch(b_idx):
        outputs = jax.vmap(lambda h: process_head(b_idx, h))(jnp.arange(num_q_heads))
        return outputs  # [num_q_heads, num_query_blocks, block_size, head_dim]

    # Process all batches
    outputs = jax.vmap(process_batch)(jnp.arange(batch))
    # outputs: [batch, num_q_heads, num_query_blocks, block_size, head_dim]

    # Reshape back to [batch, seq_len, num_q_heads, head_dim]
    outputs = outputs.transpose(0, 2, 3, 1, 4).reshape(batch, -1, num_q_heads, head_dim)

    # Remove padding
    outputs = outputs[:, :seq_len, :, :]

    return outputs
