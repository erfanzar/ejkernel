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
def _sparse_attention_bwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_key_blocks"],
    block_counts: Int[Array, "batch num_kv_heads num_query_blocks"],
    block_size: int,
    scale: float,
    do: Float[Array, "batch seq_len num_q_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len num_q_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
]:
    """
    Backward pass for sparse attention with GQA support.

    Args:
        q: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        v: Value tensor [batch, seq_len, num_kv_heads, head_dim]
        block_indices: Sparsity pattern [batch, num_kv_heads, num_query_blocks, num_key_blocks]
        block_counts: Number of valid blocks per query block
        block_size: Size of each block
        scale: Attention scale factor
        do: Gradient of output [batch, seq_len, num_q_heads, head_dim]

    Returns:
        Tuple of (dq, dk, dv)
    """
    batch, seq_len, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[2]
    group_size = num_q_heads // num_kv_heads
    num_query_blocks = seq_len // block_size

    # Reshape into blocks
    q_blocks = q.reshape(batch, num_query_blocks, block_size, num_q_heads, head_dim)
    k_blocks = k.reshape(batch, num_query_blocks, block_size, num_kv_heads, head_dim)
    v_blocks = v.reshape(batch, num_query_blocks, block_size, num_kv_heads, head_dim)
    do_blocks = do.reshape(batch, num_query_blocks, block_size, num_q_heads, head_dim)

    # Transpose to [batch, num_heads, num_blocks, block_size, head_dim]
    q_blocks = q_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_q_heads, num_query_blocks, block_size, head_dim]
    k_blocks = k_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_kv_heads, num_query_blocks, block_size, head_dim]
    v_blocks = v_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_kv_heads, num_query_blocks, block_size, head_dim]
    do_blocks = do_blocks.transpose(0, 3, 1, 2, 4)  # [batch, num_q_heads, num_query_blocks, block_size, head_dim]

    # Initialize gradient accumulators
    # dq_blocks = jnp.zeros_like(q_blocks)
    # dk_blocks = jnp.zeros_like(k_blocks)
    # dv_blocks = jnp.zeros_like(v_blocks)

    def backward_query_block(b_idx, q_h_idx, qb_idx):
        """Compute gradients for a single query block."""
        # Map query head to KV head for GQA
        kv_h_idx = q_h_idx // group_size

        q_block = q_blocks[b_idx, q_h_idx, qb_idx]  # [block_size, head_dim]
        do_block = do_blocks[b_idx, q_h_idx, qb_idx]  # [block_size, head_dim]

        # Get which key blocks to attend to (indexed by kv head)
        num_blocks_for_this_query = block_counts[b_idx, kv_h_idx, qb_idx]
        all_key_block_indices = block_indices[b_idx, kv_h_idx, qb_idx]

        def backward_key_block(kb_pos):
            """Process backward for a single key block position."""
            kb_idx = all_key_block_indices[kb_pos]
            k_block = k_blocks[b_idx, kv_h_idx, kb_idx]  # [block_size, head_dim]
            v_block = v_blocks[b_idx, kv_h_idx, kb_idx]  # [block_size, head_dim]

            # Recompute forward attention scores
            scores = jnp.einsum("qd,kd->qk", q_block, k_block) * scale
            is_valid = kb_pos < num_blocks_for_this_query
            scores = jnp.where(is_valid, scores, -1e9)

            return scores, v_block, kb_idx, is_valid

        # Gather all attended key blocks
        max_num_key_blocks = all_key_block_indices.shape[0]
        scores_list, v_list, kb_indices, is_valid_list = jax.vmap(backward_key_block)(jnp.arange(max_num_key_blocks))

        # Reshape to merge key blocks
        all_scores = scores_list.transpose(1, 0, 2).reshape(block_size, -1)
        all_values = v_list.transpose(1, 0, 2).reshape(-1, head_dim)

        # Recompute attention weights
        attn_weights = jax.nn.softmax(all_scores, axis=-1)  # [block_size, max_num_key_blocks * block_size]

        # Gradient of output w.r.t attention weights and values
        # do = attn_weights @ values
        # dvalues = attn_weights^T @ do
        # dattn_weights = do @ values^T
        dvalues = jnp.einsum("qk,qd->kd", attn_weights, do_block)  # [max_num_key_blocks * block_size, head_dim]
        dattn_weights = jnp.einsum("qd,kd->qk", do_block, all_values)  # [block_size, max_num_key_blocks * block_size]

        # Gradient through softmax
        # softmax gradient: d_scores = attn * (d_attn - sum(d_attn * attn))
        dscores = attn_weights * (
            dattn_weights - jnp.sum(dattn_weights * attn_weights, axis=-1, keepdims=True)
        )  # [block_size, max_num_key_blocks * block_size]

        # Reshape back to blocks
        dscores_blocks = dscores.reshape(block_size, max_num_key_blocks, block_size).transpose(1, 0, 2)
        dvalues_blocks = dvalues.reshape(max_num_key_blocks, block_size, head_dim)

        # Gradient of scores w.r.t q, k
        # scores = q @ k^T * scale
        # dq = dscores @ k * scale
        # dk = dscores^T @ q * scale
        def compute_grad_for_key_block(kb_pos):
            kb_idx = kb_indices[kb_pos]
            is_valid = is_valid_list[kb_pos]
            k_block_kb = k_blocks[b_idx, kv_h_idx, kb_idx]
            dscores_kb = dscores_blocks[kb_pos]  # [block_size, block_size]
            dvalues_kb = dvalues_blocks[kb_pos]  # [block_size, head_dim]

            # Mask out invalid blocks
            dscores_kb = jnp.where(is_valid, dscores_kb, 0.0)
            dvalues_kb = jnp.where(is_valid, dvalues_kb, 0.0)

            # dq contribution from this key block
            dq_contrib = jnp.einsum("qk,kd->qd", dscores_kb, k_block_kb) * scale

            # dk for this key block
            dk_kb = jnp.einsum("qk,qd->kd", dscores_kb, q_block) * scale

            # dv for this key block
            dv_kb = dvalues_kb

            return dq_contrib, dk_kb, dv_kb, kb_idx

        dq_contribs, dk_list, dv_list, kb_indices_out = jax.vmap(compute_grad_for_key_block)(
            jnp.arange(max_num_key_blocks)
        )

        # Sum up dq contributions
        dq_block = jnp.sum(dq_contribs, axis=0)  # [block_size, head_dim]

        return dq_block, dk_list, dv_list, kb_indices_out

    def process_head(b_idx, q_h_idx):
        """Process backward for all query blocks in a query head."""

        def process_query_block(qb_idx):
            return backward_query_block(b_idx, q_h_idx, qb_idx)

        dq_h, dk_h, dv_h, kb_indices_h = jax.vmap(process_query_block)(jnp.arange(num_query_blocks))
        # dq_h: [num_query_blocks, block_size, head_dim]
        # dk_h: [num_query_blocks, max_num_key_blocks, block_size, head_dim]
        # dv_h: [num_query_blocks, max_num_key_blocks, block_size, head_dim]
        # kb_indices_h: [num_query_blocks, max_num_key_blocks]

        return dq_h, dk_h, dv_h, kb_indices_h, q_h_idx // group_size  # Return kv_h_idx

    def process_batch(b_idx):
        """Process backward for all query heads in a batch."""
        dq_b, dk_b, dv_b, kb_indices_b, kv_h_indices = jax.vmap(lambda h: process_head(b_idx, h))(
            jnp.arange(num_q_heads)
        )
        # dq_b: [num_q_heads, num_query_blocks, block_size, head_dim]
        # dk_b: [num_q_heads, num_query_blocks, max_num_key_blocks, block_size, head_dim]
        # kv_h_indices: [num_q_heads] - maps each q head to its kv head

        return dq_b, dk_b, dv_b, kb_indices_b, kv_h_indices

    dq_all, dk_all, dv_all, kb_indices_all, kv_h_indices_all = jax.vmap(process_batch)(jnp.arange(batch))

    # dq can be directly reshaped
    # dq_all: [batch, num_q_heads, num_query_blocks, block_size, head_dim]
    dq = dq_all.transpose(0, 2, 3, 1, 4).reshape(batch, seq_len, num_q_heads, head_dim)

    # dk and dv need scatter_add because:
    # 1. Multiple query blocks may attend to the same key block
    # 2. Multiple query heads in the same GQA group share the same KV head
    dk = jnp.zeros((batch, num_kv_heads, num_query_blocks, block_size, head_dim))
    dv = jnp.zeros((batch, num_kv_heads, num_query_blocks, block_size, head_dim))

    # Accumulate gradients for each key block
    for b in range(batch):
        for q_h in range(num_q_heads):
            kv_h = kv_h_indices_all[b, q_h]
            for qb in range(num_query_blocks):
                max_kb = block_indices[b, kv_h, qb].shape[0]
                for kb_pos in range(max_kb):
                    kb_idx = kb_indices_all[b, q_h, qb, kb_pos]
                    dk = dk.at[b, kv_h, kb_idx].add(dk_all[b, q_h, qb, kb_pos])
                    dv = dv.at[b, kv_h, kb_idx].add(dv_all[b, q_h, qb, kb_pos])

    # Reshape back to sequence
    dk = dk.transpose(0, 2, 3, 1, 4).reshape(batch, seq_len, num_kv_heads, head_dim)
    dv = dv.transpose(0, 2, 3, 1, 4).reshape(batch, seq_len, num_kv_heads, head_dim)

    return dq, dk, dv
