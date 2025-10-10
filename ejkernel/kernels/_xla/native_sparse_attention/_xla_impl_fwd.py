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


@ejit(static_argnums=(5,))
def _sparse_attention_fwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    Forward pass for block-sparse attention with per-token selection and causal masking.
    """
    B, T, HQ, D = q.shape
    HKV = k.shape[2]
    G = HQ // HKV

    NB = (T + block_size - 1) // block_size
    pad_len = NB * block_size - T
    if pad_len > 0:
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))

    k_blocks = k.reshape(B, NB, block_size, HKV, D)
    v_blocks = v.reshape(B, NB, block_size, HKV, D)

    def attend_for_head(hq, q_b, k_blocks_b, v_blocks_b, bi_b, bc_b):
        kvh = hq // G

        def attend_for_token(t):
            q_vec = q_b[t, hq]
            inds = bi_b[t, kvh]
            cnt = bc_b[t, kvh]

            k_sel = k_blocks_b[inds, :, kvh, :]
            v_sel = v_blocks_b[inds, :, kvh, :]

            bs = jnp.arange(block_size)
            local_limit = t - inds * block_size
            pos_mask = bs[None, :] <= local_limit[:, None]
            s = jnp.arange(inds.shape[0])
            blk_mask = s < cnt
            valid_mask = pos_mask & blk_mask[:, None]

            scores = jnp.einsum("d,sbd->sb", q_vec, k_sel) * softmax_scale
            scores = jnp.where(valid_mask, scores, jnp.full_like(scores, -1e9))
            w = jax.nn.softmax(scores.reshape(-1), axis=-1)

            v_flat = v_sel.reshape(-1, D)
            return (w[:, None] * v_flat).sum(axis=0)

        return jax.vmap(attend_for_token, in_axes=(0,), out_axes=0)(jnp.arange(T))

    outs = jax.vmap(
        lambda q_b, kb_b, vb_b, bi_b, bc_b: jax.vmap(
            attend_for_head,
            in_axes=(0, None, None, None, None, None),
            out_axes=0,
        )(jnp.arange(HQ), q_b, kb_b, vb_b, bi_b, bc_b).transpose(1, 0, 2),
        in_axes=(0, 0, 0, 0, 0),
        out_axes=0,
    )(q, k_blocks, v_blocks, block_indices, block_counts)

    return outs
