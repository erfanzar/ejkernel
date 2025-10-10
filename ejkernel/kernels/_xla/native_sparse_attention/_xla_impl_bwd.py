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
def _sparse_attention_bwd(
    q: Float[Array, "batch seq_len num_q_heads head_dim"],
    k: Float[Array, "batch seq_len num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
    do: Float[Array, "batch seq_len num_q_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len num_q_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
    Float[Array, "batch seq_len num_kv_heads head_dim"],
]:
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

    dq = jnp.zeros_like(q)
    bs = jnp.arange(block_size)

    def bkvh_backward(b, kvh):
        hq_start = kvh * G

        q_b = q[b]
        do_b = do[b]
        q_grp = jax.lax.dynamic_slice(q_b, start_indices=(0, hq_start, 0), slice_sizes=(T, G, D))
        do_grp = jax.lax.dynamic_slice(do_b, start_indices=(0, hq_start, 0), slice_sizes=(T, G, D))

        inds_bt = block_indices[b, :, kvh, :]
        cnt_bt = block_counts[b, :, kvh]

        def token_bwd(t):
            inds = inds_bt[t]
            cnt = cnt_bt[t]

            k_sel = k_blocks[b, inds, :, kvh, :]
            v_sel = v_blocks[b, inds, :, kvh, :]

            local_limit = t - inds * block_size
            pos_mask = bs[None, :] <= local_limit[:, None]
            s_ar = jnp.arange(inds.shape[0])
            blk_mask = s_ar < cnt
            valid_mask = pos_mask & blk_mask[:, None]
            mask_flat = valid_mask.reshape(-1)

            k_flat = k_sel.reshape(-1, D)
            v_flat = v_sel.reshape(-1, D)

            def head_bwd(g):
                q_vec = q_grp[t, g]
                do_vec = do_grp[t, g]

                scores = (k_flat @ q_vec) * softmax_scale
                scores = jnp.where(mask_flat, scores, -1e9)
                w = jax.nn.softmax(scores, axis=-1)

                z = v_flat @ do_vec
                mu = (w * z).sum()
                ds = w * (z - mu)
                ds = jnp.where(mask_flat, ds, 0.0)

                dQ = softmax_scale * (ds[:, None] * k_flat).sum(axis=0)
                dK_flat = softmax_scale * (ds[:, None] * q_vec[None, :])
                dV_flat = w[:, None] * do_vec[None, :]

                dV_flat = jnp.where(mask_flat[:, None], dV_flat, 0.0)

                return dQ, dK_flat.reshape(-1, block_size, D), dV_flat.reshape(-1, block_size, D)

            dQ_g, dK_sel_g, dV_sel_g = jax.vmap(head_bwd, in_axes=(0,), out_axes=(0, 0, 0))(jnp.arange(G))

            dK_sel_sum = dK_sel_g.sum(axis=0)
            dV_sel_sum = dV_sel_g.sum(axis=0)

            dk_upd = jnp.zeros((NB, block_size, D))
            dv_upd = jnp.zeros((NB, block_size, D))
            dk_upd = dk_upd.at[inds].add(dK_sel_sum)
            dv_upd = dv_upd.at[inds].add(dV_sel_sum)

            return dQ_g, dk_upd, dv_upd

        dq_heads_t, dk_bt, dv_bt = jax.vmap(token_bwd, in_axes=(0,))(jnp.arange(T))

        dq_b = jnp.zeros((T, HQ, D))
        dq_b = jax.lax.dynamic_update_slice(dq_b, dq_heads_t, (0, hq_start, 0))

        dk_b = dk_bt.sum(axis=0)
        dv_b = dv_bt.sum(axis=0)
        return dq_b, dk_b, dv_b

    dq_b, dk_b, dv_b = jax.vmap(
        jax.vmap(bkvh_backward, in_axes=(None, 0), out_axes=(0, 0, 0)),
        in_axes=(0, None),
        out_axes=(0, 0, 0),
    )(jnp.arange(B), jnp.arange(HKV))

    dq = dq_b.sum(axis=1)

    dk = dk_b.reshape(B, HKV, NB * block_size, D).transpose(0, 2, 1, 3)
    dv = dv_b.reshape(B, HKV, NB * block_size, D).transpose(0, 2, 1, 3)

    dk = dk[:, :T, :, :]
    dv = dv[:, :T, :, :]

    return dq, dk, dv
