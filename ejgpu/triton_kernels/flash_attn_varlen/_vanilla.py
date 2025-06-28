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

import jax.numpy as jnp


def _attention_forward_block(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sm_scale: float,
    causal: bool,
    use_exp2: bool,
) -> tuple[jnp.ndarray, ...]:
    """
    Core attention computation implementation in JAX.
    """
    _, seq_len_q, head_dim = q.shape
    seq_len_k = k.shape[1]
    attention_scores = jnp.einsum("bqd,bkd->bqk", q, k) / jnp.sqrt(head_dim)
    attention_scaled_scores = attention_scores * sm_scale
    if causal:
        mask = jnp.tril(jnp.ones((seq_len_q, seq_len_k)))
        attention_scaled_scores = jnp.where(mask, attention_scaled_scores, -jnp.inf)
    max_scores = jnp.max(attention_scaled_scores, axis=-1, keepdims=True)
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores
    if use_exp2:
        exp_scores = jnp.exp2(attention_shifted_scaled_scores)
    else:
        exp_scores = jnp.exp(attention_shifted_scaled_scores)
    softmax = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)
    softmax_lse = jnp.log(jnp.sum(exp_scores, axis=-1)) + jnp.squeeze(max_scores, axis=-1)
    o = jnp.einsum("bqk,bkd->bqd", softmax, v)
    return (
        o,
        softmax_lse,
        exp_scores,
        softmax,
        attention_shifted_scaled_scores,
        attention_scaled_scores,
        attention_scores,
    )


def attention_vanilla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sm_scale: float,
    causal: bool,
    layout: str,
    use_exp2: bool,
) -> tuple[jnp.ndarray, ...]:
    """Compute reference output and softmax_lse using JAX"""
    if layout == "bshd":
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
    elif layout != "bhsd":
        raise ValueError(f"Unknown layout {layout}")
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    kv_heads = k.shape[1]
    assert kv_heads == num_heads, "GQA's not supported in vanilla ref."
    seq_len_k = k.shape[2]
    q = q.reshape(batch_size * num_heads, seq_len_q, head_dim)
    k = k.reshape(batch_size * kv_heads, seq_len_k, head_dim)
    v = v.reshape(batch_size * kv_heads, seq_len_k, head_dim)
    (
        o,
        softmax_lse,
        exp_scores,
        softmax,
        attention_shifted_scaled_scores,
        attention_scaled_scores,
        attention_scores,
    ) = _attention_forward_block(q, k, v, sm_scale, causal, use_exp2)
    o = o.reshape(batch_size, num_heads, seq_len_q, head_dim)
    softmax_lse = softmax_lse.reshape(batch_size, num_heads, seq_len_q)
    exp_scores = exp_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    softmax = softmax.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    attention_shifted_scaled_scores = attention_shifted_scaled_scores.reshape(
        batch_size, num_heads, seq_len_q, seq_len_k
    )
    attention_scaled_scores = attention_scaled_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    attention_scores = attention_scores.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
    if layout == "bshd":
        o = jnp.transpose(o, (0, 2, 1, 3))

    return (
        o,
        softmax_lse,
        exp_scores,
        softmax,
        attention_shifted_scaled_scores,
        attention_scaled_scores,
        attention_scores,
    )


def attention_varlen(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sm_scale: float,
    causal: bool,
    layout: str,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlens_q: int | None = None,
    max_seqlens_k: int | None = None,
    use_exp2: bool = False,
) -> tuple[jnp.ndarray, ...]:
    if layout != "thd":
        raise ValueError(f"Unsupported layout {layout}. Expected 'thd'.")

    batch_size = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]

    total_L_q = q.shape[0]

    o = jnp.zeros((total_L_q, num_heads, head_dim), dtype=q.dtype)
    softmax_lse = jnp.zeros((total_L_q, num_heads), dtype=jnp.float32)
    for i in range(batch_size):
        start_q = int(cu_seqlens_q[i])
        end_q = int(cu_seqlens_q[i + 1])
        start_k = int(cu_seqlens_k[i])
        end_k = int(cu_seqlens_k[i + 1])
        q_i = q[start_q:end_q, :, :]
        k_i = k[start_k:end_k, :, :]
        v_i = v[start_k:end_k, :, :]
        q_i = jnp.transpose(q_i, (1, 0, 2))
        k_i = jnp.transpose(k_i, (1, 0, 2))
        v_i = jnp.transpose(v_i, (1, 0, 2))
        (
            o_i,
            softmax_lse_i,
            exp_scores_i,
            softmax_i,
            attention_shifted_scaled_scores_i,
            attention_scaled_scores_i,
            attention_scores_i,
        ) = _attention_forward_block(q_i, k_i, v_i, sm_scale, causal, use_exp2)
        o_i = jnp.transpose(o_i, (1, 0, 2)).astype(jnp.float16)
        o = o.at[start_q:end_q, :, :].set(o_i)
        softmax_lse = softmax_lse.at[start_q:end_q, :].set(jnp.transpose(softmax_lse_i, (1, 0)))
        exp_scores_i = jnp.transpose(exp_scores_i, (1, 0, 2))
        softmax_i = jnp.transpose(softmax_i, (1, 0, 2))
        attention_shifted_scaled_scores_i = jnp.transpose(attention_shifted_scaled_scores_i, (1, 0, 2))
        attention_scaled_scores_i = jnp.transpose(attention_scaled_scores_i, (1, 0, 2))
        attention_scores_i = jnp.transpose(attention_scores_i, (1, 0, 2))

    return (
        o,
        softmax_lse,
        None,
        None,
        None,
        None,
        None,
    )


def complex_attention_vanilla(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    sm_scale: float,
    causal: bool,
    layout: str,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlens_q: int | None = None,
    max_seqlens_k: int | None = None,
    use_exp2: bool = False,
) -> tuple[jnp.ndarray, ...]:
    if layout == "thd":
        return attention_varlen(
            q,
            k,
            v,
            sm_scale,
            causal,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlens_q,
            max_seqlens_k,
            use_exp2,
        )
    return attention_vanilla(
        q,
        k,
        v,
        sm_scale,
        causal,
        layout,
        use_exp2,
    )
