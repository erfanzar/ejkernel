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


import chex
import jax
import jax.lax as lax
from jax import numpy as jnp
from jaxtyping import DTypeLike, PRNGKeyArray


def _maybe_broadcast_kv_to_q_heads(k: chex.Array, v: chex.Array, hq: int) -> tuple[chex.Array, chex.Array]:
    """Broadcast KV heads to match Q heads for GQA/MQA support."""
    if k.shape[-2] == hq:
        return k, v
    if k.shape[-2] != 1:
        raise ValueError(f"K/V heads must be either 1 or match Q heads. Got Hk={k.shape[-2]}, Hq={hq}.")
    k = jnp.broadcast_to(k, (*k.shape[:-2], hq, k.shape[-1]))
    v = jnp.broadcast_to(v, (*v.shape[:-2], hq, v.shape[-1]))
    return k, v


def _apply_logits_transforms(
    logits: chex.Array,
    *,
    softmax_scale: float,
    bias: chex.Array | None,
    logits_soft_cap: float | None,
    mask: chex.Array | None,
    logits_dtype: DTypeLike,
) -> chex.Array:
    """Apply transformations to attention logits: scaling, bias, soft cap, masking."""
    logits = logits.astype(logits_dtype)
    logits = logits * softmax_scale

    if bias is not None:
        logits = logits + bias.astype(logits.dtype)

    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    if mask is not None:
        mask = mask.astype(bool)
        mask_value = jnp.finfo(logits.dtype).min
        logits = jnp.where(mask, logits, mask_value)

    logits = logits.astype(jnp.promote_types(logits.dtype, jnp.float32))
    return logits


def _causal_mask_for_chunk(
    q_start: int,
    q_len: int,
    k_start: int,
    k_len: int,
) -> chex.Array:
    """
    Create causal attention mask for a chunk.

    Returns:
        [1, 1, q_len, k_len] where True means attend
    """
    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    mask = k_pos[None, :] <= q_pos[:, None]
    return mask[None, None, ...]


def _window_mask_for_chunk(
    q_start: int,
    q_len: int,
    k_start: int,
    k_len: int,
    window: tuple[int, int] | None,
) -> chex.Array | None:
    """
    Create sliding window attention mask for a chunk.

    Returns:
        [1, 1, q_len, k_len] where True means attend, or None
    """
    if window is None:
        return None

    w_left, w_right = window
    w_left = jnp.asarray(w_left)
    w_right = jnp.asarray(w_right)

    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    diff = k_pos[None, :] - q_pos[:, None]
    ok = (diff >= -w_left) & (diff <= w_right)
    return ok[None, None, ...]


def _slice_broadcast_qk(x: chex.Array | None, q_start: int, q_len: int, k_start: int, k_len: int) -> chex.Array | None:
    """Slice x along query/key axes with broadcasting-aware semantics to [*,*,q_len,k_len]."""
    if x is None:
        return None
    if x.ndim > 4:
        raise ValueError(f"bias/mask must be broadcastable to [B,H,Tq,Tk], got shape {x.shape}")
    if x.ndim < 4:
        x = x.reshape((1,) * (4 - x.ndim) + x.shape)

    if x.shape[-2] == 1:
        x_q = x
    else:
        x_q = lax.dynamic_slice_in_dim(x, q_start, q_len, axis=-2)

    if x_q.shape[-1] == 1:
        x_qk = x_q
    else:
        x_qk = lax.dynamic_slice_in_dim(x_q, k_start, k_len, axis=-1)

    target_shape = (x_qk.shape[0], x_qk.shape[1], q_len, k_len)
    x_qk = jnp.broadcast_to(x_qk, target_shape)
    return x_qk


def _attend_chunk(
    q_chunk: chex.Array,
    k_chunk: chex.Array,
    v_chunk: chex.Array,
    accum: chex.Array,
    x_max: chex.Array,
    denom: chex.Array,
    *,
    softmax_scale: float,
    bias_chunk: chex.Array | None,
    mask_chunk: chex.Array | None,
    window_mask: chex.Array | None,
    causal_mask: chex.Array | None,
    logits_soft_cap: float | None,
    logits_dtype: DTypeLike,
    precision: lax.PrecisionLike,
    dropout_prob: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
    softmax_aux: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, PRNGKeyArray | None]:
    """
    Process a single KV chunk with online softmax and optional attention sinks.
    """

    logits = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q_chunk,
        k_chunk,
        precision=precision,
    )

    combined_mask = mask_chunk
    if window_mask is not None:
        combined_mask = window_mask if combined_mask is None else jnp.logical_and(combined_mask, window_mask)
    if causal_mask is not None:
        combined_mask = causal_mask if combined_mask is None else jnp.logical_and(combined_mask, causal_mask)

    logits = _apply_logits_transforms(
        logits,
        softmax_scale=softmax_scale,
        bias=bias_chunk,
        logits_soft_cap=logits_soft_cap,
        mask=combined_mask,
        logits_dtype=logits_dtype,
    )

    if softmax_aux is not None:
        if softmax_aux.ndim == 1:
            sinks = softmax_aux.reshape(1, 1, 1, -1)
        elif softmax_aux.ndim == 2:
            sinks = softmax_aux.reshape(1, -1, 1, softmax_aux.shape[-1])
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

        B, H, q, k = logits.shape
        sinks = jnp.broadcast_to(sinks, (B, H, q, sinks.shape[-1]))
        combined_logits = jnp.concatenate([logits, sinks], axis=-1)

        loc_x_max = jnp.max(combined_logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        combined_weights = jnp.exp(combined_logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + combined_weights.sum(axis=-1)

        weights = combined_weights[..., :k]
    else:
        loc_x_max = jnp.max(logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        weights = jnp.exp(logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + weights.sum(axis=-1)

    next_key = dropout_key
    if dropout_prob > 0.0 and dropout_key is not None:
        dropout_key, next_key = jax.random.split(dropout_key)
        keep_prob = 1.0 - dropout_prob
        dropout_mask = jax.random.bernoulli(dropout_key, keep_prob, shape=weights.shape)
        weights = weights * dropout_mask / keep_prob

    att_v = jnp.einsum("bhqk,bkhd->bqhd", weights, v_chunk, precision=precision).astype(accum.dtype)
    accum = accum + att_v

    return accum, x_max, denom, next_key


def _flash_attention_fwd(
    q: chex.Array,
    k: chex.Array,
    v: chex.Array,
    *,
    softmax_scale: float,
    logits_soft_cap: float | None,
    bias: chex.Array | None,
    mask: chex.Array | None,
    window: tuple[int, int] | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: DTypeLike,
    softmax_aux: chex.Array | None = None,
    causal: bool = False,
    dropout_prob: float = 0.0,
    dropout_key: PRNGKeyArray | None = None,
) -> chex.Array:
    """
    Forward pass for chunked flash attention with online softmax and optional attention sinks.
    """
    B, Tq, Hq, D = q.shape
    _, Tk, Hk, Dk = k.shape
    if D != Dk:
        raise ValueError(f"q and k must have same depth. Got Dq={D}, Dk={Dk}.")
    if v.shape[1] != Tk:
        raise ValueError("k and v must share sequence length.")
    if v.shape[-2] != Hk:
        raise ValueError("k and v must share head count.")
    d_out = v.shape[-1]

    S_Q = min(chunk_size_q, Tq)
    S_K = min(chunk_size_k, Tk)

    outputs = []
    n_q_full = Tq // chunk_size_q
    q_rem = Tq % chunk_size_q

    def q_step(carry, i):
        dropout_key_i = carry
        q_chunk_start = i * chunk_size_q
        q_chunk_len = S_Q
        q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, q_chunk_len, axis=1)

        acc = jnp.zeros((B, q_chunk_len, Hq, d_out), dtype=jnp.float32)
        x_max = jnp.full((B, Hq, q_chunk_len), -jnp.inf, dtype=jnp.float32)
        denom = jnp.zeros((B, Hq, q_chunk_len), dtype=jnp.float32)

        chunk_dropout_key = dropout_key_i
        if dropout_prob > 0.0 and dropout_key_i is not None:
            dropout_key_i, chunk_dropout_key = jax.random.split(dropout_key_i)

        def kv_step(carry, j):
            acc_, x_max_, denom_, dk_ = carry
            kv_chunk_start = j * chunk_size_k
            kv_chunk_len = S_K
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
            mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
            )

            acc2, x2, d2, dk2 = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc_,
                x_max_,
                denom_,
                softmax_scale=softmax_scale,
                bias_chunk=bias_qk,
                mask_chunk=mask_qk,
                window_mask=win_mask,
                causal_mask=causal_mask,
                logits_soft_cap=logits_soft_cap,
                logits_dtype=logits_dtype,
                precision=precision,
                dropout_prob=dropout_prob,
                dropout_key=dk_,
                softmax_aux=softmax_aux,
            )
            return (acc2, x2, d2, dk2), None

        n_k_full = Tk // chunk_size_k
        (acc, x_max, denom, chunk_dropout_key), _ = lax.scan(
            kv_step, (acc, x_max, denom, chunk_dropout_key), jnp.arange(n_k_full)
        )

        k_rem = Tk % chunk_size_k

        def handle_k_rem():
            if k_rem == 0:
                return acc, x_max, denom, chunk_dropout_key

            kv_chunk_start = n_k_full * chunk_size_k
            kv_chunk_len = k_rem
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
            mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
            )

            return _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc,
                x_max,
                denom,
                softmax_scale=softmax_scale,
                bias_chunk=bias_qk,
                mask_chunk=mask_qk,
                window_mask=win_mask,
                causal_mask=causal_mask,
                logits_soft_cap=logits_soft_cap,
                logits_dtype=logits_dtype,
                precision=precision,
                dropout_prob=dropout_prob,
                dropout_key=chunk_dropout_key,
                softmax_aux=softmax_aux,
            )

        acc, x_max, denom, chunk_dropout_key = lax.cond(
            k_rem > 0,
            handle_k_rem,
            lambda: (acc, x_max, denom, chunk_dropout_key),
        )

        out_chunk = acc / denom.swapaxes(-1, -2)[..., None] if normalize_output else acc
        return dropout_key_i, out_chunk.astype(q.dtype)

    dropout_key, full_out = lax.scan(q_step, dropout_key, jnp.arange(n_q_full))
    full_out = jnp.swapaxes(full_out, 0, 1).reshape(B, n_q_full * chunk_size_q, Hq, d_out)
    outputs.append(full_out)

    if q_rem > 0:

        def process_remainder():
            q_chunk_start = n_q_full * chunk_size_q
            q_chunk_len = q_rem
            q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, q_chunk_len, axis=1)

            acc = jnp.zeros((B, q_chunk_len, Hq, d_out), dtype=jnp.float32)
            x_max = jnp.full((B, Hq, q_chunk_len), -jnp.inf, dtype=jnp.float32)
            denom = jnp.zeros((B, Hq, q_chunk_len), dtype=jnp.float32)

            n_k_full = Tk // chunk_size_k
            k_rem = Tk % chunk_size_k

            if dropout_prob > 0.0 and dropout_key is not None:
                _, chunk_dropout_key = jax.random.split(dropout_key)
            else:
                chunk_dropout_key = dropout_key

            def kv_step_rem(carry, j):
                acc_, x_max_, denom_, dk_ = carry
                kv_chunk_start = j * chunk_size_k
                kv_chunk_len = S_K
                k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
                v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
                k_chunk_bc, v_chunk_bc = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

                bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
                mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

                win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
                causal_mask = (
                    _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
                )

                acc2, x2, d2, dk2 = _attend_chunk(
                    q_chunk,
                    k_chunk_bc,
                    v_chunk_bc,
                    acc_,
                    x_max_,
                    denom_,
                    softmax_scale=softmax_scale,
                    bias_chunk=bias_qk,
                    mask_chunk=mask_qk,
                    window_mask=win_mask,
                    causal_mask=causal_mask,
                    logits_soft_cap=logits_soft_cap,
                    logits_dtype=logits_dtype,
                    precision=precision,
                    dropout_prob=dropout_prob,
                    dropout_key=dk_,
                    softmax_aux=softmax_aux,
                )
                return (acc2, x2, d2, dk2), None

            (acc, x_max, denom, chunk_dropout_key), _ = lax.scan(
                kv_step_rem, (acc, x_max, denom, chunk_dropout_key), jnp.arange(n_k_full)
            )

            if k_rem > 0:
                kv_chunk_start = n_k_full * chunk_size_k
                kv_chunk_len = k_rem
                k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, kv_chunk_len, axis=1)
                v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, kv_chunk_len, axis=1)
                k_chunk_bc, v_chunk_bc = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

                bias_qk = _slice_broadcast_qk(bias, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)
                mask_qk = _slice_broadcast_qk(mask, q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len)

                win_mask = _window_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len, window)
                causal_mask = (
                    _causal_mask_for_chunk(q_chunk_start, q_chunk_len, kv_chunk_start, kv_chunk_len) if causal else None
                )

                acc, x_max, denom, chunk_dropout_key = _attend_chunk(
                    q_chunk,
                    k_chunk_bc,
                    v_chunk_bc,
                    acc,
                    x_max,
                    denom,
                    softmax_scale=softmax_scale,
                    bias_chunk=bias_qk,
                    mask_chunk=mask_qk,
                    window_mask=win_mask,
                    causal_mask=causal_mask,
                    logits_soft_cap=logits_soft_cap,
                    logits_dtype=logits_dtype,
                    precision=precision,
                    dropout_prob=dropout_prob,
                    dropout_key=chunk_dropout_key,
                    softmax_aux=softmax_aux,
                )

            rem_out = acc / denom.swapaxes(-1, -2)[..., None] if normalize_output else acc
            return rem_out.astype(q.dtype)

        rem_out = process_remainder()
    else:
        rem_out = jnp.zeros((B, 0, Hq, d_out), dtype=q.dtype)

    outputs.append(rem_out)
    return jnp.concatenate(outputs, axis=1)
