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

import chex
import jax
import jax.lax as lax
from jax import numpy as jnp


def _slice_along_axis(x: chex.Array | None, start: int, size: int, axis: int) -> chex.Array | None:
    """Slice array along a given axis."""
    if x is None:
        return None
    if x.shape[axis] == 1:
        return x
    return lax.dynamic_slice_in_dim(x, start_index=start, slice_size=size, axis=axis)


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
    scale: float,
    bias: chex.Array | None,
    logits_soft_cap: float | None,
    mask: chex.Array | None,
    window_mask: chex.Array | None,
    logits_dtype: jnp.dtype,
) -> chex.Array:
    """Apply transformations to attention logits: scaling, bias, soft cap, masking."""
    logits = logits.astype(logits_dtype)
    logits = logits * scale

    if bias is not None:
        logits = logits + bias.astype(logits.dtype)

    if logits_soft_cap is not None:
        logits = logits_soft_cap * jnp.tanh(logits / logits_soft_cap)

    # Combine all masks
    masks_to_combine = []
    if mask is not None:
        masks_to_combine.append(mask)
    if window_mask is not None:
        masks_to_combine.append(window_mask)

    if len(masks_to_combine) > 0:
        combined = masks_to_combine[0]
        for m in masks_to_combine[1:]:
            combined = jnp.logical_and(combined, m)
        mask_value = jnp.finfo(logits.dtype).min
        logits = jnp.where(combined, logits, mask_value)

    # Promote to at least float32 for stability
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

    Args:
        q_start: Starting position of query chunk in sequence
        q_len: Length of query chunk
        k_start: Starting position of key chunk in sequence
        k_len: Length of key chunk

    Returns:
        Boolean mask of shape [1, 1, q_len, k_len] where True means attend
    """
    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    # Allow attention only to positions <= current position
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

    Args:
        q_start: Starting position of query chunk in sequence
        q_len: Length of query chunk
        k_start: Starting position of key chunk in sequence
        k_len: Length of key chunk
        window: Optional (left_window, right_window) for local attention

    Returns:
        Boolean mask of shape [1, 1, q_len, k_len] where True means attend, or None
    """
    if window is None:
        return None
    w_left, w_right = window
    q_pos = q_start + jnp.arange(q_len)
    k_pos = k_start + jnp.arange(k_len)
    diff = k_pos[None, :] - q_pos[:, None]
    ok = (diff >= -int(w_left)) & (diff <= int(w_right))
    return ok[None, None, ...]


def _attend_chunk(
    q_chunk: chex.Array,
    k_chunk: chex.Array,
    v_chunk: chex.Array,
    accum: chex.Array,
    x_max: chex.Array,
    denom: chex.Array,
    *,
    scale: float,
    bias_chunk: chex.Array | None,
    mask_chunk: chex.Array | None,
    window_mask: chex.Array | None,
    causal_mask: chex.Array | None,
    logits_soft_cap: float | None,
    logits_dtype: jnp.dtype,
    precision: lax.PrecisionLike,
    dropout_prob: float = 0.0,
    dropout_key: chex.PRNGKey | None = None,
    softmax_aux: chex.Array | None = None,
) -> tuple[chex.Array, chex.Array, chex.Array, chex.PRNGKey | None]:
    """
    Process a single KV chunk with online softmax and optional attention sinks.

    This function computes attention for a chunk of keys/values, updating the running
    statistics for online softmax computation. Optionally supports attention sinks
    (auxiliary softmax logits) that absorb probability mass without contributing to output.

    Args:
        q_chunk: Query chunk of shape [B, q, H, D]
        k_chunk: Key chunk of shape [B, k, H, D]
        v_chunk: Value chunk of shape [B, k, H, d]
        accum: Accumulated weighted sum of shape [B, q, H, d]
        x_max: Running max of logits of shape [B, H, q]
        denom: Running sum of exp weights of shape [B, H, q]
        scale: Softmax scale factor
        bias_chunk: Optional bias of shape [B or 1, H or 1, q or 1, k or 1]
        mask_chunk: Optional mask of shape [B or 1, H or 1, q or 1, k or 1]
        window_mask: Optional window mask of shape [1, 1, q, k]
        causal_mask: Optional causal mask of shape [1, 1, q, k]
        logits_soft_cap: Optional soft cap value
        logits_dtype: Dtype for logits computation
        precision: Matrix multiplication precision
        dropout_prob: Dropout probability for attention weights
        dropout_key: PRNG key for dropout
        softmax_aux: Optional attention sink logits of shape [1, H, num_sinks, 1, 1]
                     or [H, num_sinks]. These participate in softmax but don't produce output.

    Returns:
        Updated (accum, x_max, denom, next_dropout_key) tuple

    Note:
        Attention sinks are auxiliary logits that participate in the softmax normalization
        but don't contribute to the output. They can absorb probability mass, improving
        numerical stability and model behavior.
    """
    # Compute logits: [B, H, q, k]
    logits = jnp.einsum(
        "...qhd,...khd->...hqk",
        q_chunk,
        k_chunk,
        precision=precision,
    )

    # Combine masks including causal mask
    combined_mask = mask_chunk
    if window_mask is not None:
        combined_mask = window_mask if combined_mask is None else jnp.logical_and(combined_mask, window_mask)
    if causal_mask is not None:
        combined_mask = causal_mask if combined_mask is None else jnp.logical_and(combined_mask, causal_mask)

    logits = _apply_logits_transforms(
        logits,
        scale=scale,
        bias=bias_chunk,
        logits_soft_cap=logits_soft_cap,
        mask=combined_mask,
        window_mask=None,  # Already combined above
        logits_dtype=logits_dtype,
    )

    # Add attention sink logits if provided
    if softmax_aux is not None:
        # Reshape sink logits to match batch/head dimensions
        if softmax_aux.ndim == 1:
            # [num_sinks] -> [1, 1, 1, num_sinks]
            sinks = softmax_aux.reshape(1, 1, 1, -1)
        elif softmax_aux.ndim == 2:
            # [H, num_sinks] -> [1, H, 1, num_sinks]
            sinks = softmax_aux.reshape(1, -1, 1, softmax_aux.shape[-1])
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {softmax_aux.shape}")

        # Broadcast to match logits shape: [B, H, q, num_sinks]
        B, H, q, k = logits.shape
        sinks = jnp.broadcast_to(sinks, (B, H, q, sinks.shape[-1]))

        # Concatenate sink logits with attention logits
        combined_logits = jnp.concatenate([logits, sinks], axis=-1)

        # Streaming softmax update with sinks
        loc_x_max = jnp.max(combined_logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        combined_weights = jnp.exp(combined_logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + combined_weights.sum(axis=-1)

        # Extract only the non-sink weights for output computation
        weights = combined_weights[..., :k]
    else:
        # Standard streaming softmax update without sinks
        loc_x_max = jnp.max(logits, axis=-1)
        new_x_max = jnp.maximum(x_max, loc_x_max)
        weights = jnp.exp(logits - new_x_max[..., None])
        alpha = jnp.exp(x_max - new_x_max)

        x_max = new_x_max
        accum = accum * alpha.swapaxes(-1, -2)[..., None]
        denom = (denom * alpha) + weights.sum(axis=-1)

    # Apply dropout if specified
    next_key = dropout_key
    if dropout_prob > 0.0 and dropout_key is not None:
        # Split key for this chunk
        dropout_key, next_key = jax.random.split(dropout_key)
        keep_prob = 1.0 - dropout_prob
        # Generate dropout mask
        dropout_mask = jax.random.bernoulli(dropout_key, keep_prob, shape=weights.shape)
        # Apply dropout and scale
        weights = weights * dropout_mask / keep_prob

    # Weighted sum (only non-sink weights contribute to output)
    weights = weights.astype(v_chunk.dtype)
    accum = accum + jnp.einsum(
        "...hqk,...khd->...qhd",
        weights,
        v_chunk,
        precision=precision,
    )
    return accum, x_max, denom, next_key


def _flash_attention_fwd(
    q: chex.Array,
    k: chex.Array,
    v: chex.Array,
    *,
    scale: float,
    logits_soft_cap: float | None,
    bias: chex.Array | None,
    mask: chex.Array | None,
    window: tuple[int, int] | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision: lax.PrecisionLike,
    logits_dtype: jnp.dtype,
    softmax_aux: chex.Array | None = None,
    causal: bool = False,
    dropout_prob: float = 0.0,
    dropout_key: chex.PRNGKey | None = None,
) -> chex.Array:
    """
    Forward pass for chunked flash attention with online softmax and optional attention sinks.

    Computes memory-efficient attention using chunked computation with O(N) memory complexity.
    Supports sliding window attention, logit soft capping, GQA/MQA, and attention sinks.

    Args:
        q: Query array of shape [B, Tq, H, D]
        k: Key array of shape [B, Tk, Hk, D] where Hk can be 1 (MQA) or H (MHA)
        v: Value array of shape [B, Tk, Hk, d]
        scale: Softmax scale factor (typically 1/sqrt(D))
        logits_soft_cap: Optional soft cap value for logits to prevent overflow
        bias: Optional bias array broadcastable to [B, H, Tq, Tk]
        mask: Optional boolean mask array broadcastable to [B, H, Tq, Tk]
        window: Optional (left_window, right_window) for sliding window/local attention
        chunk_size_q: Query chunk size for memory efficiency
        chunk_size_k: Key chunk size for memory efficiency
        normalize_output: Whether to normalize output by softmax denominator
        precision: Matrix multiplication precision
        logits_dtype: Dtype for logits computation (promoted to at least float32)
        softmax_aux: Optional attention sink logits of shape [H, num_sinks] or [num_sinks].
                     These participate in softmax but don't contribute to output, allowing
                     the model to absorb probability mass without affecting the result.

    Returns:
        Output array of shape [B, Tq, H, d]

    Note:
        Attention sinks are auxiliary logits that participate in the softmax normalization
        but don't contribute to the output. They improve numerical stability and can help
        models learn better attention distributions by providing "absorption" points.
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

    outputs = []
    n_q_full = Tq // chunk_size_q
    q_rem = Tq % chunk_size_q

    def q_step(carry, i):
        dropout_key_i = carry
        q_chunk_start = i * chunk_size_q
        q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, chunk_size_q, axis=1)

        acc = jnp.zeros((B, chunk_size_q, Hq, d_out), dtype=jnp.float32)
        x_max = jnp.full((B, Hq, chunk_size_q), float("-inf"), dtype=jnp.float32)
        denom = jnp.zeros((B, Hq, chunk_size_q), dtype=jnp.float32)

        # Split dropout key for this query chunk
        chunk_dropout_key = dropout_key_i
        if dropout_prob > 0.0 and dropout_key_i is not None:
            dropout_key_i, chunk_dropout_key = jax.random.split(dropout_key_i)

        def kv_step(carry, j):
            acc_, x_max_, denom_, dk_ = carry
            kv_chunk_start = j * chunk_size_k
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, chunk_size_k, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, chunk_size_k, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            # Slice bias/mask
            bias_qk = None
            if bias is not None:
                bias_q = lax.dynamic_slice_in_dim(bias, q_chunk_start, chunk_size_q, axis=-2)
                bias_qk = lax.dynamic_slice_in_dim(bias_q, kv_chunk_start, chunk_size_k, axis=-1)

            mask_qk = None
            if mask is not None:
                mask_q = lax.dynamic_slice_in_dim(mask, q_chunk_start, chunk_size_q, axis=-2)
                mask_qk = lax.dynamic_slice_in_dim(mask_q, kv_chunk_start, chunk_size_k, axis=-1)

            win_mask = _window_mask_for_chunk(q_chunk_start, chunk_size_q, kv_chunk_start, chunk_size_k, window)
            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, chunk_size_q, kv_chunk_start, chunk_size_k) if causal else None
            )

            acc2, x2, d2, dk2 = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc_,
                x_max_,
                denom_,
                scale=scale,
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

        # Process all KV chunks
        n_k_full = Tk // chunk_size_k
        (acc, x_max, denom, chunk_dropout_key), _ = lax.scan(
            kv_step, (acc, x_max, denom, chunk_dropout_key), jnp.arange(n_k_full)
        )

        # Handle remainder KV chunks
        k_rem = Tk % chunk_size_k
        if k_rem > 0:
            kv_chunk_start = n_k_full * chunk_size_k
            # Pad the remainder to chunk_size_k
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, Tk - kv_chunk_start, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, Tk - kv_chunk_start, axis=1)
            k_chunk = jnp.pad(k_chunk, [(0, 0), (0, chunk_size_k - k_rem), (0, 0), (0, 0)])
            v_chunk = jnp.pad(v_chunk, [(0, 0), (0, chunk_size_k - k_rem), (0, 0), (0, 0)])
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            # Create mask for padded region
            pad_mask = jnp.arange(chunk_size_k) < k_rem
            pad_mask = pad_mask[None, None, None, :]  # [1, 1, 1, k]

            bias_qk = None
            if bias is not None:
                bias_q = lax.dynamic_slice_in_dim(bias, q_chunk_start, chunk_size_q, axis=-2)
                bias_qk = lax.dynamic_slice_in_dim(bias_q, kv_chunk_start, Tk - kv_chunk_start, axis=-1)
                bias_qk = jnp.pad(bias_qk, [(0, 0), (0, 0), (0, 0), (0, chunk_size_k - k_rem)])

            mask_qk = None
            if mask is not None:
                mask_q = lax.dynamic_slice_in_dim(mask, q_chunk_start, chunk_size_q, axis=-2)
                mask_qk = lax.dynamic_slice_in_dim(mask_q, kv_chunk_start, Tk - kv_chunk_start, axis=-1)
                mask_qk = jnp.pad(mask_qk, [(0, 0), (0, 0), (0, 0), (0, chunk_size_k - k_rem)], constant_values=False)
                mask_qk = mask_qk & pad_mask
            else:
                mask_qk = pad_mask

            win_mask = _window_mask_for_chunk(q_chunk_start, chunk_size_q, kv_chunk_start, chunk_size_k, window)
            if win_mask is not None:
                win_mask = win_mask & pad_mask

            causal_mask = (
                _causal_mask_for_chunk(q_chunk_start, chunk_size_q, kv_chunk_start, chunk_size_k) if causal else None
            )
            if causal_mask is not None:
                causal_mask = causal_mask & pad_mask

            acc, x_max, denom, chunk_dropout_key = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc,
                x_max,
                denom,
                scale=scale,
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

        out_chunk = acc / denom.swapaxes(-1, -2)[..., None] if normalize_output else acc
        return dropout_key_i, out_chunk.astype(q.dtype)

    # Process all full Q chunks
    if n_q_full > 0:
        dropout_key, full_out = lax.scan(q_step, dropout_key, jnp.arange(n_q_full))
        full_out = jnp.swapaxes(full_out, 0, 1).reshape(B, n_q_full * chunk_size_q, Hq, d_out)
        outputs.append(full_out)

    # Handle remainder Q chunks
    if q_rem > 0:
        q_chunk_start = n_q_full * chunk_size_q
        q_chunk = lax.dynamic_slice_in_dim(q, q_chunk_start, q_rem, axis=1)

        acc = jnp.zeros((B, q_rem, Hq, d_out), dtype=jnp.float32)
        x_max = jnp.full((B, Hq, q_rem), float("-inf"), dtype=jnp.float32)
        denom = jnp.zeros((B, Hq, q_rem), dtype=jnp.float32)

        # Process KV chunks for remainder Q
        n_k_full = Tk // chunk_size_k

        # Split dropout key for remainder chunk
        chunk_dropout_key = dropout_key
        if dropout_prob > 0.0 and dropout_key is not None:
            dropout_key, chunk_dropout_key = jax.random.split(dropout_key)

        for j in range(n_k_full):
            kv_chunk_start = j * chunk_size_k
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, chunk_size_k, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, chunk_size_k, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = None
            if bias is not None:
                bias_q = lax.dynamic_slice_in_dim(bias, q_chunk_start, q_rem, axis=-2)
                bias_qk = lax.dynamic_slice_in_dim(bias_q, kv_chunk_start, chunk_size_k, axis=-1)

            mask_qk = None
            if mask is not None:
                mask_q = lax.dynamic_slice_in_dim(mask, q_chunk_start, q_rem, axis=-2)
                mask_qk = lax.dynamic_slice_in_dim(mask_q, kv_chunk_start, chunk_size_k, axis=-1)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_rem, kv_chunk_start, chunk_size_k, window)
            causal_mask = _causal_mask_for_chunk(q_chunk_start, q_rem, kv_chunk_start, chunk_size_k) if causal else None

            acc, x_max, denom, chunk_dropout_key = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc,
                x_max,
                denom,
                scale=scale,
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

        # Handle remainder KV for remainder Q
        k_rem = Tk % chunk_size_k
        if k_rem > 0:
            kv_chunk_start = n_k_full * chunk_size_k
            k_chunk = lax.dynamic_slice_in_dim(k, kv_chunk_start, k_rem, axis=1)
            v_chunk = lax.dynamic_slice_in_dim(v, kv_chunk_start, k_rem, axis=1)
            k_chunk, v_chunk = _maybe_broadcast_kv_to_q_heads(k_chunk, v_chunk, Hq)

            bias_qk = None
            if bias is not None:
                bias_q = lax.dynamic_slice_in_dim(bias, q_chunk_start, q_rem, axis=-2)
                bias_qk = lax.dynamic_slice_in_dim(bias_q, kv_chunk_start, k_rem, axis=-1)

            mask_qk = None
            if mask is not None:
                mask_q = lax.dynamic_slice_in_dim(mask, q_chunk_start, q_rem, axis=-2)
                mask_qk = lax.dynamic_slice_in_dim(mask_q, kv_chunk_start, k_rem, axis=-1)

            win_mask = _window_mask_for_chunk(q_chunk_start, q_rem, kv_chunk_start, k_rem, window)
            causal_mask = _causal_mask_for_chunk(q_chunk_start, q_rem, kv_chunk_start, k_rem) if causal else None

            acc, x_max, denom, chunk_dropout_key = _attend_chunk(
                q_chunk,
                k_chunk,
                v_chunk,
                acc,
                x_max,
                denom,
                scale=scale,
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
        outputs.append(rem_out.astype(q.dtype))

    return outputs[0] if len(outputs) == 1 else jnp.concatenate(outputs, axis=1)
