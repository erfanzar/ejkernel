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
import jaxtyping
from beartype import beartype
from beartype.typing import Callable
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, PRNGKeyArray

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads vhead_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    deterministic: bool = True,
    dropout_rng: PRNGKeyArray | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    dtype: DTypeLike | None = jnp.bfloat16,
    softmax_dtype: DTypeLike | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> tuple[Float[Array, "batch seq_len num_q_heads vhead_dim"], Float[Array, "batch num_heads seq_len kv_len"]]:
    """
    Computes multi-head attention using standard JAX operations.

    Supports GQA/MQA by reshaping the query tensor to match the number of
    key/value heads. Applies scaling, optional bias/attention_mask, softmax (potentially
    in float32), and optional dropout.

    Args:
        query: Query tensor with shape [batch, seq_len, num_q_heads, head_dim].
            The main input sequence to attend from.
        key: Key tensor with shape [batch, kv_len, num_kv_heads, head_dim].
            Keys for attention computation. May have fewer heads than queries (GQA/MQA).
        value: Value tensor with shape [batch, kv_len, num_kv_heads, head_dim].
            Values to aggregate based on attention weights.
        attention_mask: Optional boolean attention_mask with shape [batch, 1, seq_len, kv_len].
            True values indicate positions to attend to, False positions are masked.
            Used if `bias` is not provided.
        bias: Optional attention bias with shape [batch, num_heads, seq_len, kv_len].
            Additive bias applied to attention scores before softmax.
            Takes precedence over `attention_mask`.
        init_bias: Optional callable that returns bias tensor.
            Used to lazily initialize bias if both attention_mask and bias are None.
        deterministic: If True, disables dropout (default). If False, applies dropout.
        dropout_rng: JAX PRNG key for dropout. Required when deterministic=False
            and dropout_prob > 0 in metadata.
        softmax_aux: Optional auxiliary tensor for softmax computation.
        softmax_scale: Optional float for scaling attention scores. If None, uses 1/sqrt(head_dim).
        logits_soft_cap: Optional float for capping attention logits using tanh.
            When specified, applies: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large.
        dtype: Data type for computation. Defaults to bfloat16.
        softmax_dtype: Data type for softmax computation. Defaults to float32.
        dropout_prob: Dropout probability. Only applied when deterministic=False.
        sliding_window: Optional sliding window attention constraint. Can be:
            - int: Symmetric window (same left and right window size)
            - tuple[int, int]: Asymmetric window (left_window, right_window)
            - None: No window constraint (full attention)
            When specified, each query position can only attend to keys within the window.

    Returns:
        AttentionOutput containing:
            - attention_outputs: Float[Array, "batch seq_len num_q_heads head_dim"]
              The attended representation.
            - attention_weights: Float[Array, "batch num_heads seq_len kv_len"] | None
              The attention weights (if return_weights is True in metadata).

    Raises:
        NotImplementedError: If the bias head dimension cannot be reshaped correctly
            to match the query head structure for GQA/MQA.
    """

    softmax_scale = softmax_scale if softmax_scale is not None else query.shape[-1] ** -0.5

    if softmax_dtype is None:
        softmax_dtype = jnp.float32

    if attention_mask is None and bias is None and init_bias is not None:
        bias = init_bias()
    if bias is None and attention_mask is None and init_bias is not None:
        bias = init_bias()

    b, qs, qh, d = query.shape
    b, ks, kh, d = key.shape
    *_, vd = value.shape
    num_reps = qh // kh
    query = jnp.reshape(query, (b, qs, kh, num_reps, d))
    query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    aw = jnp.einsum("bskhd,bmkd->bkhsm", query * softmax_scale, key, optimize=True)

    if logits_soft_cap is not None:
        aw = logits_soft_cap * jnp.tanh(aw / logits_soft_cap)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            left_window = sliding_window
            right_window = sliding_window
        else:
            left_window, right_window = sliding_window

        q_pos = jnp.arange(qs)[:, None]
        k_pos = jnp.arange(ks)[None, :]

        window_mask = (k_pos >= q_pos - left_window) & (k_pos <= q_pos + right_window)
        window_mask = window_mask.reshape(1, 1, 1, qs, ks)

        aw = jnp.where(window_mask, aw, jnp.finfo(aw.dtype).min)
    if causal:
        aw = jnp.where(jnp.tril(jnp.ones((qs, ks), "b1")).reshape(1, 1, 1, qs, ks), aw, jnp.finfo(aw.dtype).min)
    if bias is not None:
        if bias.shape[1] == (kh * num_reps):
            bias = bias.reshape(b, kh, num_reps, qs, ks)
        elif bias.shape[1] == kh:
            bias = bias.reshape(b, kh, 1, qs, ks)
        elif bias.shape[1] == 1:
            bias = bias.reshape(b, 1, 1, qs, ks)
        else:
            raise NotImplementedError("bias heads wont match!")
        aw = jnp.add(aw, bias.astype(aw.dtype))

    elif attention_mask is not None:
        if attention_mask.dtype != jnp.bool_:
            attention_mask = attention_mask.astype(jnp.bool_)

        if attention_mask.ndim == 4:
            if attention_mask.shape[1] == 1:
                attention_mask = jnp.broadcast_to(attention_mask, (b, kh, qs, ks))
                attention_mask = jnp.reshape(attention_mask, (b, kh, 1, qs, ks))
            elif attention_mask.shape[1] == kh:
                attention_mask = jnp.reshape(attention_mask, (b, kh, 1, qs, ks))
            elif attention_mask.shape[1] == (kh * num_reps):
                attention_mask = jnp.reshape(attention_mask, (b, kh, num_reps, qs, ks))
            else:
                attention_mask = jnp.broadcast_to(attention_mask[:, :1], (b, 1, qs, ks))
                attention_mask = jnp.reshape(attention_mask, (b, 1, 1, qs, ks))
        elif attention_mask.ndim == 3:
            attention_mask = jnp.reshape(attention_mask, (b, 1, 1, qs, ks))
        elif attention_mask.ndim == 2:
            attention_mask = jnp.reshape(attention_mask, (b, 1, 1, 1, ks))
            attention_mask = jnp.broadcast_to(attention_mask, (b, 1, 1, qs, ks))
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

        aw = jnp.where(attention_mask, aw, jnp.finfo(aw.dtype).min)
    if softmax_aux is not None:
        if softmax_aux.ndim == 2:
            num_sinks = softmax_aux.shape[-1]
            sinks = softmax_aux.reshape(1, kh, 1, 1, num_sinks)
            sinks = jnp.broadcast_to(sinks, (b, kh, num_reps, qs, num_sinks))
        elif softmax_aux.ndim == 1:
            num_sinks = softmax_aux.shape[0]
            sinks = softmax_aux.reshape(1, 1, 1, 1, num_sinks)
            sinks = jnp.broadcast_to(sinks, (b, kh, num_reps, qs, num_sinks))
        else:
            raise ValueError(f"Unsupported softmax_aux shape: {softmax_aux.shape}")
        combined_logits = jnp.concatenate([aw, sinks], axis=-1)
        combined_logits = combined_logits - jnp.max(combined_logits, axis=-1, keepdims=True)
        probs = jax.nn.softmax(combined_logits.astype(softmax_dtype), axis=-1).astype(dtype)

        aw = probs[..., :ks]
    else:
        aw = jax.nn.softmax(aw.astype(softmax_dtype), axis=-1).astype(dtype)

    if not deterministic and dropout_prob > 0.0 and dropout_rng is not None:
        keep_prob = 1.0 - dropout_prob
        dropout_shape = tuple([1] * (key.ndim - 2)) + aw.shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
        aw = aw * multiplier

    attention = jnp.einsum("bkhsm,bmkd->bskhd", aw, value, optimize=True).reshape(b, qs, qh, vd)
    return attention, aw.reshape(b, kh * num_reps, qs, ks)
