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


import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import ragged_decode_mqa_xla


@kernel_registry.register("ragged_decode_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_decode_attention(
    query: Float[Array, "batch num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    sequence_start: Int[Array, "batch"],
    sequence_end: Int[Array, "batch"],
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    sliding_window: tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
) -> Float[Array, "batch num_q_heads head_dim"]:
    """Ragged MQA/GQA decoding with standard XLA operations.

    This function implements ragged attention for decoding scenarios where different
    sequences in a batch have different lengths. It supports Multi-Query Attention (MQA)
    and Grouped-Query Attention (GQA).

    Args:
        query: Query tensor of shape [batch, num_heads, head_dim].
            Represents the current decoding position (single token per sequence).
        key: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
            Contains all previous tokens in the KV cache.
        value: Value tensor of shape [batch, seq_len, num_kv_heads, head_dim].
            Contains all previous token values.
        sequence_start: int32 array of shape [batch].
            Start indices for each sequence in the batch.
        sequence_end: int32 array of shape [batch].
            End indices (exclusive) for each sequence in the batch.
        softmax_scale: Optional scale for attention scores. If None, uses 1/sqrt(head_dim).
        sliding_window: Optional (left, right) window sizes for local attention.
            Limits attention to tokens within the window around the query position.
            None means full attention to all valid tokens.
        logits_soft_cap: Optional soft capping value for attention logits.
            Applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large.
        softmax_aux: Optional auxiliary logits for attention sinks.
            Shape [num_heads, num_sinks] or [num_sinks]. Concatenated to attention logits
            before softmax to create attention sink behavior (e.g., always attending to
            initial tokens regardless of their position).

    Returns:
        Output tensor of shape [batch, num_heads, head_dim] after attention.

    Examples:
        >>> import jax.numpy as jnp
        >>> batch, seq_len, num_heads, head_dim = 2, 512, 8, 64
        >>>
        >>>
        >>> sequence_start = jnp.array([0, 0], dtype=jnp.int32)
        >>> sequence_end = jnp.array([384, 512], dtype=jnp.int32)
        >>>
        >>> query = jax.random.normal(jax.random.key(0), (batch, num_heads, head_dim))
        >>> key = jax.random.normal(jax.random.key(1), (batch, seq_len, num_heads, head_dim))
        >>> value = jax.random.normal(jax.random.key(2), (batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> output = ragged_decode_xla(
        ...     query, key, value,
        ...     sequence_start, sequence_end,
        ...     softmax_scale=1.0 / jnp.sqrt(head_dim)
        ... )
        >>>
        >>>
        >>> sinks = jnp.ones((num_heads, 4)) * 5.0
        >>> output = ragged_decode_xla(
        ...     query, key, value,
        ...     sequence_start, sequence_end,
        ...     softmax_scale=1.0 / jnp.sqrt(head_dim),
        ...     sliding_window=(256, 256),
        ...     logits_soft_cap=30.0,
        ...     softmax_aux=sinks
        ... )

    Notes:
        - This is a pure XLA/JAX implementation suitable for CPU/GPU/TPU
        - For TPU with Pallas optimization, use ragged_decode_attention instead
        - Supports both MQA (num_kv_heads=1) and GQA (num_kv_heads < num_heads)
        - Query position is assumed to be at sequence_end - 1 (current decode position)
    """
    return ragged_decode_mqa_xla(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
