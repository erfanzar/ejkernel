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


"""XLA reference implementation of chunked prefill paged attention.

This module contains the core implementation of paged attention for the
prefill phase. It uses standard JAX operations (gather, einsum, softmax)
to compute attention over a paged KV cache.

The implementation is designed for correctness and portability across
all XLA backends (CPU, GPU, TPU), serving as a reference for optimized
platform-specific implementations.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.float32).max)


def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Compute chunked prefill paged attention using XLA operations.

    This function processes a chunk of query tokens during the prefill phase,
    computing attention against a paged KV cache. It supports the standard
    attention features needed for LLM inference: causal masking, GQA/MQA,
    sliding window, and logits soft capping.

    The query positions are assumed to be the last `chunk_size` positions
    of the current context. For example, if context_len=512 and chunk_size=128,
    the query positions are 384, 385, ..., 511.

    Args:
        query: Query tensor [chunk_size, num_q_heads, head_dim].
            The chunk of tokens being processed in this prefill step.
        key_cache: Paged key cache [num_kv_heads, total_num_pages, page_size, head_dim].
            Contains keys for all pages, indexed by page_indices for this sequence.
        value_cache: Paged value cache [num_kv_heads, total_num_pages, page_size, head_dim].
            Same structure as key_cache.
        context_len: Total context length including this chunk [1].
            Scalar array indicating how many tokens are valid in the KV cache.
        page_indices: Physical page indices for this sequence [num_pages].
            Maps logical page positions to physical pages in the cache.
        softmax_scale: Attention scaling factor. Defaults to 1/sqrt(head_dim).
        mask_value: Value used for masked (invalid) positions.
            Defaults to a large negative value for numerical stability.
        attn_logits_soft_cap: Optional soft cap for attention logits.
            If set, applies: cap * tanh(logits / cap).
        sliding_window: Optional sliding window size.
            If set, each query can only attend to the last `sliding_window` tokens.

    Returns:
        Attention output [chunk_size, num_q_heads, head_dim].
        Same dtype as input query.

    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> chunk_size, num_heads, head_dim = 64, 8, 64
        >>> query = jnp.ones((chunk_size, num_heads, head_dim))
        >>> key_cache = jnp.ones((num_heads, 50, 16, head_dim))
        >>> value_cache = jnp.ones((num_heads, 50, 16, head_dim))
        >>> context_len = jnp.array([256])
        >>> page_indices = jnp.arange(16)
        >>>
        >>> output = prefill_page_attention(
        ...     query, key_cache, value_cache, context_len, page_indices
        ... )
        >>> output.shape
        (64, 8, 64)

    Note:
        - GQA is supported: num_q_heads can be a multiple of num_kv_heads
        - Causal masking is always applied (prefill is autoregressive)
        - Page size is inferred from key_cache.shape[2]
    """
    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, _ = key_cache.shape
    num_groups = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    # Get context length as scalar
    length = context_len[0]

    # Calculate number of pages needed
    (length + page_size - 1) // page_size

    # Gather K/V from paged cache
    # page_indices contains the physical page indices for this sequence
    k_pages = key_cache[:, page_indices]  # [num_kv_heads, num_pages, page_size, head_dim]
    v_pages = value_cache[:, page_indices]  # [num_kv_heads, num_pages, page_size, head_dim]

    # Reshape to [num_kv_heads, total_seq_len, head_dim]
    max_seq_len = page_indices.shape[0] * page_size
    k = k_pages.reshape(num_kv_heads, max_seq_len, head_dim)
    v = v_pages.reshape(num_kv_heads, max_seq_len, head_dim)

    # Repeat K/V for grouped query attention
    k = jnp.repeat(k, num_groups, axis=0)  # [num_q_heads, max_seq_len, head_dim]
    v = jnp.repeat(v, num_groups, axis=0)  # [num_q_heads, max_seq_len, head_dim]

    # Compute attention scores: [num_q_heads, chunk_size, max_seq_len]
    # q: [chunk_size, num_q_heads, head_dim] -> transpose for einsum
    qk = jnp.einsum("qhd,hsd->hqs", query, k, preferred_element_type=jnp.float32)
    qk = qk * softmax_scale

    # Apply soft cap if specified
    if attn_logits_soft_cap is not None:
        qk = attn_logits_soft_cap * jnp.tanh(qk / attn_logits_soft_cap)

    # Create masks
    # Query positions: last chunk_size positions of the sequence
    # q_pos[i] = length - chunk_size + i
    q_positions = (length - chunk_size) + jnp.arange(chunk_size)  # [chunk_size]
    kv_positions = jnp.arange(max_seq_len)  # [max_seq_len]

    # Padding mask: only attend to valid positions (< length)
    padding_mask = kv_positions[None, :] < length  # [1, max_seq_len]

    # Causal mask: q_pos >= kv_pos (can attend to current and past)
    causal_mask = q_positions[:, None] >= kv_positions[None, :]  # [chunk_size, max_seq_len]

    # Combine masks
    mask = jnp.logical_and(padding_mask, causal_mask)

    # Apply sliding window if specified
    if sliding_window is not None:
        # Can only attend to kv_pos >= q_pos - sliding_window + 1
        sliding_mask = kv_positions[None, :] >= (q_positions[:, None] - sliding_window + 1)
        mask = jnp.logical_and(mask, sliding_mask)

    # Broadcast mask to [num_q_heads, chunk_size, max_seq_len]
    mask = jnp.broadcast_to(mask, (num_q_heads, chunk_size, max_seq_len))

    qk = qk + jnp.where(mask, 0.0, mask_value)

    # Softmax
    attn = jax.nn.softmax(qk, axis=-1)

    # Compute output: [num_q_heads, chunk_size, head_dim]
    out = jnp.einsum("hqs,hsd->qhd", attn, v)

    return out.astype(query.dtype)
