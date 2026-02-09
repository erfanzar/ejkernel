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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._impl import DEFAULT_MASK_VALUE, Array, Float, Int
from ._impl import prefill_page_attention as _prefill_page_attention_impl


@kernel_registry.register("prefill_page_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    return _prefill_page_attention_impl(
        query,
        key_cache,
        value_cache,
        context_len,
        page_indices,
        softmax_scale=softmax_scale,
        mask_value=mask_value,
        attn_logits_soft_cap=attn_logits_soft_cap,
        sliding_window=sliding_window,
    )


__all__ = ("prefill_page_attention",)
