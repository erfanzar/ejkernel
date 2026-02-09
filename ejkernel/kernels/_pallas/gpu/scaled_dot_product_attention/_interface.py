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

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import Array, Bool, Callable, Float, Int
from ._pallas_impl_fwd import scaled_dot_product_attention as _scaled_dot_product_attention_impl


@kernel_registry.register("scaled_dot_product_attention", Platform.PALLAS, Backend.GPU)
@kernel_registry.register("scaled_dot_product_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def scaled_dot_product_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch kv_len num_kv_heads head_dim"],
    value: Float[Array, "batch kv_len num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    init_bias: Callable[[], Float[Array, "batch num_heads seq_len kv_len"]] | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    cum_seqlens_q: Int[Array, "batch"] | None = None,
    cum_seqlens_k: Int[Array, "batch"] | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """Compute scaled dot-product attention using GPU-optimized cuDNN backend.
    This function provides a high-performance GPU implementation of scaled
    dot-product attention using NVIDIA's cuDNN library via JAX's Pallas/Triton
    framework. It's optimized for NVIDIA GPUs and supports various attention
    patterns including causal, local, and variable-length sequences.
    Args:
        query: Query tensor of shape [batch, seq_len, num_q_heads, head_dim].
            Contains the query vectors for attention computation.
        key: Key tensor of shape [batch, kv_len, num_kv_heads, head_dim].
            Contains the key vectors for attention computation.
        value: Value tensor of shape [batch, kv_len, num_kv_heads, head_dim].
            Contains the value vectors to be aggregated.
        attention_mask: Optional boolean mask of shape [batch, 1, seq_len, kv_len].
            Positions with True are attended, False positions are masked out.
            Legacy parameter; prefer using bias for attention masking.
        bias: Optional attention bias of shape [batch, num_heads, seq_len, kv_len].
            Added to attention logits before softmax. More flexible than attention_mask.
        init_bias: Optional callable that generates bias tensor lazily.
            Called only if bias is None. Useful for deferring expensive bias
            computations until needed.
        softmax_scale: Scaling factor applied to attention logits before softmax.
            Typically 1/sqrt(head_dim) for numerical stability. Defaults to
            1/sqrt(head_dim) if None.
        causal: If True, applies causal (autoregressive) masking where each position
            can only attend to itself and earlier positions. Essential for
            autoregressive language modeling.
        sliding_window: Optional local attention window size. Can be:
            - int: symmetric window size (limits attention to nearby positions)
            - tuple[int, int]: (past_window, future_window) for asymmetric windows
            When set, limits attention to local context for efficiency.
        cum_seqlens_q: Optional cumulative sequence lengths for query, shape [batch+1].
            Used for variable-length sequences in packed format. Element i contains
            the sum of all sequence lengths from 0 to i.
        cum_seqlens_k: Optional cumulative sequence lengths for key/value, shape [batch+1].
            Used for variable-length sequences in packed format.
    Returns:
        Attention output tensor of shape [batch, seq_len, num_q_heads, head_dim].
        Contains the weighted aggregation of values based on attention scores.
    Note:
        This implementation uses the cuDNN backend which provides highly optimized
        GPU kernels for attention computation. It's significantly faster than the
        XLA implementation on NVIDIA GPUs and supports features like:
        - Flash Attention optimizations for memory-efficient computation
        - Support for grouped query attention (GQA) and multi-query attention (MQA)
        - Variable-length sequence handling without padding overhead
        - Local attention patterns for long sequences
    Example:
        >>> import jax.numpy as jnp
        >>>
        >>> query = jnp.ones((2, 512, 8, 64))
        >>> key = jnp.ones((2, 512, 8, 64))
        >>> value = jnp.ones((2, 512, 8, 64))
        >>> output = scaled_dot_product_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> output = scaled_dot_product_attention(
        ...     query, key, value,
        ...     causal=True,
        ...     sliding_window=256
        ... )
        >>>
        >>>
        >>> bias = jnp.ones((2, 8, 512, 512)) * -1e9
        >>> output = scaled_dot_product_attention(query, key, value, bias=bias)
    """
    return _scaled_dot_product_attention_impl(
        query,
        key,
        value,
        attention_mask,
        bias,
        init_bias,
        softmax_scale,
        causal,
        sliding_window,
        cum_seqlens_q,
        cum_seqlens_k,
    )


__all__ = ("scaled_dot_product_attention",)
