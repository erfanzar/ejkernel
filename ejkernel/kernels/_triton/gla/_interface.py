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


"""Gated Linear Attention (GLA) implementation using Triton kernels.

This module provides a specialized implementation of Gated Linear Attention,
a variant of linear attention that incorporates learnable gating mechanisms
to improve model expressiveness while maintaining O(N) time complexity.

GLA extends standard linear attention by applying element-wise gates to the
attention computation, allowing the model to dynamically control information
flow. This is particularly useful for capturing long-range dependencies while
maintaining computational efficiency.

The implementation is built on top of the general recurrent linear attention
kernel, configured specifically for GLA's gating patterns.

Key features:
- O(N) time complexity via recurrent formulation
- Learnable gates (g) for enhanced expressiveness
- Optional decay factors (g_gamma) for temporal dynamics
- Support for variable-length sequences
- GPU-optimized Triton kernels

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.gla import recurrent_gla
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> g = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>> output, final_state = recurrent_gla(q, k, v, g=g)
"""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ..recurrent import recurrent


@kernel_registry.register("gla", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent_gla(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """
    Computes Gated Linear Attention (GLA) in a recurrent, linear-time manner.

    This function provides a convenient wrapper around the core `recurrent`
    implementation, tailored for GLA. It processes sequences step-by-step,
    making it highly efficient for very long sequences and suitable for
    autoregressive decoding.

    It supports both standard batch processing and variable-length sequence
    processing using cumulative sequence lengths (`cu_seqlens`).

    Args:
        query: The query tensor. Expected shape is `(batch, seq_len, num_heads, head_dim)`
            or `(total_tokens, num_heads, head_dim)` if `cu_seqlens` is used.
        key: The key tensor. Must have the same shape as `q`.
        value: The value tensor. Must have the same shape as `q`.
        g: The gate tensor, specific to Gated Linear Attention. If provided, it
            should have the same shape as `q`.
        g_gamma: The gate decay factor.
        softmax_scale: A scaling factor applied to the query before the recurrent
            computation. If `None`, it defaults to `1 / sqrt(head_dim)`.
        initial_state: The initial hidden state for the recurrence. Useful for
            chunked processing of long sequences.
        reverse: If `True`, the sequence is processed in reverse order.
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
            This is a 1D tensor like `[0, len_seq1, len_seq1+len_seq2, ...]`.
            If provided, the input tensors `query, key, value, g` are expected to be
            "packed" with a shape of `(total_tokens, ...)`.

    Returns:
        A tuple containing:
            - o (jax.Array): The output tensor, with the same shape as `q`.
            - final_state (jax.Array): The final hidden state of the recurrence,
              which can be used as `initial_state` for a subsequent segment.

    Raises:
        ValueError: If `cu_seqlens` is provided and the batch size of `q` is
            not 1.
        ValueError: If `cu_seqlens` is provided and the number of initial states
            does not match the number of sequences.
    """
    if cu_seqlens is not None:
        if query.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {query.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if softmax_scale is None:
        softmax_scale = key.shape[-1] ** -0.5
    o, final_state = recurrent(
        query=query,
        key=key,
        value=value,
        g=g,
        g_gamma=g_gamma,
        softmax_scale=softmax_scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
