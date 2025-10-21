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


"""Flash Attention implementation using Triton kernels.

This module provides a highly optimized implementation of Flash Attention,
an IO-aware exact attention algorithm that reduces memory usage from O(N²)
to O(N) through tiling and recomputation strategies.

Flash Attention is a breakthrough in efficient attention computation that
maintains exact attention semantics while dramatically reducing memory footprint.
The key insight is to split the attention computation into blocks and fuse
operations to minimize memory reads/writes between GPU HBM and SRAM.

Key advantages over standard attention:
1. Subquadratic memory: O(N) instead of O(N²) for sequence length N
2. Faster wall-clock time: Reduced memory I/O translates to speed improvements
3. Exact attention: No approximation, produces identical results to standard attention
4. Better scaling: Enables processing of much longer sequences

Algorithm overview:
- Query and key-value sequences are split into blocks
- Attention is computed block-by-block using online softmax
- Partial results are accumulated incrementally
- No full attention matrix is ever materialized

Supported features:
- Causal and non-causal attention
- Attention bias and masking
- Dropout during training
- Variable-length sequences (via cu_seqlens)
- Sliding window attention for local patterns
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Attention sinks via softmax_aux parameter
- Logits soft capping for numerical stability

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.flash_attention import flash_attention
    >>>
    >>> batch, seq_len, num_heads, head_dim = 2, 2048, 12, 64
    >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
    >>>
    >>>
    >>> output = flash_attention(q, k, v)
    >>>
    >>>
    >>> output = flash_attention(q, k, v, causal=True)
    >>>
    >>>
    >>> output = flash_attention(q, k, v, dropout_prob=0.1, dropout_seed=42)

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/abs/2205.14135
"""

import functools

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import _bwd_attention_kernel_call
from ._triton_impl_fwd import _fwd_attention_kernel_call


def _jax_fwd_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
) -> tuple[Float[Array, "batch seq_len_q num_heads head_dim"], tuple[Float[Array, "..."], ...]]:
    """Forward pass for flash attention with custom gradient support.

    Computes scaled dot-product attention with optional masking and dropout.
    Returns both the attention output and residuals needed for backward pass.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        value: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        attention_mask: Optional attention mask (legacy, use bias instead)
        bias: Optional attention bias of shape [batch, num_heads, seq_len, seq_len]
        softmax_scale: Scaling factor for QK^T before softmax
        dropout_prob: Dropout probability applied to attention weights
        causal: Whether to apply causal masking
        dropout_seed: Seed for dropout random number generation
        cum_seqlens_q: Cumulative sequence lengths for queries in variable-length mode
        cum_seqlens_k: Cumulative sequence lengths for keys in variable-length mode
        sliding_window: Window size for local attention (int or tuple of left/right)
        logits_soft_cap: Optional soft cap value for logits
        softmax_aux: Optional attention sink logits of shape [H, num_sinks] or [num_sinks]

    Returns:
        tuple: (attention_output, residuals) where residuals contain intermediate
               values needed for backward pass gradient computation
    """
    out, lse = _fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )

    return out, (query, key, value, bias, attention_mask, out, lse, dropout_seed, cum_seqlens_q, cum_seqlens_k)


def _jax_bwd_attention_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    residual: tuple[Float[Array, "..."], ...],
    dO: Float[Array, "batch seq_len num_heads head_dim"],
) -> tuple[
    Float[Array, "batch seq_len_q num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    Float[Array, "batch seq_len_k num_heads head_dim"] | None,
    None,
    None,
    None,
    None,
    None,
    None,
]:
    """Backward pass for flash attention gradient computation.

    Computes gradients with respect to queries, keys, and values using
    the saved residuals from the forward pass.

    Args:
        softmax_scale: Scaling factor used in forward pass
        dropout_prob: Dropout probability used in forward pass
        causal: Whether causal masking was applied
        sliding_window: Window size for local attention if used
        residual: Saved tensors from forward pass containing query, key, value, bias,
                 attention_mask, output, log-sum-exp, and other metadata
        dO: Gradient of loss with respect to attention output

    Returns:
        tuple: Gradients (dq, dk, dv, d_attention_mask, d_bias, d_scale, d_prob, d_seed)
               where only dq, dk, dv are non-None for differentiable parameters
    """
    query, key, value, bias, attention_mask, out, lse, dropout_seed, cum_seqlens_q, cum_seqlens_k = residual
    dq, dk, dv = _bwd_attention_kernel_call(
        dO=dO,
        q=query,
        k=key,
        v=value,
        bias=bias,
        attention_mask=attention_mask,
        o=out,
        M=lse,
        dropout_prob=dropout_prob,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        dropout_seed=dropout_seed,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        cum_seqlens_k=cum_seqlens_k,
        cum_seqlens_q=cum_seqlens_q,
        logits_soft_cap=logits_soft_cap,
    )
    return dq, dk, dv, None, None, None, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9, 10, 13, 14))
@ejit(static_argnums=(5, 6, 7, 9, 10, 13, 14))
def flash_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Flash attention with custom gradient computation.

    Efficient attention implementation using tiling and online softmax computation.
    Supports variable sequence lengths, causal masking, and sliding windows.

    This function is decorated with custom_vjp for efficient backward pass and
    ejit for JIT compilation with static arguments.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        value: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        attention_mask: Optional legacy attention mask (deprecated, use bias)
        bias: Attention bias tensor for masking or positional encoding
        softmax_scale: Scale factor for attention scores (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights
        causal: Apply causal (autoregressive) masking
        dropout_seed: Random seed for dropout
        cum_seqlens_q: Cumulative sequence lengths for variable-length queries
        cum_seqlens_k: Cumulative sequence lengths for variable-length keys
        sliding_window: Local attention window size (int or (left, right) tuple)

    Returns:
        chex.Array: Attention output tensor with same shape as query

    Note:
        Arguments at positions 5, 6, 7, 11 (softmax_scale, dropout_prob,
        causal, sliding_window) are marked as non-differentiable.
    """
    return _fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )[0]


flash_attention_call.defvjp(_jax_fwd_attention_call, _jax_bwd_attention_call)


@kernel_registry.register("flash_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Compute flash attention for efficient scaled dot-product attention.

    Flash Attention is a memory-efficient and fast implementation of exact
    attention that uses tiling and recomputation to reduce memory usage
    from O(N²) to O(N) where N is sequence length.

    Args:
        query: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        key: Key tensor of shape [batch, seq_len_k, num_heads, head_dim]
        value: Value tensor of shape [batch, seq_len_k, num_heads, head_dim]
        attention_mask: Optional attention mask (legacy, prefer bias parameter)
        bias: Attention bias for masking or relative position encoding
        softmax_scale: Scaling factor for QK^T (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability for attention weights (0-1)
        causal: Whether to apply causal masking for autoregressive models
        dropout_seed: Random seed for reproducible dropout
        cum_seqlens_q: Cumulative sequence lengths for packed variable-length sequences
        cum_seqlens_k: Cumulative sequence lengths for keys in variable-length mode
        sliding_window: Size of local attention window for sparse patterns
        logits_soft_cap: Optional soft cap value for logits (e.g., 20.0 for Gemma)
        softmax_aux: Optional attention sink logits of shape [H, num_sinks] or [num_sinks]
        segment_ids: Not supported in Triton implementation (raises NotImplementedError)

    Returns:
        chex.Array: Attention output with shape [batch, seq_len, num_heads, head_dim]

    Examples:
        >>>
        >>> out = flash_attention(query, key, value, causal=True)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, dropout_prob=0.1, softmax_scale=0.125)
        >>>
        >>>
        >>> out = flash_attention(query, key, value, cum_seqlens_q=cum_lens, cum_seqlens_k=cum_lens)
    """
    del precision, logits_dtype, normalize_output
    if q_segment_ids is not None and attention_mask is None:
        raise NotImplementedError("`q_segment_ids` is not implemented in triton impl yet!")
    if kv_segment_ids is not None and attention_mask is None:
        raise NotImplementedError("`kv_segment_ids` is not implemented in triton impl yet!")

    return flash_attention_call(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
