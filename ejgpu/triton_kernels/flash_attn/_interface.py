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
import functools

import jax

from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    q: jax.Array | None,
    k: jax.Array | None,
    v: jax.Array | None,
    attention_mask: jax.Array | None = None,
    bias: jax.Array | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    varlen_mode: bool = True,
):
    """
    Forward pass for Flash Attention in a custom VJP.

    This function computes the attention output and saves intermediate values
    (residuals) required for the backward pass.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        attention_mask: Optional mask to prevent attention to certain positions.
        bias: Optional bias added to the attention scores.
        softmax_scale: Optional scaling factor for softmax.
        dropout_prob: Dropout probability.
        causal: If True, applies causal masking.
        dropout_seed: Optional seed for dropout.
        varlen_mode: If True, enables variable-length sequence processing.

    Returns:
        A tuple containing:
            - The attention output tensor.
            - A tuple of residuals for the backward pass.
    """
    out, lse = fwd_triton_impl(
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        varlen_mode=varlen_mode,
    )
    residual = (q, k, v, bias, attention_mask, out, lse, dropout_seed)
    return out, residual


def _bwd_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    varlen_mode: bool,
    residual: tuple[jax.Array],
    dO: jax.Array,
):
    """
    Backward pass for Flash Attention in a custom VJP.

    This function computes the gradients of the query, key, and value tensors
    using the residuals from the forward pass.

    Args:
        softmax_scale: The scaling factor for softmax.
        dropout_prob: Dropout probability.
        causal: If True, causal masking was applied.
        varlen_mode: If True, variable-length sequence processing was enabled.
        residual: A tuple of tensors saved from the forward pass.
        dO: The gradient of the output tensor.

    Returns:
        A tuple of gradients (dq, dk, dv, None, None, None) corresponding
        to the inputs (q, k, v, attention_mask, bias, ...).
    """
    q, k, v, bias, attention_mask, out, lse, dropout_seed = residual
    dq, dk, dv = bwd_triton_impl(
        dO=dO,
        q=q,
        k=k,
        v=v,
        bias=bias,
        attention_mask=attention_mask,
        o=out,
        M=lse,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        softmax_scale=softmax_scale,
        varlen_mode=varlen_mode,
    )
    return dq, dk, dv, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 9))
@functools.partial(jax.jit, static_argnums=(5, 6, 7, 9))
def flash_attention_call(
    q: jax.Array | None,
    k: jax.Array | None,
    v: jax.Array | None,
    attention_mask: jax.Array | None = None,
    bias: jax.Array | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    varlen_mode: bool = True,
) -> jax.Array:
    """
    Computes Flash Attention using a Triton kernel via a custom VJP.

    This function is JIT-compiled and has a custom gradient definition for
    memory-efficient training and inference. It serves as the core
    implementation that can handle both standard and variable-length attention.

    Args:
        q: Query tensor with shape `(batch, heads, q_seq_len, dim_per_head)`.
        k: Key tensor with shape `(batch, heads, k_seq_len, dim_per_head)`.
        v: Value tensor with shape `(batch, heads, k_seq_len, dim_per_head)`.
        attention_mask: Optional attention mask to apply.
        bias: Optional bias tensor to add to the attention scores.
        softmax_scale: Optional float scaling factor to apply before softmax. If
            None, it defaults to `1 / sqrt(dim_per_head)`.
        dropout_prob: Dropout probability applied to the attention scores.
        causal: If True, applies a causal mask to prevent attending to future
            tokens.
        dropout_seed: Optional seed for dropout for reproducibility.
        varlen_mode: If True, enables variable-length sequence processing, which
            is more efficient for inputs with different sequence lengths.

    Returns:
        The attention output tensor with shape
        `(batch, heads, q_seq_len, dim_per_head)`.
    """
    return fwd_triton_impl(
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        varlen_mode=varlen_mode,
    )[0]


flash_attention_call.defvjp(_fwd_call, _bwd_call)


def flash_attention(
    q: jax.Array | None,
    k: jax.Array | None,
    v: jax.Array | None,
    attention_mask: jax.Array | None = None,
    bias: jax.Array | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    varlen_mode: bool = True,
) -> jax.Array:
    """
    User-facing function for Flash Attention for standard (non-variable length) sequences.

    This function is a wrapper around `flash_attention_call` that defaults to
    non-variable length mode. It provides a simplified interface for common use
    cases where all sequences in a batch are of the same length.

    Args:
        q: Query tensor with shape `(batch, heads, q_seq_len, dim_per_head)`.
        k: Key tensor with shape `(batch, heads, k_seq_len, dim_per_head)`.
        v: Value tensor with shape `(batch, heads, k_seq_len, dim_per_head)`.
        attention_mask: Optional attention mask to apply.
        bias: Optional bias tensor to add to the attention scores.
        softmax_scale: Optional float scaling factor to apply before softmax. If
            None, it defaults to `1 / sqrt(dim_per_head)`.
        dropout_prob: Dropout probability applied to the attention scores.
        causal: If True, applies a causal mask to prevent attending to future
            tokens.
        dropout_seed: Optional seed for dropout for reproducibility.
        varlen_mode: This argument is ignored and always set to `False`.

    Returns:
        The attention output tensor with shape
        `(batch, heads, q_seq_len, dim_per_head)`.
    """
    del varlen_mode
    return flash_attention_call(
        q=q,
        k=k,
        v=v,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        varlen_mode=False,
    )
