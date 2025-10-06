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


"""Interface for Flash Multi-Latent Attention (MLA) operations."""

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ..._registry import Backend, Platform, kernel_registry


@kernel_registry.register("flash_mla_attention_call", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_mla_attention_call(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len head_dim"],
    latent_key: Float[Array, "head_dim latent_dim"],
    latent_value: Float[Array, "head_dim latent_dim"],
    bias: Float[Array, "batch num_heads seq_len seq_len"] | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float[Array, "batch num_heads seq_len head_dim"]:
    """
    Execute Multi-Latent Attention using Triton kernels.

    Multi-Latent Attention reduces memory and computation by projecting
    key and value tensors to lower-dimensional latent spaces before
    computing attention.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim).
        key: Key tensor of shape (batch, heads, seq_len, head_dim).
        value: Value tensor of shape (batch, heads, seq_len, head_dim).
        latent_key: Latent key projection matrix of shape (head_dim, latent_dim).
        latent_value: Latent value projection matrix of shape (head_dim, latent_dim).
        bias: Optional attention bias of shape (batch, heads, seq_len, seq_len).
        causal: Whether to apply causal masking.
        softmax_scale: Scale factor for softmax. Defaults to 1/sqrt(head_dim).

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim).
    """
    raise NotImplementedError("Flash MLA attention kernel not yet implemented")


@kernel_registry.register("flash_mla", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_mla_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len head_dim"],
    latent_key: Float[Array, "head_dim latent_dim"],
    latent_value: Float[Array, "head_dim latent_dim"],
    bias: Float[Array, "batch num_heads seq_len seq_len"] | None = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Float[Array, "batch num_heads seq_len head_dim"]:
    """
    Multi-Latent Attention with automatic differentiation support.

    This function wraps flash_mla_attention_call with JAX's custom gradient
    support for efficient backpropagation through the attention operation.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim).
        key: Key tensor of shape (batch, heads, seq_len, head_dim).
        value: Value tensor of shape (batch, heads, seq_len, head_dim).
        latent_key: Latent key projection matrix of shape (head_dim, latent_dim).
        latent_value: Latent value projection matrix of shape (head_dim, latent_dim).
        bias: Optional attention bias of shape (batch, heads, seq_len, seq_len).
        causal: Whether to apply causal masking.
        softmax_scale: Scale factor for softmax. Defaults to 1/sqrt(head_dim).

    Returns:
        Output tensor of shape (batch, heads, seq_len, head_dim).
    """
    return flash_mla_attention_call(
        query=query,
        key=key,
        value=value,
        latent_key=latent_key,
        latent_value=latent_value,
        bias=bias,
        causal=causal,
        softmax_scale=softmax_scale,
    )
