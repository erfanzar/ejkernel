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

"""Ring Attention Implementation using Triton Flash Attention.

This module provides a ring attention implementation that wraps the Triton
flash attention kernel for distributed execution across multiple GPU devices.

Key features:
- Uses flash attention as the inner kernel for optimized GPU execution
- Supports distributed ring topology via lax.ppermute
- All flash attention features (causal, sliding window, dropout, etc.)
- Full backward pass support with custom VJP
"""

from __future__ import annotations

import typing
from collections.abc import Callable

import jaxtyping
from beartype import beartype
from jax import Array
from jaxtyping import Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._ring_kernel import ring_flash_attention_call

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


@kernel_registry.register("ring_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_heads"] | Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    fused_backward: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Ring attention using Triton flash attention kernels.

    Distributes attention computation across devices using a ring topology,
    where each device holds its query partition and rotates key/value blocks
    through all devices, computing partial attention and combining results
    using online softmax.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        attention_mask: Optional attention mask
        bias: Optional attention bias tensor
        softmax_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
        dropout_prob: Dropout probability (default: 0.0)
        causal: Whether to use causal masking (default: False)
        dropout_seed: Random seed for dropout
        sliding_window: Sliding window size. Can be:
            - int: symmetric window (same size left and right)
            - tuple[int, int]: (left_window, right_window) for asymmetric
            - None: no sliding window
        logits_soft_cap: Soft cap value for attention logits (tanh-based capping)
        axis_name: Name of the axis for ring communication (None for single device)
        fwd_params: Forward pass block size parameters
        bwd_params: Backward pass block size parameters

    Returns:
        Output tensor [batch, seq_len_q, num_heads, head_dim]

    Example:
        >>> # Basic causal ring attention
        >>> output = ring_attention(q, k, v, causal=True, axis_name="sp")

        >>> # With sliding window
        >>> output = ring_attention(
        ...     q, k, v,
        ...     sliding_window=256,
        ...     causal=True,
        ...     axis_name="sp",
        ... )
    """
    # Set defaults
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)

    return ring_flash_attention_call(
        query=query,
        key=key,
        value=value,
        attention_mask=None,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=0.0,
        causal=causal,
        dropout_seed=None,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        axis_name=axis_name,
    )
