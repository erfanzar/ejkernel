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
from ._kernel import Array, Bool, BwdParams, Callable, Float, FwdParams, Int
from ._kernel import blocksparse_attention as _blocksparse_attention_impl


@kernel_registry.register("blocksparse_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"]
    | Int[Array, "batch num_heads_or_1 seq_len kv_len"]
    | None = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Pallas TPU block-sparse attention kernel implementation.
    Computes attention over sparse block patterns using Pallas kernels optimized for TPU execution.
    Args:
        query: Query tensor [batch num_heads seq_len head_dim]
        key: Key tensor [batch kv_num_heads kv_len head_dim]
        value: Value tensor [batch kv_num_heads kv_len vhead_dim]
        q_segment_ids: Optional query segment ids [batch, seq_len]
        kv_segment_ids: Optional KV segment ids [batch, kv_len]
        q_positions: Optional query position indices [batch, seq_len] (not implemented for TPU)
        kv_positions: Optional KV position indices [batch, kv_len] (not implemented for TPU)
        softmax_aux: Optional auxiliary softmax values for attention sinks
        bias: Optional attention bias [batch num_heads seq_len head_dim]
        sequence_parallelism_mesh_axis_name: Optional mesh axis name for sequence parallelism
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            This prevents attention scores from becoming too large, improving numerical
            stability (Gemma-2 style). Gradients are computed with proper Jacobian.
        qkv_layouts: Optional pre-computed attention mask layouts
        softmax_scale: Attention score scaling factor (default: 1/sqrt(head_dim))
        mask_builder: Custom mask builder function
        sliding_window: Sliding window size. Can be:
            - int: symmetric window (same size left and right)
            - tuple[int, int]: (left_window, right_window) for asymmetric
            - None: no sliding window
        chunk_size: Size of chunks for chunked causal attention (like Llama4)
            - int: enable chunked causal mask with specified chunk size
            - None: no chunking
        causal: Whether to use causal masking (default True)
        fused_backward: Whether to use fused backward kernel
    Returns:
        Attention output [batch num_heads seq_len vhead_dim]
    """
    return _blocksparse_attention_impl(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_positions,
        kv_positions,
        softmax_aux,
        bias,
        attention_mask,
        sequence_parallelism_mesh_axis_name,
        logits_soft_cap,
        qkv_layouts,
        softmax_scale,
        fwd_params,
        bwd_params,
        mask_builder,
        sliding_window,
        chunk_size,
        causal,
        fused_backward,
    )


__all__ = ("blocksparse_attention",)
