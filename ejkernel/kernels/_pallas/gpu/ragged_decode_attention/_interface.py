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


from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import _ragged_decode_attention_call


@kernel_registry.register("ragged_decode_attention", Platform.PALLAS, Backend.GPU)
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
    """Performs attention decoding over ragged sequences using a GPU-optimized kernel.

    This function serves as the public API for decoding attention across variable-length
    sequences (ragged) using head-blocked GPU kernels. It supports multi-head attention (MHA),
    multi-query attention (MQA), and grouped-query attention (GQA) layouts.

    Args:
        query (chex.Array): Query tensor of shape (batch_size, num_query_heads, head_dim).
        key (chex.Array): Key tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        value (chex.Array): Value tensor of shape (batch_size, sequence_length, num_kv_heads, head_dim).
        sequence_start (chex.Array, optional): Optional start indices of valid sequence ranges, shape (batch_size,).
        sequence_end (chex.Array, optional): Optional end indices of valid sequence ranges, shape (batch_size,).
        softmax_scale (float, optional): Optional scaling factor for the attention softmax.
            Defaults to 1 / sqrt(head_dim) if not provided.
        block_size_heads (int): Size of the head dimension block. Affects tiling for attention computation.
        block_size_keys (int): Size of the key block per thread block.
        num_key_splits (int): Number of splits (tiles) in the key dimension.
        num_warps (int, optional): Number of GPU warps per thread block.
        num_stages (int): Pipeline stages for kernel execution.

    Returns:
        chex.Array: Output tensor of shape (batch_size, num_query_heads, head_dim) after attention is applied.

    Raises:
        ValueError: If `key` and `value` have different head dimensions.
        ValueError: If `query` heads are not divisible by the number of KV heads.

    """
    return _ragged_decode_attention_call(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
    )
