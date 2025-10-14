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


import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ejkernel.ops import FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import inner_decode_tpu


@kernel_registry.register("ragged_decode_attention", Platform.PALLAS, Backend.TPU)
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
    """Ragged MQA decoding entry point with TPU-accelerated Flash Attention.

    Args:
        query: Query tensor of shape [batch, num_heads, head_dim].
        key: Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
        value: Value tensor of shape [batch, seq_len, num_kv_heads, head_dim].
        sequence_start: int32 array of shape [batch], start indices of sequences.
        sequence_end: int32 array of shape [batch], end indices of sequences.
        softmax_scale: Optional scale for attention logits. Default is 1.
        block_size: Block size used for kernel tiling. Default is 256.
        sliding_window: Optional (left, right) sliding window sizes.
            If specified, limits attention to tokens within the window. None means full attention.
        logits_soft_cap: Optional soft capping value for attention logits.
            Applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
        softmax_aux: Optional auxiliary logits for attention sinks.
            Shape [num_heads, num_sinks] or [num_sinks]. Concatenated to attention logits
            before softmax to create attention sink behavior.

    Returns:
        Output tensor of shape [batch, num_heads, head_dim] after attention decoding.
    """
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return inner_decode_tpu(
        query=query,
        key=key,
        value=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
