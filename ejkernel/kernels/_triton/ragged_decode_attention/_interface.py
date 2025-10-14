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

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import inner_decode_triton


@kernel_registry.register("ragged_decode_attention", Platform.TRITON, Backend.GPU)
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
    """
    Ragged decode attention (GPU/Triton), functionally matching the TPU/Pallas version.

    Args:
        query: [B, HQ, D]
        key:   [B, S, HKV, D]
        value: [B, S, HKV, D]
        sequence_start: [B] int32 (inclusive)
        sequence_end:   [B] int32 (exclusive)
        softmax_scale: logits scale
        block_size: tile size along sequence axis
        sliding_window: optional (left, right) window; None => full attention
        logits_soft_cap: optional tanh-cap for logits
        softmax_aux: optional sinks:
            - [HKV, NS] (per kv head), or
            - [NS] (broadcast to each kv head)

    Returns:
        Output: [B, HQ, D]
    """

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    return inner_decode_triton(
        query_tensor=query,
        key_tensor=key,
        value_tensor=value,
        sequence_start=sequence_start,
        sequence_end=sequence_end,
        softmax_scale=softmax_scale,
        fwd_params=fwd_params,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
    )
