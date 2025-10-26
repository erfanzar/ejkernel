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


import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _ragged_paged_attention


@kernel_registry.register("ragged_page_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Performs paged attention for batched, ragged sequences with optional attention sinks.

    This function implements a FlashAttention-style algorithm to compute attention
    for multiple sequences of varying lengths. The Key-Value (KV) cache for these
    sequences is stored in non-contiguous memory blocks called "pages". This is
    a common technique in LLM inference servers to manage memory efficiently.

    The attention is computed by iterating through blocks of queries and, for each
    query block, iterating through the relevant blocks of key-value pages. An
    online softmax algorithm is used to compute the attention output in a single
    pass over the KV data, which is memory-efficient.

    Args:
        queries: The query tensor for all sequences, concatenated together.
            Shape: `[total_query_tokens, num_q_heads, head_size]`.
        kv_pages: The paged Key/value cache.
            Shape: `[num_pages, page_size, num_kv_heads_combined, head_size]`.
        context_lens: The total length of each sequence in the KV cache.
            Shape: `[num_seqs]`.
        block_tables: A map from each sequence to its list of
            physical page indices in the KV cache.
            Shape: `[num_seqs, max_pages_per_sequence]`.
        query_start_loc: The cumulative sum of query lengths for each
            sequence, used to index into the `queries` tensor.
            Shape: `[num_seqs + 1]`.
        num_seqs: The total number of sequences in the batch which should be a shape[1] int32.
        softmax_scale: The scaling factor to apply to the attention scores
            before the softmax operation (typically `1 / sqrt(head_size)`).
        logits_soft_cap: An optional value to cap the attention scores with `tanh`.
        compute_dtype: The dtype to use for computation (default: bfloat16).
        optimized: Whether to use the optimized implementation (default: False).
        sliding_window: Optional sliding window size for local attention.
        softmax_aux: Optional attention sink logits of shape [num_kv_heads, num_sinks] or [num_sinks].
            These are auxiliary logits that participate in softmax normalization but don't
            contribute to output, allowing the model to absorb probability mass.

    Returns:
        The attention output tensor, with the same shape and dtype as `queries`.
        Shape: `[total_query_tokens, num_q_heads, head_size]`.

    Note:
        Attention sinks are learnable parameters that participate in the softmax normalization
        but don't produce output. They allow the model to "dump" attention probability mass,
        improving numerical stability and model behavior.
    """
    del mask_value, num_kv_pages_per_block, num_queries_per_block, vmem_limit_bytes
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    fn = _ragged_paged_attention
    return fn(
        queries=queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        logits_soft_cap=logits_soft_cap,
        compute_dtype=compute_dtype,
        sliding_window=sliding_window,
        softmax_aux=softmax_aux,
    )
