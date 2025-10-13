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


"""Ragged Paged Attention implementation using Triton kernels.

This module implements ragged paged attention, an extension of paged attention
that efficiently handles batches of sequences with highly variable lengths.
Unlike standard paged attention which processes one query per sequence, ragged
paged attention processes multiple queries per sequence in a single batch,
making it ideal for prefill operations during LLM inference.

Key differences from standard page_attention:
1. **Ragged queries**: Multiple queries per sequence packed into a single tensor
2. **Query-level granularity**: Each query token can attend to the appropriate
   portion of the KV cache based on its position
3. **Prefill-optimized**: Designed for processing prompt tokens efficiently
4. **Combined KV format**: Keys and values are interleaved in memory

The "ragged" nature refers to handling variable-length sequences in a packed
format, where query_start_loc indicates the boundaries between sequences.

Architecture:
- Queries from multiple sequences are concatenated: [seq0_queries, seq1_queries, ...]
- Each query knows its position within its sequence via metadata
- KV cache is organized in pages, with each page containing both K and V
- Block tables map logical pages to physical pages for each sequence

Use cases:
- Prefill phase: Processing entire prompts before generation
- Chunked prefill: Processing long prompts in multiple passes
- Variable-length batching: Efficiently batching requests of different lengths

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.ragged_page_attention import ragged_page_attention
    >>>
    >>>
    >>> total_tokens = 16
    >>> num_q_heads, head_dim = 12, 64
    >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
    >>>
    >>>
    >>> num_pages, page_size, num_kv_heads = 50, 16, 12
    >>> kv_pages = jnp.ones((num_pages, page_size, 2 * num_kv_heads, head_dim))
    >>>
    >>>
    >>> num_seqs = 3
    >>> context_lens = jnp.array([5, 8, 3])
    >>> query_start_loc = jnp.array([0, 5, 13, 16])
    >>> block_tables = jnp.zeros((num_seqs, 10), dtype=jnp.int32)
    >>>
    >>> output = ragged_page_attention(
    ...     queries, kv_pages, context_lens, block_tables,
    ...     query_start_loc, num_seqs
    ... )
    >>> print(output.shape)

Reference:
    vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention
    https://arxiv.org/abs/2309.06180
"""

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.callib import cdiv, triton_call

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import _ragged_paged_attn_prefetch_kernel_combined

DEFAULT_MASK_VALUE = -2.381976426469702e38


@kernel_registry.register("ragged_page_attention", Platform.TRITON, Backend.GPU)
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
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Compute ragged paged attention for variable-length sequences.

    This function efficiently processes multiple variable-length sequences in a
    single batch, where queries from all sequences are packed into a flat tensor.
    It's particularly useful for the prefill phase of LLM inference where entire
    prompts of varying lengths need to be processed.

    The KV cache is organized in pages, with keys and values interleaved in the
    same tensor (combined format). Each sequence can span multiple pages, and
    the block_tables parameter maps logical page indices to physical page locations.

    Args:
        queries: Packed query tensor of shape (total_tokens, num_q_heads, head_dim),
            where total_tokens is the sum of all sequence lengths. Queries from
            different sequences are concatenated.
        kv_pages: Paged KV cache of shape (num_pages, page_size, num_combined_kv_heads, head_dim),
            where num_combined_kv_heads = 2 * num_kv_heads (keys and values interleaved).
            The first half of the head dimension contains keys, the second half values.
        context_lens: Context length for each sequence, shape (num_seqs,). Specifies
            how many KV tokens are valid for each sequence.
        block_tables: Page table mapping logical to physical pages, shape
            (num_seqs, pages_per_seq). For each sequence, maps logical page indices
            to physical page indices in kv_pages. Use -1 or any invalid index for
            unused page slots.
        query_start_loc: Cumulative query offsets, shape (num_seqs + 1,). Indicates
            where each sequence's queries start in the packed queries tensor.
            Example: [0, 5, 13, 16] means sequence 0 has queries 0:5, sequence 1
            has queries 5:13, sequence 2 has queries 13:16.
        num_seqs: Number of sequences in the batch. Can be an integer or a
            scalar JAX array.
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        logits_soft_cap: Optional soft capping value for attention logits. When specified,
            applies tanh-based soft capping: logits_soft_cap * tanh(logits / logits_soft_cap).
            Helps with numerical stability (e.g., Gemma-2 uses 20.0).
        compute_dtype: Computation dtype (ignored in Triton implementation).
        optimized: Optimization flag (ignored in Triton implementation).
        sliding_window: Optional sliding window size for local attention. If specified,
            each query only attends to the last `sliding_window` tokens.
        softmax_aux: Not supported in Triton implementation (raises error if provided).
        mask_value: Value to use for masked positions. Defaults to -2.38e38.
        num_kv_pages_per_block: Number of KV pages to process per block. Higher
            values may improve performance but increase memory usage. Defaults to 8.
        num_queries_per_block: Not supported in Triton (TPU-specific parameter).
        vmem_limit_bytes: Not supported in Triton (TPU-specific parameter).

    Returns:
        Attention output of shape (total_tokens, num_q_heads, head_dim), with
        results packed in the same order as the input queries.

    Raises:
        NotImplementedError: If softmax_aux, num_queries_per_block, or vmem_limit_bytes
            are provided (these are TPU-specific features).
        AssertionError: If combined KV heads is not even, or if dimensions mismatch.

    Example:
        >>> import jax.numpy as jnp
        >>> from ejkernel.kernels._triton.ragged_page_attention import ragged_page_attention
        >>>
        >>>
        >>> num_seqs = 3
        >>> query_lens = [4, 6, 3]
        >>> total_tokens = sum(query_lens)
        >>>
        >>>
        >>> num_q_heads, head_dim = 8, 64
        >>> queries = jnp.ones((total_tokens, num_q_heads, head_dim))
        >>>
        >>>
        >>> num_pages, page_size, num_kv_heads = 20, 16, 8
        >>> kv_pages = jnp.ones((num_pages, page_size, 2 * num_kv_heads, head_dim))
        >>>
        >>>
        >>> context_lens = jnp.array([10, 20, 8])
        >>> query_start_loc = jnp.array([0, 4, 10, 13])
        >>> block_tables = jnp.array([
        ...     [0, 1, -1, -1],
        ...     [2, 3, 4, -1],
        ...     [5, -1, -1, -1],
        ... ])
        >>>
        >>> output = ragged_page_attention(
        ...     queries, kv_pages, context_lens, block_tables,
        ...     query_start_loc, num_seqs
        ... )
        >>> print(output.shape)
    """
    del compute_dtype, optimized
    T, QH, D = queries.shape
    if softmax_aux is not None:
        raise NotImplementedError("`softmax_aux` is not implemented in triton impl yet!")

    if num_queries_per_block is not None:
        raise NotImplementedError("num_queries_per_block is TPU-specific and not supported on GPU")
    if vmem_limit_bytes is not None:
        raise NotImplementedError("vmem_limit_bytes is TPU-specific and not supported on GPU")

    if softmax_scale is None:
        softmax_scale = 1.0 / (D**0.5)

    kv_pages_per_block = num_kv_pages_per_block if num_kv_pages_per_block is not None else 8

    if mask_value is None:
        mask_value = -2.381976426469702e38

    use_soft_cap = logits_soft_cap is not None
    soft_cap_value = logits_soft_cap if use_soft_cap else 1.0

    use_sliding_window = sliding_window is not None
    sliding_window_size = sliding_window if use_sliding_window else 0
    _P, PS, C, Dk = kv_pages.shape
    assert D == Dk, "head_size mismatch"
    assert C % 2 == 0, "combined kv heads must be even"
    KVH = C // 2
    assert QH % KVH == 0
    QHG = QH // KVH
    pages_per_seq_max = int(block_tables.shape[1])

    q4 = queries.reshape(T, KVH, QHG, D)
    T_padded = max(T, 1)
    if T_padded > T:
        pad = jnp.zeros((T_padded - T, KVH, QHG, D), dtype=q4.dtype)
        q4 = jnp.concatenate([q4, pad], axis=0)

    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    q_lens = (ends - starts).astype(jnp.int32)

    t_idx = jnp.arange(T_padded, dtype=jnp.int32)
    t_clamped = jnp.minimum(t_idx, jnp.int32(max(T - 1, 0)))
    row_seq = jnp.searchsorted(ends, t_clamped, side="right").astype(jnp.int32)

    row_start = starts[row_seq]
    row_qlen = q_lens[row_seq]
    row_kvlen = context_lens[row_seq]
    row_qoff = t_idx - row_start
    row_firstk = (row_kvlen - row_qlen + row_qoff).astype(jnp.int32)

    ns_dev = jnp.asarray(num_seqs, dtype=jnp.int32)
    row_valid = (t_idx < T) & (row_seq < ns_dev)

    KV_PAGES_PER_BLOCK = int(kv_pages_per_block)
    MAX_KV_SUPERBLOCKS = cdiv(pages_per_seq_max, KV_PAGES_PER_BLOCK)

    out_shape = jax.ShapeDtypeStruct((T_padded, KVH, QHG, D), jnp.float32)

    def grid(meta):
        return (T_padded, KVH, QHG)

    metaparams = dict(
        grid=grid,
        T=T_padded,
        KVH=KVH,
        QHG=QHG,
        D=D,
        PS=PS,
        PAGES_PER_SEQ_MAX=pages_per_seq_max,
        KV_PAGES_PER_BLOCK=KV_PAGES_PER_BLOCK,
        MAX_KV_SUPERBLOCKS=int(MAX_KV_SUPERBLOCKS),
        SCALE=softmax_scale,
        SOFT_CAP=soft_cap_value,
        USE_SOFT_CAP=use_soft_cap,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        USE_SLIDING_WINDOW=use_sliding_window,
        MASK_VALUE=mask_value,
    )

    out4_padded = triton_call(
        q4,
        kv_pages,
        block_tables.astype(jnp.int32),
        row_seq.astype(jnp.int32),
        row_firstk.astype(jnp.int32),
        row_kvlen.astype(jnp.int32),
        row_valid.astype(jnp.bool_),
        kernel=_ragged_paged_attn_prefetch_kernel_combined,
        out_shape=out_shape,
        name="ejkernel::triton::ragged_page_attn",
        **metaparams,
    )

    return out4_padded[:T].reshape(T, QH, D).astype(queries.dtype)
