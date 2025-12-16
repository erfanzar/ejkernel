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


import functools

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float, Int

from ejkernel.callib import ejit

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import (
    DEFAULT_MASK_VALUE,
    chunked_prefill_attention_kernel,
)


@kernel_registry.register("prefill_page_attention", Platform.PALLAS, Backend.TPU)
@ejit(
    static_argnames=[
        "softmax_scale",
        "mask_value",
        "attn_logits_soft_cap",
        "sliding_window",
    ],
)
@jaxtyping.jaxtyped(typechecker=beartype)
def prefill_page_attention(
    query: Float[Array, "chunk_size num_heads head_dim"],
    key_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    value_cache: Float[Array, "num_kv_heads total_num_pages page_size head_dim"],
    context_len: Int[Array, "1"],
    page_indices: Int[Array, "num_pages"],
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
) -> Float[Array, "chunk_size num_heads head_dim"]:
    """Chunked prefill attention with paged KV cache for TPU.

    This kernel processes a chunk of query tokens during prefill phase,
    reading from a paged KV cache. It supports causal masking and sliding window.

    Args:
        query: Query tensor [chunk_size, num_q_heads, head_dim] for prefill tokens.
        key_cache: Paged key cache [num_kv_heads, total_num_pages, page_size, head_dim].
        value_cache: Paged value cache [num_kv_heads, total_num_pages, page_size, head_dim].
        context_len: Total context length including this chunk [1].
        page_indices: Page indices for this sequence [num_pages].
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim)).
        mask_value: Value used for masked positions in attention.
        attn_logits_soft_cap: Optional soft cap for attention logits.
        sliding_window: If set, only attend to the last `sliding_window` tokens.

    Returns:
        Attention output [chunk_size, num_q_heads, head_dim].

    Note:
        - This is designed for prefill phase where we process multiple query tokens
        - Uses causal masking so each query can only attend to itself and past tokens
        - The KV cache should already contain the keys/values for this sequence
        - chunk_size must be divisible by page_size
    """
    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, head_dim_k = key_cache.shape

    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache and value_cache must have the same shape. Got {key_cache.shape} and {value_cache.shape}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"Number of Q heads must be divisible by number of KV heads. Got {num_q_heads} and {num_kv_heads}."
        )
    if head_dim_k != head_dim:
        raise ValueError(f"head_dim of Q must be the same as that of K/V. Got {head_dim} and {head_dim_k}.")
    if chunk_size % page_size != 0:
        raise ValueError(f"chunk_size must be divisible by page_size. Got {chunk_size} and {page_size}.")

    attn_group_size = num_q_heads // num_kv_heads
    pages_per_chunk = chunk_size // page_size
    num_pages = page_indices.shape[0]
    # Number of KV chunks is determined by pages (static shape)
    num_kv_chunks = num_pages // pages_per_chunk

    # Apply softmax scale to query
    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    # Transpose query for kernel: [chunk_size, num_q_heads, head_dim] -> [num_q_heads, chunk_size, head_dim]
    # Then reshape for GQA: [num_kv_heads, group_size, chunk_size, head_dim]
    q = query.transpose((1, 0, 2))  # [num_q_heads, chunk_size, head_dim]
    q = q.reshape(num_kv_heads, attn_group_size, chunk_size, head_dim)
    q = q * softmax_scale

    # Block specs need to match the 4D shape of q: [num_kv_heads, group_size, chunk_size, head_dim]
    # Grid iterates over num_kv_heads, so we select one kv_head at a time
    q_block_spec = pl.BlockSpec((1, attn_group_size, chunk_size, head_dim), lambda i, *_: (i, 0, 0, 0))
    lm_block_spec = pl.BlockSpec((1, attn_group_size, chunk_size, 1), lambda i, *_: (i, 0, 0, 0))
    lm_shape = jax.ShapeDtypeStruct(shape=(num_kv_heads, attn_group_size, chunk_size, 1), dtype=jnp.float32)

    out, _, _ = pl.pallas_call(
        functools.partial(
            chunked_prefill_attention_kernel,
            chunk_size=chunk_size,
            page_size=page_size,
            num_kv_chunks=num_kv_chunks,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            sliding_window=sliding_window,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[
                q_block_spec,
                pl.BlockSpec(memory_space=pltpu.ANY),
                pl.BlockSpec(memory_space=pltpu.ANY),
            ],
            out_specs=[
                q_block_spec,
                lm_block_spec,
                lm_block_spec,
            ],
            scratch_shapes=[
                pltpu.VMEM((2, pages_per_chunk, page_size, head_dim), key_cache.dtype),
                pltpu.VMEM((2, pages_per_chunk, page_size, head_dim), value_cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
            grid=(num_kv_heads,),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((num_kv_heads, attn_group_size, chunk_size, head_dim), q.dtype),
            lm_shape,
            lm_shape,
        ],
    )(
        context_len.reshape((1,)),
        page_indices,
        jnp.asarray([0], jnp.int32),
        q,
        key_cache,
        value_cache,
    )

    # Transpose output back: [num_kv_heads, group_size, chunk_size, head_dim] -> [chunk_size, num_q_heads, head_dim]
    out = out.reshape(num_q_heads, chunk_size, head_dim)
    out = out.transpose((1, 0, 2))

    return out.astype(query.dtype)
