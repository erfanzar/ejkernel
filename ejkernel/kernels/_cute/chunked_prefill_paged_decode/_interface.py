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

"""Chunked prefill + paged decode attention implemented in CuTe (GPU)."""

from __future__ import annotations

import math

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._cute_impl_fwd import chunked_prefill_paged_decode_cute


@kernel_registry.register("chunked_prefill_paged_decode", Platform.CUTE, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def chunked_prefill_paged_decode(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    keys: Float[Array, "total_tokens num_kv_heads head_dim"],
    values: Float[Array, "total_tokens num_kv_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[
    Float[Array, "total_tokens num_q_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    Float[Array, "num_blocks block_size num_kv_heads head_dim"],
]:
    """CuTe GPU implementation of chunked prefill + paged decode attention.

    This implementation updates the block-tabled KV cache using a CuTe DSL
    kernel and computes the attention output on top of the updated cache.

    Args:
        queries: Packed query vectors [total_tokens, num_q_heads, head_dim]
        keys: Packed keys to insert into cache [total_tokens, num_kv_heads, head_dim]
        values: Packed values to insert into cache [total_tokens, num_kv_heads, head_dim]
        key_cache: Block-tabled key cache [num_blocks, block_size, num_kv_heads, head_dim]
        value_cache: Block-tabled value cache [num_blocks, block_size, num_kv_heads, head_dim]
        kv_lens: KV lengths per sequence [num_seqs]
        block_tables: Logical-to-physical block mapping [num_seqs, max_blocks_per_seq]
        query_start_loc: Cumulative query positions [num_seqs + 1]
        alibi_slopes: Optional ALiBi slopes [num_q_heads]
        softmax_aux: Optional auxiliary softmax parameters [num_q_heads]
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking (must be True)
        sliding_window: Optional local attention window size
        logits_soft_cap: Optional logit soft-capping value
        seq_threshold_3d: Optional Triton attention tuning hint
        num_par_softmax_segments: Optional Triton attention tuning hint
        num_warps: Optional Triton attention tuning hint
        num_stages: Optional Triton attention tuning hint

    Returns:
        Tuple of (attention_output, updated_key_cache, updated_value_cache).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(queries.shape[-1])
    return chunked_prefill_paged_decode_cute(
        queries=queries,
        keys=keys,
        values=values,
        key_cache=key_cache,
        value_cache=value_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        alibi_slopes=alibi_slopes,
        softmax_aux=softmax_aux,
        softmax_scale=float(softmax_scale),
        causal=bool(causal),
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )
