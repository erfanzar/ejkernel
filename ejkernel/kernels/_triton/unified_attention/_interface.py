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

"""vLLM-style unified (paged) attention implemented in Triton.

This is a JAX/Triton port of vLLM's `triton_unified_attention.py`, adapted to
ejkernel's `triton_call` interface.

Core inputs:
- `queries`: packed ragged queries, shape `[total_tokens, num_q_heads, head_dim]`
- `key_cache`/`value_cache`: paged KV cache, shape `[num_blocks, block_size, num_kv_heads, head_dim]`
- `query_start_loc`: cumulative query offsets, shape `[num_seqs + 1]` (int32)
- `kv_lens`: KV lengths per sequence, shape `[num_seqs]` (int32)
- `block_tables`: mapping `[num_seqs, max_blocks_per_seq]` (int32)

Supported features (inference-only):
- causal masking (required)
- optional sliding window via `sliding_window` (window length)
- optional logit softcap (`logits_soft_cap`)
- optional attention sink (`attention_sink`): contributes to softmax normalizer only
- optional ALiBi slopes (`alibi_slopes`)
- optional query-query bias (`qq_bias`) for TreeAttention-like decode
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import unified_attention_triton


@kernel_registry.register("unified_attention", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    attention_sink: Float[Array, "num_q_heads"] | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    return unified_attention_triton(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        kv_lens=kv_lens,
        query_start_loc=query_start_loc,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        attention_sink=attention_sink,
        num_warps=num_warps,
        num_stages=num_stages,
    )
