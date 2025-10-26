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
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.callib import ejit

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_fwd import (
    DEFAULT_MASK_VALUE,
    cdiv,
    get_min_heads_per_blk,
    ragged_page_attention_kernel,
    static_validate_inputs,
)
from ._utils import get_tuned_block_sizes


@ejit(
    static_argnames=[
        "softmax_scale",
        "mask_value",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "sliding_window",
        "logits_soft_cap",
    ]
)
def _ragged_page_attention(
    q: Float[Array, "max_num_batched_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "total_num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "max_num_seqs"],
    block_tables: Int[Array, "max_num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "max_num_seqs_plus_one"],
    num_seqs: jax.Array,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Ragged paged attention that supports mixed prefill and decode.

    Args:
      q: concatenated all sequences' queries.
      kv_pages: paged KV cache. Normally in HBM.
      context_lens: padded kv lengths. Only the first num_seqs values are valid.
      block_tables: the first index indicates which page to use in the kv cache
        for each sequence. Only the first num_seqs values are valid.
      query_start_loc: the cumulative sum of the effective query lengths. Similar to
        context_lens, only the first num_seqs+1 values are valid.
      num_seqs: the dynamic number of sequences.
      softmax_scale: the softmax softmax_scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      logits_soft_cap: the logit soft cap for the attention.
      mask_value: mask value for causal mask.
      num_kv_pages_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      num_queries_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of the attention.
    """
    static_validate_inputs(
        q,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    num_q_tokens, num_q_heads, head_dim = q.shape
    _, page_size, num_combined_kv_heads, _ = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    _, pages_per_seq = block_tables.shape
    num_q_heads_per_blk, num_combined_kv_heads_per_blk = get_min_heads_per_blk(
        num_q_heads,
        num_combined_kv_heads,
        q.dtype,
        kv_pages.dtype,
    )
    num_q_per_blk = num_queries_per_block
    num_kv_pages_per_blk = num_kv_pages_per_block
    if num_q_per_blk is None or num_kv_pages_per_blk is None:
        num_kv_pages_per_blk, num_q_per_blk = get_tuned_block_sizes(
            q.dtype,
            kv_pages.dtype,
            num_q_heads_per_blk,
            num_combined_kv_heads_per_blk // 2,
            head_dim,
            page_size,
            num_q_tokens,
            pages_per_seq,
        )
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads
    num_q_blks = cdiv(num_q_tokens, num_q_per_blk)
    assert num_combined_kv_heads_per_blk % 2 == 0
    num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
    assert num_q_heads_per_blk % num_q_heads_per_kv_head == 0
    num_heads_blks = num_q_heads // num_q_heads_per_blk
    grid = (num_heads_blks, num_q_blks)

    def q_index_map(heads_blk_idx, q_blk_idx, *_):
        return (q_blk_idx, heads_blk_idx, 0)

    q_block_spec = pl.BlockSpec((num_q_per_blk, num_q_heads_per_blk, head_dim), q_index_map)
    in_specs = [q_block_spec, pl.BlockSpec(memory_space=pltpu.ANY)]
    out_specs = q_block_spec
    lm_scratch = pltpu.VMEM((num_kv_heads_per_blk, num_q_per_blk * num_q_heads_per_kv_head, 128), jnp.float32)
    acc_scratch = pltpu.VMEM((num_q_per_blk, num_q_heads_per_blk, head_dim), jnp.float32)
    double_buf_scratch = pltpu.VMEM(
        (2, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, head_dim),
        kv_pages.dtype,
    )
    scratch_shapes = [double_buf_scratch, pltpu.SemaphoreType.DMA((2,)), lm_scratch, lm_scratch, acc_scratch]
    scalar_prefetches = (context_lens, block_tables, query_start_loc, jnp.array((0, 0), jnp.int32), num_seqs)
    kernel = pl.pallas_call(
        functools.partial(
            ragged_page_attention_kernel,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            mask_value=mask_value,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar_prefetches),
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        out_shape=jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
        name="ragged_page_attention_kernel",
    )

    return kernel(*scalar_prefetches, q, kv_pages)


@kernel_registry.register("ragged_page_attention", Platform.PALLAS, Backend.TPU)
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
    """Ragged paged attention that supports mixed prefill and decode.

    Args:
      queries: concatenated all sequences' queries.
      kv_pages: paged KV cache. Normally in HBM.
      context_lens: padded kv lengths. Only the first num_seqs values are valid.
      block_tables: the first index indicates which page to use in the kv cache
        for each sequence. Only the first num_seqs values are valid.
      query_start_loc: the cumulative sum of the effective query lengths. Similar to
        context_lens, only the first num_seqs+1 values are valid.
      num_seqs: the dynamic number of sequences.
      softmax_scale: the softmax softmax_scale which will be applied to the Q@K^T.
      sliding_window: the sliding window size for the attention.
      logits_soft_cap: the logit soft cap for the attention.
      mask_value: mask value for causal mask.
      num_kv_pages_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      num_queries_per_block: number of kv pages to be processed in one flash
        attention block in the pallas kernel.
      vmem_limit_bytes: the vmem limit for the pallas kernel.

    Returns:
      The output of the attention.
    """
    del optimized, compute_dtype
    if softmax_aux is not None:
        raise NotImplementedError("`softmax_aux` is not implemented in tpu-pallas kernel yet!")
    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5
    return _ragged_page_attention(
        queries,
        kv_pages=kv_pages,
        context_lens=context_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        num_seqs=num_seqs,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
