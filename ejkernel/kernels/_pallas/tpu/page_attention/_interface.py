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
from collections.abc import Sequence
from typing import Literal

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
    paged_flash_attention_kernel,
    paged_flash_attention_kernel_inline_seq_dim,
)


@kernel_registry.register("page_attention", Platform.PALLAS, Backend.TPU)
@ejit(
    static_argnames=[
        "pages_per_compute_block",
        "attn_logits_soft_cap",
        "mask_value",
        "megacore_mode",
        "inline_seq_dim",
    ],
)
@jaxtyping.jaxtyped(typechecker=beartype)
def page_attention(
    query: Float[Array, "num_seqs num_heads head_dim"],
    key_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    value_cache: Float[Array, "num_blocks num_kv_heads block_size head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs max_blocks"],
    attn_scale: float | None = None,
    max_context_len: int | None = None,
    num_splits: int = 0,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """Paged grouped query attention.

    Args:
      query: A [batch_size, num_q_heads, head_dim] jax.Array.
      key_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
      value_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
      context_lens: A i32[batch_size] jax.Array the length of each example.
      block_tables: A i32[batch_size, pages_per_sequence] jax.Array. Each entry
        should be in the range of [0, total_num_pages), indicating where to locate
        the page in `key_cache` or `value_cache`.
      attn_scale: Attention scaling factor (not used in PALLAS TPU implementation).
      max_context_len: Maximum context length (not used in PALLAS TPU implementation).
      num_splits: Number of splits for partitioned attention (not used in PALLAS TPU implementation).
      mask_value: The value used for padding in attention. By default it is a very
        negative floating point number.
      attn_logits_soft_cap: The value used for soft capping the attention logits.
      pages_per_compute_block: how many pages to be processed in one flash
        attention block in the pallas kernel.
      megacore_mode: if set, enable megacore to parallelize the computation. Must
        be one of ['kv_head', 'batch', None]. Caveat: set this only if megacore is
        enabled, otherwise the kernel may hang. If you are not sure, leave it to
        None.
        * None: disable megacore parallelism.
        * kv_head: megacore parallelism on KV heads; requires number of KV heads
          divisible by 2.
        * batch: megacore parallelism on batch dimension; requires batch divisible
          by 2.
      inline_seq_dim: whether to fuse kernel instances along the sequence dim into
        one kernel.

    Returns:
      The output of attention([batch_size, num_q_heads, head_dim]).
    """

    if attn_scale is not None:
        raise NotImplementedError("attn_scale is not supported in PALLAS TPU implementation")
    if max_context_len is not None:
        raise NotImplementedError("max_context_len is not supported in PALLAS TPU implementation")
    if num_splits != 0:
        raise NotImplementedError("num_splits is not supported in PALLAS TPU implementation")

    if pages_per_compute_block is None:
        raise ValueError("pages_per_compute_block is required for PALLAS TPU implementation")

    batch_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _, page_size, head_dim_k = key_cache.shape
    batch_size_paged_indices, pages_per_sequence = block_tables.shape

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
    if pages_per_sequence % pages_per_compute_block != 0:
        raise ValueError(
            "pages_per_compute_block must be divisible by pages per sequence. Got"
            f" {pages_per_compute_block} and {pages_per_sequence}."
        )
    if context_lens.shape != (batch_size,):
        raise ValueError("`context_lens` and `query` must have the same batch size")
    if batch_size_paged_indices != batch_size:
        raise ValueError("`block_tables` and `query` must have the same batch size")
    if context_lens.dtype != jnp.int32:
        raise ValueError("The dtype of `context_lens` must be int32. Got {context_lens.dtype}")

    if megacore_mode == "kv_head":
        if num_kv_heads % 2 != 0:
            raise ValueError("number of KV heads must be even when megacore_mode is 'kv_head'")
        num_cores = 2
    elif megacore_mode == "batch":
        if batch_size % 2 != 0:
            raise ValueError("batch size must be even when megacore_mode is 'batch'")
        num_cores = 2
    elif megacore_mode is None:
        num_cores = 1
    else:
        raise ValueError("megacore_mode must be one of ['kv_head', 'batch', None]")

    num_groups = num_q_heads // num_kv_heads
    if (num_groups) % 8 != 0:
        q = query.reshape(batch_size, num_q_heads, 1, head_dim)
        if megacore_mode == "kv_head":
            q_block_spec = pl.BlockSpec(
                (None, num_groups, None, head_dim),
                lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0, 0),
            )
        elif megacore_mode == "batch":
            q_block_spec = pl.BlockSpec(
                (None, num_groups, None, head_dim),
                lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0, 0),
            )
        else:
            q_block_spec = pl.BlockSpec(
                (None, num_groups, None, head_dim),
                lambda core_index, b, h, *_: (b, h, 0, 0),
            )
        q_dtype_for_kernel_launch = jnp.float32
    else:
        if megacore_mode == "kv_head":
            q_block_spec = pl.BlockSpec(
                (None, num_groups, head_dim),
                lambda core_index, b, h, *_: (b, h * num_cores + core_index, 0),
            )
        elif megacore_mode == "batch":
            q_block_spec = pl.BlockSpec(
                (None, num_groups, head_dim),
                lambda core_index, b, h, *_: (b * num_cores + core_index, h, 0),
            )
        else:
            q_block_spec = pl.BlockSpec(
                (None, num_groups, head_dim),
                lambda core_index, b, h, *_: (b, h, 0),
            )
        q_dtype_for_kernel_launch = query.dtype
        q = query

    dimension_semantics: Sequence[Literal["parallel", "arbitrary"]]
    if inline_seq_dim:
        kernel = paged_flash_attention_kernel_inline_seq_dim
        grid = (
            num_cores,
            batch_size // num_cores if megacore_mode == "batch" else batch_size,
            num_kv_heads // num_cores if megacore_mode == "kv_head" else num_kv_heads,
        )
        dimension_semantics = ("parallel", "arbitrary", "arbitrary")
    else:
        kernel = paged_flash_attention_kernel
        grid = (
            num_cores,
            batch_size // num_cores if megacore_mode == "batch" else batch_size,
            num_kv_heads // num_cores if megacore_mode == "kv_head" else num_kv_heads,
            pages_per_sequence // pages_per_compute_block,
        )
        dimension_semantics = ("parallel", "arbitrary", "arbitrary", "arbitrary")

    in_specs = [
        q_block_spec,
        pl.BlockSpec(memory_space=pltpu.ANY),
        pl.BlockSpec(memory_space=pltpu.ANY),
    ]
    scratch_shapes = (
        pltpu.VMEM((2, pages_per_compute_block, page_size, head_dim), key_cache.dtype),
        pltpu.VMEM((2, pages_per_compute_block, page_size, head_dim), value_cache.dtype),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    )

    out, _, _ = pl.pallas_call(
        functools.partial(
            kernel,
            pages_per_sequence=pages_per_sequence,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            megacore_mode=megacore_mode,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            in_specs=in_specs,
            out_specs=[
                q_block_spec,
                q_block_spec,
                q_block_spec,
            ],
            grid=grid,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=dimension_semantics),
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q_dtype_for_kernel_launch),
            jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
            jax.ShapeDtypeStruct((*q.shape[:-1], 1), jnp.float32),
        ],
    )(
        context_lens,
        block_tables.reshape(-1),
        jnp.zeros((1,), jnp.int32),
        jnp.ones((1,), jnp.int32),
        q.astype(q_dtype_for_kernel_launch),
        key_cache,
        value_cache,
    )
    return out.reshape(batch_size, num_q_heads, head_dim).astype(query.dtype)
