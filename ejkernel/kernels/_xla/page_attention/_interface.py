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
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import _page_attention_fwd


@kernel_registry.register("page_attention", Platform.XLA, Backend.ANY)
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
    mask_value: float = -2.381976426469702e38,
    attn_logits_soft_cap: float | None = None,
    pages_per_compute_block: int | None = None,
    megacore_mode: str | None = None,
    inline_seq_dim: bool = True,
) -> Float[Array, "num_seqs num_heads head_dim"]:
    """
    Paged attention for efficient KV cache management using JAX/XLA.

    This function implements paged attention where the KV cache is organized into
    fixed-size blocks (pages). Each sequence maintains a block table that maps
    logical KV positions to physical block indices. This enables efficient memory
    management for variable-length sequences and dynamic batching.

    Args:
        query: Query tensor of shape [num_seqs, num_heads, head_dim]. Each sequence
            has a single query token (typically for decode/generation).
        key_cache: Paged key cache of shape [num_blocks, num_kv_heads, block_size, head_dim].
            The total KV cache is divided into blocks of size `block_size`.
        value_cache: Paged value cache of shape [num_blocks, num_kv_heads, block_size, head_dim].
            Must have the same structure as key_cache.
        context_lens: Context length for each sequence [num_seqs]. Indicates how many
            tokens are valid in the KV cache for each sequence.
        block_tables: Block table mapping [num_seqs, max_blocks]. For each sequence,
            maps logical block indices to physical block indices in the cache.
        attn_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).
        max_context_len: Maximum context length (not used in XLA implementation).
        num_splits: Number of splits for partitioned attention (not used in XLA implementation).
        mask_value: Value used for masking (not used in XLA implementation).
        attn_logits_soft_cap: Soft cap for attention logits (not used in XLA implementation).
        pages_per_compute_block: Pages per compute block (not used in XLA implementation).
        megacore_mode: Megacore parallelization mode (not used in XLA implementation).
        inline_seq_dim: Whether to inline sequence dimension (not used in XLA implementation).

    Returns:
        Attention output of shape [num_seqs, num_heads, head_dim].

    Notes:
        - Supports Grouped Query Attention (GQA) where num_heads >= num_kv_heads
        - Each sequence can use a different number of blocks based on context_lens
        - Blocks are indexed via block_tables to avoid fragmentation
        - This is a simpler version compared to ragged_page_attention which handles
          multiple query tokens per sequence

    Examples:
        >>> num_seqs, num_heads, head_dim = 2, 8, 64
        >>> num_kv_heads = 8
        >>> num_blocks, block_size = 10, 16
        >>>
        >>> query = jnp.ones((num_seqs, num_heads, head_dim))
        >>> key_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>> value_cache = jnp.ones((num_blocks, num_kv_heads, block_size, head_dim))
        >>> context_lens = jnp.array([48, 32])
        >>> block_tables = jnp.array([[0, 1, 2, -1], [3, 4, -1, -1]])
        >>>
        >>> output = page_attention(query, key_cache, value_cache,
        ...                         context_lens, block_tables)
        >>> output.shape
        (2, 8, 64)
    """
    if max_context_len is not None:
        raise NotImplementedError("max_context_len is not supported in XLA implementation")
    if num_splits != 0:
        raise NotImplementedError("num_splits is not supported in XLA implementation")
    if pages_per_compute_block is not None:
        raise NotImplementedError("pages_per_compute_block is not supported in XLA implementation")
    if megacore_mode is not None:
        raise NotImplementedError("megacore_mode is not supported in XLA implementation")
    if not inline_seq_dim:
        raise NotImplementedError("inline_seq_dim=False is not supported in XLA implementation")
    if attn_logits_soft_cap is not None:
        raise NotImplementedError("attn_logits_soft_cap is not supported in XLA implementation")

    if attn_scale is None:
        attn_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)

    block_size = key_cache.shape[2]

    return _page_attention_fwd(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        context_lens=context_lens,
        block_tables=block_tables,
        attn_scale=attn_scale,
        block_size=block_size,
    )
