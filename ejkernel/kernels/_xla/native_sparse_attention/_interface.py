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


import warnings
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _sparse_attention_bwd
from ._xla_impl_fwd import _sparse_attention_fwd


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _sparse_attention_with_vjp(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    block_indices: Int[Array, "batch num_heads num_query_blocks num_key_blocks"],
    block_counts: Int[Array, "batch num_heads num_query_blocks"],
    block_size: int,
    scale: float,
) -> Float[Array, "batch seq_len num_heads head_dim"]:
    """Sparse attention with custom VJP."""
    return _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, scale)


def _sparse_attention_fwd_vjp(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    block_indices: Int[Array, "batch num_heads num_query_blocks num_key_blocks"],
    block_counts: Int[Array, "batch num_heads num_query_blocks"],
    block_size: int,
    scale: float,
) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], tuple]:
    """Forward pass storing residuals."""
    output = _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, scale)
    residuals = (query, key, value, block_indices, block_counts)
    return output, residuals


def _sparse_attention_bwd_vjp(
    block_size: int,
    scale: float,
    residuals: tuple,
    do: Float[Array, "batch seq_len num_heads head_dim"],
) -> tuple:
    """Backward pass with custom gradient computation."""
    query, key, value, block_indices, block_counts = residuals
    dq, dk, dv = _sparse_attention_bwd(query, key, value, block_indices, block_counts, block_size, scale, do)
    return (dq, dk, dv, None, None)


_sparse_attention_with_vjp.defvjp(_sparse_attention_fwd_vjp, _sparse_attention_bwd_vjp)


def _nsa_compression_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    v_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    block_size: int,
    scale: float,
) -> tuple[Float[Array, "batch seq_len num_q_heads head_dim"], Float[Array, "batch seq_len num_q_heads"]]:
    """
    Compute compressed attention over mean-pooled key/value blocks with GQA support.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k_cmp: Compressed (mean-pooled) keys [batch, num_blocks, num_kv_heads, head_dim]
        v_cmp: Compressed (mean-pooled) values [batch, num_blocks, num_kv_heads, head_dim]
        block_size: Size of each block
        scale: Attention scaling factor

    Returns:
        Tuple of (output, log_sum_exp) where:
            - output: [batch, seq_len, num_q_heads, head_dim]
            - lse: [batch, seq_len, num_q_heads]
    """
    _batch, seq_len, num_q_heads, _head_dim = query.shape
    num_kv_heads = k_cmp.shape[2]
    group_size = num_q_heads // num_kv_heads
    num_blocks = k_cmp.shape[1]

    k_cmp_expanded = jnp.repeat(k_cmp, group_size, axis=2)
    v_cmp_expanded = jnp.repeat(v_cmp, group_size, axis=2)

    scores = jnp.einsum("bsnd,bmnd->bsnm", query, k_cmp_expanded) * scale

    token_block_idx = jnp.arange(seq_len) // block_size
    block_mask = jnp.arange(num_blocks)[None, :] <= token_block_idx[:, None]
    block_mask = block_mask[None, :, None, :]
    scores = jnp.where(block_mask, scores, -1e9)

    lse = jax.nn.logsumexp(scores, axis=-1)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    output = jnp.einsum("bsnm,bmnd->bsnd", attn_weights, v_cmp_expanded)

    return output, lse


def _nsa_topk_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    lse: Float[Array, "batch seq_len num_q_heads"],
    block_counts: int,
    block_size: int,
    scale: float,
) -> Int[Array, "batch num_kv_heads num_query_blocks num_selected_blocks"]:
    """
    Select top-k key blocks for each query block based on compressed attention scores with GQA support.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k_cmp: Compressed keys [batch, num_blocks, num_kv_heads, head_dim]
        lse: Log-sum-exp from compressed attention [batch, seq_len, num_q_heads]
        block_counts: Number of blocks to select (k)
        block_size: Size of each block
        scale: Attention scaling factor

    Returns:
        Block indices [batch, num_kv_heads, num_query_blocks, block_counts]
    """
    batch, seq_len, num_q_heads, head_dim = query.shape
    num_kv_heads = k_cmp.shape[2]
    group_size = num_q_heads // num_kv_heads
    num_key_blocks = k_cmp.shape[1]
    num_query_blocks = (seq_len + block_size - 1) // block_size

    pad_len = num_query_blocks * block_size - seq_len
    if pad_len > 0:
        query_padded = jnp.pad(query, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        lse = jnp.pad(lse, ((0, 0), (0, pad_len), (0, 0)), constant_values=-jnp.inf)
    else:
        query_padded = query

    q_blocks = query_padded.reshape(batch, num_query_blocks, block_size, num_q_heads, head_dim)
    q_blocks_avg = jnp.mean(q_blocks, axis=2)

    q_blocks_grouped = q_blocks_avg.reshape(batch, num_query_blocks, num_kv_heads, group_size, head_dim)

    q_blocks_kv = jnp.mean(q_blocks_grouped, axis=3)

    scores = jnp.einsum("bqnd,bmnd->bqnm", q_blocks_kv, k_cmp) * scale

    query_block_mask = (
        jnp.arange(num_key_blocks)[None, None, None, :] <= jnp.arange(num_query_blocks)[None, :, None, None]
    )
    scores = jnp.where(query_block_mask, scores, -jnp.inf)

    scores = scores.transpose(0, 2, 1, 3)

    _, top_indices = jax.lax.top_k(scores, min(block_counts, num_key_blocks))

    if block_counts > num_key_blocks:
        pad_size = block_counts - num_key_blocks
        top_indices = jnp.pad(top_indices, ((0, 0), (0, 0), (0, 0), (0, pad_size)), constant_values=0)

    return top_indices.astype(jnp.int32)


@kernel_registry.register("apply_native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def apply_native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_key_blocks"],
    block_counts: Int[Array, "batch num_kv_heads num_query_blocks"] | int = 16,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
    token_indices: Int[Array, "total_tokens"] | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    Applies block-sparse attention using a pre-computed sparsity pattern with JAX/XLA.

    This function implements sparse attention where each query block attends to a
    subset of key blocks specified by the sparsity pattern. This reduces computational
    complexity from O(N²) to O(N·S) where S is the sparsity (number of blocks attended).

    Args:
        query: Query tensor of shape `(batch, seq_len, num_heads, head_dim)`.
        key: Key tensor of shape `(batch, seq_len, num_heads, head_dim)`.
        value: Value tensor of shape `(batch, seq_len, num_heads, head_dim)`.
        block_indices: A tensor of shape `(batch, num_heads, num_query_blocks, num_key_blocks)`
            specifying which key blocks each query block should attend to. Each entry
            contains the index of a key block.
        block_counts: Number of key blocks each query block attends to. Can be:
            - int: uniform sparsity for all query blocks
            - tensor [batch, num_heads, num_query_blocks]: per-block sparsity
        block_size: Size of each block (both query and key blocks).
        scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).

    Returns:
        Attention output of shape `(batch, seq_len, num_heads, head_dim)`.

    Notes:
        - The sequence is divided into blocks of size `block_size`
        - Each query block computes attention over selected key blocks only
        - Sparsity is determined by `block_indices` and `block_counts`
        - Useful for long-range attention with reduced computation

    Examples:
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> block_size = 64
        >>> num_blocks = seq_len // block_size
        >>>
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> block_counts = 4
        >>> block_indices = jnp.tile(
        ...     jnp.arange(4)[None, None, None, :],
        ...     (batch, num_heads, num_blocks, 1)
        ... )
        >>>
        >>> output = apply_native_sparse_attention(
        ...     query, key, value, block_indices, block_counts, block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)

        >>>
        >>> def create_local_pattern(num_blocks, window=2):
        ...     indices = []
        ...     for i in range(num_blocks):
        ...         local = list(range(max(0, i-window), min(num_blocks, i+window+1)))
        ...
        ...         local = local + [0] * (window*2+1 - len(local))
        ...         indices.append(local)
        ...     return jnp.array(indices)
        >>>
        >>> local_indices = create_local_pattern(num_blocks, window=2)
        >>> local_indices = jnp.tile(local_indices[None, None, :, :], (batch, num_heads, 1, 1))
        >>> output = apply_native_sparse_attention(
        ...     query, key, value, local_indices, block_counts=5, block_size=block_size
        ... )
    """
    if cu_seqlens is not None:
        raise NotImplementedError("cu_seqlens is not supported in XLA apply_native_sparse_attention implementation")
    if token_indices is not None:
        raise NotImplementedError("token_indices is not supported in XLA apply_native_sparse_attention implementation")

    if scale is None:
        scale = 1.0 / jnp.sqrt(query.shape[-1])

    if isinstance(block_counts, int):
        batch = query.shape[0]
        num_kv_heads = key.shape[2]
        num_query_blocks = block_indices.shape[2]
        block_counts = jnp.full((batch, num_kv_heads, num_query_blocks), block_counts, dtype=jnp.int32)

    return _sparse_attention_with_vjp(query, key, value, block_indices, block_counts, block_size, scale)


@kernel_registry.register("native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len hidden_dim"] | None = None,
    g_slc: Float[Array, "batch seq_len hidden_dim"] | None = None,
    block_indices: Int[Array, "batch num_kv_heads num_query_blocks num_keys_blocks"] | None = None,
    block_counts: Int[Array, "batch num_kv_heads num_query_blocks"] | int = 16,
    block_size: int = 64,
    scale: float | None = None,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    """
    Native Sparse Attention (NSA) with XLA/JAX implementation.

    NSA is a sparse attention mechanism that combines two components:
    1.  **Compressed Attention**: A coarse-grained attention over mean-pooled
        (compressed) key-value blocks. This provides a global context summary.
    2.  **Selected Attention**: A fine-grained, sparse attention where each
        query attends to a small subset of the original key-value blocks.

    The key idea is that the selection of blocks for the second component can be
    determined efficiently using the compressed representations from the first.
    The final output is a gated combination of these two components.

    Args:
        query: Query tensor of shape `(batch_size, sequence, num_heads, head_dim)`.
        key: Key tensor of shape `(batch_size, sequence, num_heads, head_dim)`.
        value: Value tensor of shape `(batch_size, sequence, num_heads, head_dim)`.
        g_cmp: Optional gate tensor for compressed attention, shape `(batch_size, sequence, hidden_dim)`.
            If provided, the compressed attention component is computed.
        g_slc: Optional gate tensor for selected attention, shape `(batch_size, sequence, hidden_dim)`.
        block_indices: Optional tensor of pre-computed block indices for selected
            attention, shape `(batch_size, num_heads, num_query_blocks, block_counts)`.
            If `g_cmp` is provided, this argument is ignored, and block indices are
            computed dynamically via top-k selection over the compressed keys.
            If `g_cmp` is NOT provided, this argument is required.
        block_counts: Number of blocks to select for each query. Can be:
            - int: uniform sparsity for all query blocks
            - tensor [batch, num_heads, num_query_blocks]: per-block sparsity
            Defaults to 16.
        block_size: The size of each attention block. Defaults to 64.
        scale: Scale factor for attention scores. Defaults to `1 / sqrt(head_dim)`.
        cu_seqlens: Cumulative sequence lengths of shape `(N+1)` for
            variable-length training. If provided, batch size must be 1.
            Note: Variable-length sequences are not yet fully supported in XLA version.

    Returns:
        The output tensor of shape `(batch_size, sequence, num_heads, head_dim)`.

    Notes:
        - The XLA implementation uses pure JAX operations without custom kernels
        - For variable-length sequences (cu_seqlens), this uses the mean_pooling function
        - The compressed attention component uses mean-pooled key/value blocks
        - Top-k block selection is based on attention scores from compressed keys

    Examples:
        >>> batch, seq_len, num_heads, head_dim = 2, 1024, 8, 64
        >>> block_size = 64
        >>> block_counts = 16
        >>>
        >>> q = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> k = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>> v = jnp.ones((batch, seq_len, num_heads, head_dim))
        >>>
        >>>
        >>> g_cmp = jnp.ones((batch, seq_len, num_heads * head_dim))
        >>> output = native_sparse_attention(
        ...     query, key, value, g_cmp=g_cmp, block_counts=block_counts, block_size=block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)
        >>>
        >>>
        >>> num_blocks = seq_len // block_size
        >>> block_indices = jnp.tile(
        ...     jnp.arange(block_counts)[None, None, None, :],
        ...     (batch, num_heads, num_blocks, 1)
        ... )
        >>> output = native_sparse_attention(
        ...     query, key, value, block_indices=block_indices, block_counts=block_counts, block_size=block_size
        ... )
        >>> output.shape
        (2, 1024, 8, 64)
    """
    if scale is None:
        scale = 1.0 / jnp.sqrt(query.shape[-1])

    if cu_seqlens is not None:
        batch_size = query.shape[0]
        if batch_size != 1:
            warnings.warn(
                "cu_seqlens with batch_size != 1 may not work correctly in XLA implementation. "
                "Consider using batch_size=1 for variable-length sequences.",
                stacklevel=2,
            )

    batch, seq_len, _num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[2]
    num_blocks = (seq_len + block_size - 1) // block_size

    pad_len = num_blocks * block_size - seq_len
    if pad_len > 0:
        k_padded = jnp.pad(key, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v_padded = jnp.pad(value, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
    else:
        k_padded = key
        v_padded = value

    k_cmp = k_padded.reshape(batch, num_blocks, block_size, num_kv_heads, head_dim).mean(axis=2)
    v_cmp = v_padded.reshape(batch, num_blocks, block_size, num_kv_heads, head_dim).mean(axis=2)
    o_cmp = None

    if g_cmp is not None:
        o_cmp, lse_cmp = _nsa_compression_xla(
            query=query,
            k_cmp=k_cmp,
            v_cmp=v_cmp,
            block_size=block_size,
            scale=scale,
        )
        if block_indices is not None:
            warnings.warn(
                "`block_indices` will be ignored when `g_cmp` is provided",
                stacklevel=2,
            )

        block_indices = _nsa_topk_xla(
            query=query,
            k_cmp=k_cmp,
            lse=lse_cmp,
            block_counts=block_counts if isinstance(block_counts, int) else block_counts[0, 0, 0].item(),
            block_size=block_size,
            scale=scale,
        )

    if block_indices is None:
        raise ValueError("Either `g_cmp` must be provided or `block_indices` must be passed.")

    if isinstance(block_counts, int):
        num_query_blocks = block_indices.shape[2]
        block_counts = jnp.full((batch, num_kv_heads, num_query_blocks), block_counts, dtype=jnp.int32)

    o_slc = apply_native_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        scale=scale,
    )

    o = o_slc
    if g_slc is not None:
        o = o_slc * jnp.expand_dims(g_slc, -1)

    if o_cmp is not None and g_cmp is not None:
        o = o + o_cmp * jnp.expand_dims(g_cmp, -1)

    return o
