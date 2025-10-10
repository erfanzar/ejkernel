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
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _sparse_attention_bwd
from ._xla_impl_fwd import _sparse_attention_fwd


@partial(jax.custom_vjp, nondiff_argnums=(5, 6))
def _sparse_attention_with_vjp(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
) -> Float[Array, "batch seq_len num_q_heads head_dim"]:
    return _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, softmax_scale)


def _sparse_attention_fwd_vjp(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"],
    block_size: int,
    softmax_scale: float,
):
    output = _sparse_attention_fwd(query, key, value, block_indices, block_counts, block_size, softmax_scale)
    residuals = (query, key, value, block_indices, block_counts, block_size, softmax_scale)
    return output, residuals


def _sparse_attention_bwd_vjp(
    block_size: int,
    softmax_scale: float,
    residuals: tuple,
    do: Float[Array, "batch seq_len num_q_heads head_dim"],
):
    query, key, value, block_indices, block_counts, block_size_, softmax_scale_ = residuals
    dq, dk, dv = _sparse_attention_bwd(query, key, value, block_indices, block_counts, block_size_, softmax_scale_, do)

    return (dq, dk, dv, None, None)


_sparse_attention_with_vjp.defvjp(_sparse_attention_fwd_vjp, _sparse_attention_bwd_vjp)


def _nsa_compression_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    v_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    block_size: int,
    softmax_scale: float,
) -> tuple[Float[Array, "batch seq_len num_q_heads head_dim"], Float[Array, "batch seq_len num_q_heads"]]:
    """
    Compute compressed attention over mean-pooled key/value blocks with GQA support.

    Args:
        query: Query tensor [batch, seq_len, num_q_heads, head_dim]
        k_cmp: Compressed (mean-pooled) keys [batch, num_blocks, num_kv_heads, head_dim]
        v_cmp: Compressed (mean-pooled) values [batch, num_blocks, num_kv_heads, head_dim]
        block_size: Size of each block
        softmax_scale: Attention scaling factor

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

    scores = jnp.einsum("bsnd,bmnd->bsnm", query, k_cmp_expanded) * softmax_scale

    t_ids = jnp.arange(seq_len, dtype=jnp.int32)
    s_completed = (t_ids + 1) // block_size
    c_ids = jnp.arange(num_blocks, dtype=jnp.int32)

    block_mask = c_ids[None, :] < s_completed[:, None]
    block_mask = block_mask[None, :, None, :]

    scores_masked = jnp.where(block_mask, scores, -jnp.inf)

    lse_raw = logsumexp(scores_masked, axis=-1)
    lse = jnp.where(s_completed[None, :, None] > 0, lse_raw, 0.0)

    p = jnp.exp(scores - lse[..., None])
    p = jnp.where(block_mask, p, 0.0)

    p_sum = p.sum(axis=-1)
    o_num = jnp.einsum("bsnm,bmnd->bsnd", p, v_cmp_expanded)
    output = jnp.where(
        (p_sum > 0)[..., None],
        o_num / jnp.maximum(p_sum, 1e-9)[..., None],
        jnp.zeros_like(o_num),
    )

    return output, lse


def _nsa_topk_xla(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    k_cmp: Float[Array, "batch num_blocks num_kv_heads head_dim"],
    lse: Float[Array, "batch seq_len num_q_heads"],
    block_counts: int,
    block_size: int,
    softmax_scale: float,
) -> Int[Array, "batch seq_len num_kv_heads num_selected_blocks"]:
    """
    Per-token top-k selection matching Triton:
    p = exp(score - lse), force-include current block (p=1.0), sum over groups G, top-k across blocks.
    """
    B, T, HQ, _D = query.shape
    C = k_cmp.shape[1]
    H = k_cmp.shape[2]
    G = HQ // H

    k_cmp_exp = jnp.repeat(k_cmp, G, axis=2)
    k_cmp_exp = jnp.swapaxes(k_cmp_exp, 1, 2)

    scores = jnp.einsum("bthd,bhcd->bthc", query, k_cmp_exp) * softmax_scale

    qb = jnp.arange(T, dtype=jnp.int32) // block_size
    c_ids = jnp.arange(C, dtype=jnp.int32)
    mask = c_ids[None, None, None, :] <= qb[None, :, None, None]
    scores = jnp.where(mask, scores, -jnp.inf)

    p = jnp.exp(scores - lse[..., None])

    one_hot_qb = jax.nn.one_hot(qb, C, dtype=p.dtype)
    p = jnp.where(one_hot_qb[None, :, None, :] > 0, 1.0, p)

    p_sum = p.reshape(B, T, H, G, C).sum(axis=3)
    future_mask = jnp.arange(C, dtype=jnp.int32)[None, None, None, :] > qb[None, :, None, None]
    p_sum = jnp.where(future_mask, -jnp.inf, p_sum)

    tie = (C - 1.0 - jnp.arange(C, dtype=p_sum.dtype)[None, None, None, :]) * jnp.array(1e-8, dtype=p_sum.dtype)
    p_sum_adj = p_sum + tie
    S = block_counts if isinstance(block_counts, int) else int(block_counts)
    p_flat = p_sum_adj.reshape(B * T * H, C)
    _, idx_flat = jax.lax.top_k(p_flat, S)
    idx = idx_flat.reshape(B, T, H, S).astype(jnp.int32)
    idx = jnp.minimum(idx, qb[None, :, None, None])
    return idx


@kernel_registry.register("apply_native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def apply_native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"],
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
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
        softmax_scale: Attention scaling factor. If None, defaults to 1/sqrt(head_dim).

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

    if softmax_scale is None:
        softmax_scale = float(1.0 / jnp.sqrt(query.shape[-1]))
    else:
        softmax_scale = float(softmax_scale)

    if isinstance(block_counts, int):
        batch = query.shape[0]
        seq_len = query.shape[1]
        num_kv_heads = key.shape[2]
        block_counts = jnp.full((batch, seq_len, num_kv_heads), block_counts, dtype=jnp.int32)

    return _sparse_attention_with_vjp(query, key, value, block_indices, block_counts, block_size, softmax_scale)


@kernel_registry.register("native_sparse_attention", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def native_sparse_attention(
    query: Float[Array, "batch seq_len num_q_heads head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads head_dim"],
    g_cmp: Float[Array, "batch seq_len num_q_heads"] | None = None,
    g_slc: Float[Array, "batch seq_len num_q_heads"] | None = None,
    block_indices: Int[Array, "batch seq_len num_kv_heads num_selected_blocks"] | None = None,
    block_counts: Int[Array, "batch seq_len num_kv_heads"] | int = 16,
    block_size: int = 64,
    softmax_scale: float | None = None,
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
        softmax_scale: Scale factor for attention scores. Defaults to `1 / sqrt(head_dim)`.
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
    if softmax_scale is None:
        softmax_scale = float(1.0 / jnp.sqrt(query.shape[-1]))
    else:
        softmax_scale = float(softmax_scale)

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
            softmax_scale=softmax_scale,
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
            block_counts=block_counts if isinstance(block_counts, int) else int(block_counts[0, 0, 0]),
            block_size=block_size,
            softmax_scale=softmax_scale,
        )
    if block_indices is None:
        raise ValueError("Either `g_cmp` must be provided or `block_indices` must be passed.")

    o_slc = apply_native_sparse_attention(
        query=query,
        key=key,
        value=value,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        softmax_scale=softmax_scale,
    )

    o = o_slc
    if g_slc is not None:
        o = o_slc * jnp.expand_dims(g_slc, -1)

    if o_cmp is not None and g_cmp is not None:
        o = o + o_cmp * jnp.expand_dims(g_cmp, -1)

    return o
