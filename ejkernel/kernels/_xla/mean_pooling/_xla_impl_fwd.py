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

"""Mean pooling interface for sequence embedding aggregation.

This module provides the public API for mean pooling operations over
sequence dimensions. Supports both fixed-length and variable-length
(packed) sequences with custom VJP for efficient gradient computation.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from . import _xla_impl_bwd as _mean_pooling_bwd_mod
from ._xla_impl_bwd import _mean_pooling_bwd


def _mean_pooling_varlen(
    x: Float[Array, "total_tokens hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
) -> Float[Array, "num_seqs hidden_dim"]:
    """Mean pooling for variable-length (packed) sequences.

    Computes the mean of token embeddings for each variable-length sequence
    in a packed tensor. Sequences are identified by cumulative sequence
    lengths, avoiding the need for padding.

    Args:
        x: Input tensor of shape [total_tokens, hidden_dim] containing
            concatenated token embeddings for all sequences.
        cu_seqlens: Cumulative sequence lengths of shape [num_seqs + 1].
            For example, [0, 10, 25] indicates two sequences: tokens 0-9
            and tokens 10-24.

    Returns:
        Mean-pooled tensor of shape [num_seqs, hidden_dim] where each row
        is the average of the token embeddings for the corresponding sequence.
    """
    num_seqs = len(cu_seqlens) - 1
    max_seq_len = jnp.max(cu_seqlens[1:] - cu_seqlens[:-1])

    def pool_sequence(i):
        """Compute mean-pooled embedding for a single variable-length sequence.

        Args:
            i: Sequence index within the packed tensor.

        Returns:
            Mean embedding vector of shape [hidden_dim] for this sequence.
        """
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        seq_len = end - start

        seq_tokens = jax.lax.dynamic_slice(x, (start, 0), (max_seq_len, x.shape[-1]))

        mask = jnp.arange(max_seq_len) < seq_len

        masked_tokens = jnp.where(mask[:, None], seq_tokens, 0)
        return jnp.sum(masked_tokens, axis=0) / seq_len

    return jax.vmap(pool_sequence)(jnp.arange(num_seqs))


def _mean_pooling_fixed(
    x: Float[Array, "batch seq_len hidden_dim"],
) -> Float[Array, "batch hidden_dim"]:
    """Mean pooling for fixed-length (padded) sequences.

    Computes the mean of token embeddings along the sequence dimension
    for a standard batched tensor where all sequences have the same length.

    Args:
        x: Input tensor of shape [batch, seq_len, hidden_dim] where all
            sequences are padded to the same length.

    Returns:
        Mean-pooled tensor of shape [batch, hidden_dim] where each row
        is the average across all seq_len tokens for that batch element.
    """
    return jnp.mean(x, axis=1)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _mean_pooling_core(
    x: Float[Array, "... hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Core mean pooling implementation with custom VJP.

    Dispatches to either variable-length or fixed-length mean pooling
    based on whether cumulative sequence lengths are provided. This
    function is wrapped with ``jax.custom_vjp`` to enable efficient
    gradient computation that avoids recomputing the forward pass.

    Args:
        x: Input tensor. Either [total_tokens, hidden_dim] for packed
            sequences or [batch, seq_len, hidden_dim] for fixed-length.
        cu_seqlens: Optional cumulative sequence lengths [num_seqs + 1].
            If provided, uses variable-length pooling.

    Returns:
        Mean-pooled tensor of shape [batch_or_num_seqs, hidden_dim].
    """
    if cu_seqlens is not None:
        return _mean_pooling_varlen(x, cu_seqlens)
    else:
        return _mean_pooling_fixed(x)


_mean_pooling_bwd_mod._mean_pooling_core = _mean_pooling_core
_mean_pooling_core.defvjp(_mean_pooling_bwd_mod._mean_pooling_fwd, _mean_pooling_bwd)


def mean_pooling(
    x: Float[Array, "... hidden_dim"],
    chunk_size: int = 32,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """
    Performs mean pooling over the sequence dimension using JAX/XLA.

    This function calculates the mean of token embeddings for each sequence in a
    batch. It supports both standard (padded) and variable-length sequences.

    Args:
        x: The input tensor of shape `(batch_size, sequence_length, hidden_dim)`.
            If `cu_seqlens` is provided for variable-length inputs, the shape
            should be `(total_tokens, hidden_dim)`.
        chunk_size: Performance tuning parameter (ignored in XLA, only used by Triton).
        cu_seqlens: An optional 1D tensor of cumulative sequence lengths for
            handling variable-length sequences in a packed format.
            Example: `[0, len_seq1, len_seq1+len_seq2, ...]`. If provided, the
            function will compute the mean pooling for each of the packed
            sequences.

    Returns:
        A tensor of shape `(batch_size, hidden_dim)` containing the mean-pooled
        embeddings for each sequence. If `cu_seqlens` is used, the batch size in
        the output shape will correspond to the number of sequences defined by
        `cu_seqlens` (i.e., `len(cu_seqlens) - 1`).

    Examples:
        >>>
        >>> x = jnp.ones((2, 10, 128))
        >>> out = mean_pooling(x)
        >>> out.shape
        (2, 128)

        >>>
        >>> x = jnp.ones((25, 128))
        >>> cu_seqlens = jnp.array([0, 10, 25])
        >>> out = mean_pooling(x, cu_seqlens=cu_seqlens)
        >>> out.shape
        (2, 128)
    """

    return _mean_pooling_core(x, cu_seqlens)
