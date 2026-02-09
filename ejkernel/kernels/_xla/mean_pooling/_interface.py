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
"""Kernel public interface and registration wrappers."""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, Int
from ._xla_impl_fwd import mean_pooling as _mean_pooling_impl


@kernel_registry.register("mean_pooling", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def mean_pooling(
    x: Float[Array, "... hidden_dim"], chunk_size: int = 32, cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None
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
    return _mean_pooling_impl(x, chunk_size, cu_seqlens)


__all__ = ("mean_pooling",)
