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


"""XLA-specific utilities for sequence processing and sharding.

This module provides utilities for working with packed sequences, cumulative
operations, and sharding specifications in JAX/XLA computations.

Key Features:
    - Packed sequence utilities for efficient batch processing
    - Chunked cumulative sum operations for attention mechanisms
    - Sharding utilities for distributed computation
    - Sequence reordering for ring attention patterns

Sequence Utilities:
    - prepare_lens: Calculate sequence lengths from cumulative lengths
    - prepare_position_ids: Generate position IDs for packed sequences
    - prepare_sequence_ids: Generate sequence IDs for packed sequences
    - prepare_token_indices: Generate (seq_id, pos_id) pairs
    - prepare_chunk_indices: Generate chunk indices for tiled processing
    - prepare_cu_seqlens_from_mask: Create cumulative lengths from masks

Cumulative Sum Operations:
    - chunk_local_cumsum: Chunked local cumulative sum for attention
    - chunk_global_cumsum: Global cumulative sum with sequence boundaries

Sharding Utilities:
    - get_corrected_named_sharding: Create valid shardings for array shapes
    - reorder_sequence: Reorder sequences for ring attention patterns

Example:
    >>> from ejkernel.xla_utils import prepare_position_ids
    >>> import jax.numpy as jnp
    >>>
    >>> cu_seqlens = jnp.array([0, 3, 5, 9])  # 3 sequences of lengths [3, 2, 4]
    >>> position_ids = prepare_position_ids(cu_seqlens)
    >>> # Returns: [0, 1, 2, 0, 1, 0, 1, 2, 3]
"""

from .cumsum import chunk_global_cumsum, chunk_local_cumsum
from .shardings import get_corrected_named_sharding, reorder_sequence
from .utils import (
    cdiv,
    identity_dtype_convert,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
)

__all__ = [
    "cdiv",
    "chunk_global_cumsum",
    "chunk_local_cumsum",
    "get_corrected_named_sharding",
    "identity_dtype_convert",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens",
    "prepare_lens_from_mask",
    "prepare_position_ids",
    "prepare_sequence_ids",
    "prepare_token_indices",
    "reorder_sequence",
]
