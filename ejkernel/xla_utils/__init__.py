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


from .cumsum import chunk_global_cumsum, chunk_local_cumsum
from .utils import (
    cdiv,
    identity_dtype_convert,
    mask_to_segment_ids,
    prepare_chunk_indices,
    prepare_chunk_offsets,
    prepare_cu_seqlens_from_mask,
    prepare_lens,
    prepare_lens_from_mask,
    prepare_position_ids,
    prepare_sequence_ids,
    prepare_token_indices,
    segment_ids_to_mask,
    segment_ids_to_qkv_masks,
)

__all__ = [
    "cdiv",
    "chunk_global_cumsum",
    "chunk_local_cumsum",
    "identity_dtype_convert",
    "mask_to_segment_ids",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens",
    "prepare_lens_from_mask",
    "prepare_position_ids",
    "prepare_sequence_ids",
    "prepare_token_indices",
    "segment_ids_to_mask",
    "segment_ids_to_qkv_masks",
]
