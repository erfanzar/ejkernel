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


"""modified from google org impl"""

from ._info import MaskInfo
from ._kernel import (
    BlockSizes,
    blocksparse_attention,
    make_attention_reference,
    make_masked_mha_reference,
    make_masked_mqa_reference,
    make_splash_mha,
    make_splash_mha_single_device,
    make_splash_mqa,
    make_splash_mqa_single_device,
)
from ._masks import (
    CausalMask,
    ChunkedCausalMask,
    FullMask,
    LocalMask,
    Mask,
    MultiHeadMask,
    NumpyMask,
    make_causal_mask,
    make_chunk_attention_mask,
    make_local_attention_mask,
    make_random_mask,
)

__all__ = (
    "BlockSizes",
    "CausalMask",
    "ChunkedCausalMask",
    "FullMask",
    "LocalMask",
    "Mask",
    "MaskInfo",
    "MultiHeadMask",
    "NumpyMask",
    "blocksparse_attention",
    "make_attention_reference",
    "make_causal_mask",
    "make_chunk_attention_mask",
    "make_local_attention_mask",
    "make_masked_mha_reference",
    "make_masked_mqa_reference",
    "make_random_mask",
    "make_splash_mha",
    "make_splash_mha_single_device",
    "make_splash_mqa",
    "make_splash_mqa_single_device",
)
