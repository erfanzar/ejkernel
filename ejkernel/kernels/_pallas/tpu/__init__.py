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


"""Pallas TPU kernel implementations.

This module provides TPU-optimized kernels using Pallas with Mosaic backend.
Kernels are designed to leverage TPU's Matrix Multiply Units (MXUs) and
High Bandwidth Memory (HBM) for efficient execution.

Available Kernels:
    - flash_attention: Memory-efficient exact attention
    - flash_mla: Multi-head latent attention forward pass
    - deepseek_attn: DeepSeek Sparse Attention (MLA + Lightning Indexer)
    - gated_delta_rule: Gated delta rule linear attention
    - multi_latent_ragged_page_attention: MLA ragged paged attention
    - multi_latent_ragged_page_attention_v2: MLA ragged paged attention v2
    - blocksparse_attention: Block-sparse attention patterns
    - ring_attention: Distributed attention across devices
    - page_attention: Paged KV cache attention
    - quantized_matmul: Packed-weights quantized matrix multiplication
    - ragged_page_attention_v2/v3: Variable-length paged attention
    - ragged_decode_attention: Decode-phase attention
    - grouped_matmul/v2: Grouped matrix multiplication
"""

from .all_gather_matmul import all_gather_matmul
from .blocksparse_attention import blocksparse_attention as blocksparse_attention
from .deepseek_attn import deepseek_attn
from .flash_attention import flash_attention
from .flash_mla import flash_mla
from .gated_delta_rule import gated_delta_rule
from .grouped_matmul import grouped_matmul
from .grouped_matmulv2 import grouped_matmulv2
from .grouped_matmulv3 import grouped_matmulv3
from .multi_latent_ragged_page_attention import multi_latent_ragged_page_attention
from .multi_latent_ragged_page_attention_v2 import multi_latent_ragged_page_attention_v2
from .page_attention import page_attention
from .prefill_page_attention import prefill_page_attention
from .quantized_matmul import quantized_matmul
from .ragged_decode_attention import ragged_decode_attention
from .ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gated_delta_rule
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .reduce_scatter_matmul import reduce_scatter_matmul
from .ring_attention import ring_attention

__all__ = (
    "all_gather_matmul",
    "blocksparse_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "gated_delta_rule",
    "grouped_matmul",
    "grouped_matmulv2",
    "grouped_matmulv3",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "reduce_scatter_matmul",
    "ring_attention",
)
