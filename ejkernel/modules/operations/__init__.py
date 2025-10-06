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


"""Attention kernel modules with automatic optimization."""

from .attention import Attention, attention
from .flash import FlashAttention, flash_attention
from .gla import GLAttention, gla_attention
from .lightning import LightningAttention, lightning_attention
from .matmul import GroupedMatmul, grouped_matmul
from .mla import FlashMLA, mla_attention
from .native_sparse_attention import NativeSparseAttention, sparse_attention
from .page_attention import PageAttention, page_attention
from .pooling import MeanPooling, mean_pooling
from .ragged_page_attention import RaggedPageAttention, ragged_page_attention
from .recurrent import RecurrentAttention, recurrent_attention
from .ring import RingAttention, ring_attention
from .scaled_dot_product_attention import ScaledDotProductAttention, scaled_dot_product_attention

__all__ = (
    "Attention",
    "FlashAttention",
    "FlashMLA",
    "GLAttention",
    "GroupedMatmul",
    "LightningAttention",
    "MeanPooling",
    "NativeSparseAttention",
    "PageAttention",
    "RaggedPageAttention",
    "RecurrentAttention",
    "RingAttention",
    "ScaledDotProductAttention",
    "attention",
    "flash_attention",
    "gla_attention",
    "grouped_matmul",
    "lightning_attention",
    "mean_pooling",
    "mla_attention",
    "page_attention",
    "ragged_page_attention",
    "recurrent_attention",
    "ring_attention",
    "scaled_dot_product_attention",
    "sparse_attention",
)
