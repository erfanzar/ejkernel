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


"""EJKernel Triton Kernels Module.

This module provides optimized Triton kernel implementations for various
attention mechanisms and neural network operations. All kernels are designed
for efficient GPU execution using JAX and Triton.

Available kernels:
- flash_attention: Memory-efficient exact attention with O(N) memory complexity
- recurrent: Linear-time recurrent attention mechanisms
- recurrent_gla: Gated Linear Attention implementation
- lightning_attn: Lightning Attention with layer-adaptive decay
- mean_pooling: Efficient mean pooling over sequence dimension
- native_sparse_attention: Sparse attention with dynamic block selection

All implementations support:
- Variable-length sequences via cumulative sequence lengths
- Custom backward passes for gradient computation
- Automatic kernel tuning for optimal performance
"""

from .blocksparse_attention import blocksparse_attention
from .flash_attention import flash_attention
from .gla import recurrent_gla
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .native_sparse_attention import apply_native_sparse_attention, native_sparse_attention
from .page_attention import page_attention
from .ragged_decode_attention import ragged_decode_attention
from .ragged_page_attention import ragged_page_attention
from .recurrent import recurrent

__all__ = (
    "apply_native_sparse_attention",
    "blocksparse_attention",
    "flash_attention",
    "lightning_attn",
    "mean_pooling",
    "native_sparse_attention",
    "page_attention",
    "ragged_decode_attention",
    "ragged_page_attention",
    "recurrent",
    "recurrent_gla",
)
