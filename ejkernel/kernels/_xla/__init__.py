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


"""XLA-based kernel implementations for attention and related operations.

This module provides pure XLA/JAX implementations of attention mechanisms
and other kernels. XLA implementations are hardware-agnostic and work
across GPU, TPU, and CPU backends through JAX's XLA compiler.

XLA kernels provide:
    - Cross-platform compatibility (GPU/TPU/CPU)
    - Automatic gradient computation
    - XLA compilation optimizations
    - No platform-specific dependencies

Available Operations:
    Attention Mechanisms:
        - attention: Standard multi-head attention
        - flash_attention: Memory-efficient attention with tiling
        - blocksparse_attention: Block-sparse attention patterns
        - native_sparse_attention: Sparse attention with flexible patterns
        - ring_attention: Distributed ring attention for long sequences
        - scaled_dot_product_attention: Basic scaled dot-product attention
        - unified_attention: Unified API for multiple attention variants

    Page/Serving Attention:
        - page_attention: Paged KV-cache attention
        - ragged_page_attention_v2/v3: Variable-length page attention
        - ragged_decode_attention: Decode-phase attention

    Linear/Recurrent Attention:
        - recurrent_gla: Gated linear attention (recurrent form)
        - lightning_attn: Lightning attention with decay
        - recurrent: General recurrent attention mechanism
        - kernel_delta_attention: Delta-rule linear attention

    State Space Models:
        - state_space_v1: Mamba1-style SSM
        - state_space_v2: Mamba2-style SSM

    Utilities:
        - grouped_matmul: Efficient grouped matrix multiplication
        - mean_pooling: Sequence mean pooling

Note:
    XLA implementations are the fallback when platform-specific kernels
    (Triton, Pallas, CUDA) are not available for the current hardware.
"""

from .attention import attention
from .blocksparse_attention import blocksparse_attention
from .flash_attention import flash_attention
from .gla import recurrent_gla
from .grouped_matmul import grouped_matmul
from .kernel_delta_attention import kda, kda_decay, kernel_delta_attention
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .native_sparse_attention import apply_native_sparse_attention
from .page_attention import page_attention
from .ragged_decode_attention import ragged_decode_attention
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .recurrent import recurrent
from .ring_attention import ring_attention
from .scaled_dot_product_attention import scaled_dot_product_attention
from .state_space_v1 import state_space_v1
from .state_space_v2 import state_space_v2
from .unified_attention import unified_attention

__all__ = [
    "apply_native_sparse_attention",
    "attention",
    "blocksparse_attention",
    "flash_attention",
    "grouped_matmul",
    "kda",
    "kda_decay",
    "kernel_delta_attention",
    "lightning_attn",
    "mean_pooling",
    "page_attention",
    "ragged_decode_attention",
    "ragged_page_attention_v2",
    "ragged_page_attention_v3",
    "recurrent",
    "recurrent_gla",
    "ring_attention",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
]
