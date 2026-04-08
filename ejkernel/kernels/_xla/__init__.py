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
        - flash_mla: Multi-head Latent Attention (MLA)
        - deepseek_attn: DeepSeek Sparse Attention (MLA + Lightning Indexer)
        - blocksparse_attention: Block-sparse attention patterns
        - native_sparse_attention: Sparse attention with flexible patterns
        - ring_attention: Distributed ring attention for long sequences
        - scaled_dot_product_attention: Basic scaled dot-product attention
        - unified_attention: Unified API for multiple attention variants

    Page/Serving Attention:
        - page_attention: Paged KV-cache attention
        - multi_latent_ragged_page_attention: MLA ragged paged attention
        - multi_latent_ragged_page_attention_v2: MLA ragged paged attention v2
        - ragged_page_attention_v2/v3: Variable-length page attention
        - ragged_decode_attention: Decode-phase attention

    Linear/Recurrent Attention:
        - recurrent_gla: Gated linear attention (recurrent form)
        - gated_delta_rule: Gated delta rule linear attention
        - lightning_attn: Lightning attention with decay
        - recurrent: General recurrent attention mechanism
        - kernel_delta_attention: Delta-rule linear attention

    State Space Models:
        - state_space_v1: Mamba1-style SSM
        - state_space_v2: Mamba2-style SSM

    Utilities:
        - grouped_matmul: Efficient grouped matrix multiplication
        - mean_pooling: Sequence mean pooling
        - quantized_matmul: Packed uint32 quantized matmul

Note:
    XLA implementations are the fallback when platform-specific kernels
    (Triton, Pallas, CUDA) are not available for the current hardware.
"""

from .all_gather_matmul import all_gather_matmul
from .attention import attention
from .blocksparse_attention import blocksparse_attention
from .chunked_prefill_paged_decode import chunked_prefill_paged_decode
from .decode_attention import decode_attention
from .deepseek_attn import deepseek_attn
from .flash_attention import flash_attention
from .flash_mla import flash_mla
from .gated_delta_rule import gated_delta_rule
from .gla import recurrent_gla
from .grouped_matmul import grouped_matmul
from .grouped_matmulv3 import grouped_matmulv3
from .kernel_delta_attention import kda, kda_decay, kernel_delta_attention
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .multi_latent_ragged_page_attention import multi_latent_ragged_page_attention
from .multi_latent_ragged_page_attention_v2 import multi_latent_ragged_page_attention_v2
from .native_sparse_attention import apply_native_sparse_attention
from .page_attention import page_attention
from .prefill_page_attention import prefill_page_attention
from .quantized_matmul import quantized_matmul
from .ragged_decode_attention import ragged_decode_attention
from .ragged_gated_delta_rule import ragged_gated_delta_rule as ragged_gated_delta_rule
from .ragged_page_attention_v2 import ragged_page_attention_v2
from .ragged_page_attention_v2_turboquant import ragged_page_attention_v2_turboquant
from .ragged_page_attention_v3 import ragged_page_attention_v3
from .ragged_page_attention_v3_turboquant import ragged_page_attention_v3_turboquant
from .recurrent import recurrent
from .reduce_scatter_matmul import reduce_scatter_matmul
from .ring_attention import ring_attention
from .rwkv4 import rwkv4
from .rwkv6 import rwkv6
from .rwkv7 import rwkv7, rwkv7_mul
from .scaled_dot_product_attention import scaled_dot_product_attention
from .state_space_v1 import state_space_v1
from .state_space_v2 import state_space_v2
from .unified_attention import unified_attention

__all__ = [
    "all_gather_matmul",
    "apply_native_sparse_attention",
    "attention",
    "blocksparse_attention",
    "chunked_prefill_paged_decode",
    "decode_attention",
    "deepseek_attn",
    "flash_attention",
    "flash_mla",
    "gated_delta_rule",
    "grouped_matmul",
    "grouped_matmulv3",
    "kda",
    "kda_decay",
    "kernel_delta_attention",
    "lightning_attn",
    "mean_pooling",
    "multi_latent_ragged_page_attention",
    "multi_latent_ragged_page_attention_v2",
    "page_attention",
    "prefill_page_attention",
    "quantized_matmul",
    "ragged_decode_attention",
    "ragged_gated_delta_rule",
    "ragged_page_attention_v2",
    "ragged_page_attention_v2_turboquant",
    "ragged_page_attention_v3",
    "ragged_page_attention_v3_turboquant",
    "recurrent",
    "recurrent_gla",
    "reduce_scatter_matmul",
    "ring_attention",
    "rwkv4",
    "rwkv6",
    "rwkv7",
    "rwkv7_mul",
    "scaled_dot_product_attention",
    "state_space_v1",
    "state_space_v2",
    "unified_attention",
]
