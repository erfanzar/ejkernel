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


"""
Flash Multi-Latent Attention (MLA) module.

This module provides Triton-accelerated implementations of Multi-Latent Attention,
an efficient attention mechanism that uses latent representations to reduce
memory and computational requirements while maintaining model expressiveness.

Key Features:
- Memory-efficient attention using latent compression
- Triton-optimized forward and backward passes
- JAX integration with custom gradients
- Support for causal and non-causal attention patterns
"""

from ._interface import (
    flash_mla_attention,
    flash_mla_attention_call,
)

__all__ = [
    "flash_mla_attention",
    "flash_mla_attention_call",
]
