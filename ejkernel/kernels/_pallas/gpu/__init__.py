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


"""Pallas GPU kernel implementations.

This module provides GPU-optimized kernels using Pallas with Triton backend.
Kernels leverage GPU's parallel architecture for efficient attention and
matrix operations.

Available Kernels:
    - ragged_decode_attention: Decode-phase attention for serving
    - scaled_dot_product_attention: Standard SDPA implementation
"""

from .ragged_decode_attention import ragged_decode_attention
from .scaled_dot_product_attention import scaled_dot_product_attention

__all__ = ("ragged_decode_attention", "scaled_dot_product_attention")
