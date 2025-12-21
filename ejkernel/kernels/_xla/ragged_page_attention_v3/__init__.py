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


"""XLA backend for Ragged Page Attention V3.

This submodule provides the latest XLA-optimized implementation of ragged
page attention with improved performance and additional features.

Key Features:
    - Enhanced variable-length sequence support
    - Improved paging algorithm for better memory efficiency
    - Sliding window attention support
    - Optimized tiling for different hardware configurations
"""

from ._interface import ragged_page_attention_v3

__all__ = ("ragged_page_attention_v3",)
