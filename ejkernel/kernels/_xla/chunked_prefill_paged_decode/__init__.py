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

"""XLA backend for chunked prefill + paged decode attention.

This submodule provides XLA-optimized implementation for combined
prefill and decode operations with paged KV cache management.

Key Features:
    - Mixed prefill/decode batching for throughput optimization
    - Chunked prefill for memory-efficient processing
    - Paged KV cache with block table management
    - In-place cache updates during prefill phase
    - Variable chunk sizes for flexible scheduling

Algorithm:
    Combines prefill and decode in a single kernel invocation:
    1. Prefill sequences process multiple query tokens
    2. Decode sequences process single query token each
    3. KV cache is updated with new tokens via block tables
"""

from ._interface import chunked_prefill_paged_decode

__all__ = ("chunked_prefill_paged_decode",)
