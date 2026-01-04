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

"""XLA backend for RWKV-4 time-mix recurrence.

This submodule provides XLA-optimized implementation of RWKV-4's
time-mix mechanism, a linear-complexity attention alternative.

Key Features:
    - O(N) complexity through channel-wise recurrence
    - Time-mixing with learnable decay and receptance gates
    - Efficient token mixing without attention matrices
    - State caching for autoregressive generation

Algorithm:
    RWKV-4 computes per-channel recurrence:
        wkv_t = exp(w + k_t) * v_t + exp(u + k_t) * v_t
        y_t = sigmoid(r_t) * wkv_t
    where w is the learned decay and u provides bonus for current token.

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
    https://arxiv.org/abs/2305.13048
"""

from ._interface import rwkv4

__all__ = [
    "rwkv4",
]
