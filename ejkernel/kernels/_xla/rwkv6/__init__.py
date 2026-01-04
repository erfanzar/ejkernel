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

"""XLA backend for RWKV-6 recurrence.

This submodule provides XLA-optimized implementation of RWKV-6's
recurrence mechanism with DPLR (Decoupled Position-aware Linear Recurrence).

Key Features:
    - O(N) complexity through linear recurrence
    - DPLR state update for improved expressivity
    - Position-aware decay with data-dependent gating
    - Efficient state caching for incremental inference

Algorithm:
    RWKV-6 extends RWKV-4 with data-dependent decay:
        h_t = diag(w_t) * h_{t-1} + k_t^T * v_t
        y_t = sigmoid(r_t) * (h_t @ q_t)
    where w_t is input-dependent (unlike fixed w in RWKV-4).

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
    https://arxiv.org/abs/2305.13048
"""

from ._interface import rwkv6

__all__ = [
    "rwkv6",
]
