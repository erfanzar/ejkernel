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


"""Calling library for Triton and JAX interoperability.

This module provides utilities for integrating Triton kernels with JAX,
including JIT compilation decorators, type conversions, and helper functions
for kernel development. It bridges the gap between Triton's GPU programming
model and JAX's functional array programming paradigm.

Key Components:
    - ejit: Enhanced JIT decorator for JAX functions
    - triton_call: Interface for calling Triton kernels from JAX
    - Type conversion utilities for Triton/JAX compatibility
    - Mathematical helper functions for kernel development
"""

from ._ejit import ejit
from ._pallas_call import buffered_pallas_call
from ._triton_call import get_triton_type, triton_call
from ._utils import cdiv, next_power_of_2, strides_from_shape

__all__ = (
    "buffered_pallas_call",
    "cdiv",
    "ejit",
    "get_triton_type",
    "next_power_of_2",
    "strides_from_shape",
    "triton_call",
)
