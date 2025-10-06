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


"""GPU-optimized Pallas/Triton backend for scaled dot-product attention.

This submodule provides high-performance GPU implementations of scaled
dot-product attention using NVIDIA cuDNN via JAX's Pallas/Triton framework.
Optimized for NVIDIA GPUs with support for advanced attention patterns.
"""

from ._interface import scaled_dot_product_attention

__all__ = ("scaled_dot_product_attention",)
