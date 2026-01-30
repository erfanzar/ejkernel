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

"""CUDA backend for Flash Attention.

Provides a native CUDA implementation of the Flash Attention algorithm via
the JAX Foreign Function Interface (FFI). The kernel leverages CUTLASS and
CUDA C++ to deliver high-throughput forward and backward passes on NVIDIA
GPUs. When input configurations are outside the supported CUDA path
(e.g., non-standard dtypes, attention masks, or segment IDs), the
implementation transparently falls back to the Triton or XLA backends.

This package re-exports the public :func:`flash_attention` entry point
from :mod:`._interface`.
"""

from ._interface import flash_attention

__all__ = ("flash_attention",)
