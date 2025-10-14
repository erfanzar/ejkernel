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


"""Kernel implementations for various platforms and backends.

This module provides a unified interface to kernel implementations across
different hardware platforms (GPU, TPU) and implementation frameworks
(Triton, Pallas, XLA, CUDA).

The kernel system supports:
- Multi-platform implementations with automatic selection
- Priority-based kernel selection
- Hardware-specific optimizations
- Unified API across different backends

Submodules:
    cuda: CUDA-specific kernel implementations
    pallas: Pallas kernel implementations for TPU/GPU
    triton: Triton kernel implementations for GPU
    xla: XLA-based kernel implementations

Key Components:
    Backend: Enumeration of supported hardware backends
    Platform: Enumeration of supported implementation platforms
    kernel_registry: Central registry for kernel registration and lookup

Example:
    >>> from ejkernel.kernels import kernel_registry, Platform, Backend
    >>>
    >>> kernel = kernel_registry.get(
    ...     "flash_attention",
    ...     platform=Platform.TRITON,
    ...     backend=Backend.GPU
    ... )
    >>>
    >>> output = kernel(query, key, value)
"""

from . import _cuda as cuda
from . import _pallas as pallas
from . import _triton as triton
from . import _xla as xla
from ._registry import Backend, Platform, kernel_registry

__all__ = (
    "Backend",
    "Platform",
    "cuda",
    "kernel_registry",
    "pallas",
    "triton",
    "xla",
)
