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


"""ejKernel: High-performance kernel library for JAX.

This module provides a comprehensive collection of optimized kernels for various
deep learning operations, with support for multiple backends (XLA, Triton, Pallas)
and platforms (TPU, GPU). The library focuses on efficient attention mechanisms,
matrix operations, and other performance-critical computations.

Key Features:
    - Multi-backend support (XLA, Triton, Pallas) for optimal performance
    - Platform-specific optimizations for TPU and GPU
    - Extensive collection of attention mechanisms (Flash, Sparse, Ring, etc.)
    - Automatic kernel selection and registration
    - Modular architecture for easy extension

Example:
    >>> import ejkernel
    >>> from ejkernel import Backend, Platform
    >>>
    >>> kernel = ejkernel.kernel_registry.get_kernel(
    ...     name="flash_attention",
    ...     backend=Backend.TRITON,
    ...     platform=Platform.GPU
    ... )
"""

import os as _os

_os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
__version__ = "0.0.11"

from . import kernels, modules, types
from .kernels import Backend, Platform, kernel_registry

kernel_registry.validate_signatures(None)

__all__ = ("Backend", "Platform", "kernel_registry", "kernels", "modules", "types")
