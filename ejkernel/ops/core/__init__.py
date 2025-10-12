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


"""Core abstractions for the ejkernel.ops framework.

This module provides the fundamental building blocks for implementing JAX operations
with automatic configuration management and performance optimization.

Classes:
    Kernel: Abstract base class for implementing custom operations
    Invocation: Represents a specific call to a kernel with arguments and metadata

Type Variables:
    Cfg: Configuration type parameter for kernels
    Out: Output type parameter for kernels

Functions:
    _has_custom_vjp: Utility to detect custom VJP implementations
"""

from .kernel import Invocation, Kernel, _get_platform_method, _has_custom_vjp
from .types import Cfg, Out

__all__ = ("Cfg", "Invocation", "Kernel", "Out", "_get_platform_method", "_has_custom_vjp")
