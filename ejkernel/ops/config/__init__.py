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


"""Configuration management and caching for ejkernel.ops.

This module provides a comprehensive configuration selection and caching system
with multiple fallback layers and automatic performance tuning capabilities.

Classes:
    ConfigCache: In-memory configuration cache
    PersistentCache: Disk-based configuration persistence
    ConfigSelectorChain: Multi-tier configuration selection system
    AutotunePolicy: Policy configuration for autotuning behavior
    Tuner: Performance benchmarking and autotuning
    overlay_cache: Context manager for temporary cache overrides
    policy_override: Context manager for temporary policy changes
"""

from .cache import ConfigCache, overlay_cache
from .persistent import PersistentCache
from .selection import AutotunePolicy, ConfigSelectorChain, Tuner, policy_override

__all__ = (
    "AutotunePolicy",
    "ConfigCache",
    "ConfigSelectorChain",
    "PersistentCache",
    "Tuner",
    "overlay_cache",
    "policy_override",
)
