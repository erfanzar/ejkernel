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


"""Configuration caching system for ejkernel operations.

This module provides a thread-safe caching mechanism for storing and retrieving
optimal kernel configurations. The cache system supports:

- Device-specific configuration storage
- Operation-specific caching with call signature hashing
- Temporary cache overlays for testing and debugging
- Context-aware configuration management

Key Components:
    ConfigCache: Main cache storage with thread-safe operations
    overlay_cache: Context manager for temporary cache overrides
    _cache_overlay: Context variable for managing overlay stack

The caching system is designed to minimize recomputation of optimal configurations
by storing results based on device characteristics, operation types, and input
signatures. This significantly improves performance for repeated operations.
"""

from __future__ import annotations

import contextvars
from typing import Any, Generic, TypeVar

Cfg = TypeVar("Cfg")


class ConfigCache(Generic[Cfg]):
    """Thread-safe cache for storing kernel configurations.

    The ConfigCache stores optimal configurations for kernel operations based on
    device characteristics, operation identifiers, and call signatures. This enables
    efficient reuse of previously determined optimal configurations.

    The cache key consists of:
    - device_fingerprint: Unique identifier for the target device
    - op_id: Operation identifier (e.g., 'matmul', 'conv')
    - call_key: Hash of the call signature (args, kwargs)

    Type Parameters:
        Cfg: Configuration type (e.g., dict, dataclass, etc.)

    Attributes:
        _data: Internal storage mapping cache keys to configurations
    """

    def __init__(self):
        """Initialize an empty configuration cache."""
        self._data: dict[tuple[str, str, str], Cfg] = {}

    def get(self, dev: str, op_id: str, call_key: str) -> Cfg | None:
        """Retrieve a cached configuration.

        Args:
            dev: Device fingerprint identifying the target device
            op_id: Operation identifier
            call_key: Call signature hash

        Returns:
            Cached configuration if found, None otherwise
        """
        return self._data.get((dev, op_id, call_key))

    def put(self, dev: str, op_id: str, call_key: str, cfg: Cfg) -> None:
        """Store a configuration in the cache.

        Args:
            dev: Device fingerprint identifying the target device
            op_id: Operation identifier
            call_key: Call signature hash
            cfg: Configuration to cache
        """
        self._data[(dev, op_id, call_key)] = cfg

    def clear(self) -> None:
        """Clear all cached configurations."""
        self._data.clear()

    def size(self) -> int:
        """Get the number of cached configurations.

        Returns:
            Number of entries in the cache
        """
        return len(self._data)

    def keys(self) -> list[tuple[str, str, str]]:
        """Get all cache keys.

        Returns:
            List of (device_fingerprint, op_id, call_key) tuples
        """
        return list(self._data.keys())


_cache_overlay: contextvars.ContextVar[list[dict[tuple[str, str, str], Any]]] = contextvars.ContextVar(
    "ejkernel_ops_cache_overlay",
    default=[],  # noqa
)


class overlay_cache:
    """Context manager for temporarily overriding cache entries.

    This context manager allows temporary cache overlays that shadow the main cache
    during execution. Useful for testing, debugging, or providing custom configurations
    for specific contexts without permanently modifying the global cache.

    The overlay uses a context variable stack to support nested overlays.

    Args:
        mapping: Dictionary mapping cache keys to override configurations

    Example:
        >>> cache = ConfigCache()
        >>> cache.put('dev1', 'op1', 'key1', 'original_config')
        >>> override = {('dev1', 'op1', 'key1'): 'override_config'}
        >>> with overlay_cache(override):
        ...
        ...     config = cache.get('dev1', 'op1', 'key1')
        >>>
    """

    def __init__(self, mapping: dict[tuple[str, str, str], Any]):
        """Initialize cache overlay with mapping.

        Args:
            mapping: Dictionary of cache key -> configuration overrides
        """
        self.mapping = mapping
        self.token = None

    def __enter__(self):
        """Enter the cache overlay context.

        Pushes the overlay mapping onto the context variable stack.

        Returns:
            Self for use in with statements
        """
        stack = list(_cache_overlay.get())
        stack.append(self.mapping)
        self.token = _cache_overlay.set(stack)

        try:
            import jax

            if not hasattr(self, "_ejk_user_context"):
                self._ejk_user_context = jax.make_user_context(()) if hasattr(jax, "make_user_context") else None
            if self._ejk_user_context:
                self._ejk_user_context = self._ejk_user_context((*self._ejk_user_context.value, id(self)))
                self._ejk_user_context.__enter__()
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        """Exit the cache overlay context.

        Pops the overlay mapping from the context variable stack,
        restoring the previous cache state.

        Args:
            *exc: Exception information (ignored)
        """
        _cache_overlay.reset(self.token)
        try:
            if hasattr(self, "_ejk_user_context") and self._ejk_user_context:
                self._ejk_user_context.__exit__(*exc)
        except Exception:
            pass
