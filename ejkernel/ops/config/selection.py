# Copyright 2025 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""Configuration selection and autotuning system for kernel optimization.

This module provides a comprehensive configuration selection framework that
intelligently chooses optimal kernel configurations through a multi-tier
fallback chain. The system prioritizes cached results while supporting
automatic performance optimization when needed.

Key Components:
    ConfigSelectorChain: Main selection coordinator with fallback hierarchy
    AutotunePolicy: Configuration policy for autotuning behavior
    Tuner: Performance benchmarking and autotuning engine
    policy_override: Context manager for temporary policy changes

Selection Hierarchy (in order of priority):
    1. Override: Explicit configuration provided by caller
    2. Overlay: Temporary context-specific configuration overrides
    3. Memory Cache: Fast lookup for recently used configurations
    4. Persistent Cache: Disk-based storage across program runs
    5. Autotune: Benchmark candidates to find optimal configuration
    6. Heuristics: Kernel-provided default configuration
    7. Error: No configuration available (throws exception)

This design ensures optimal performance by:
    - Prioritizing fastest lookup methods (memory cache)
    - Preserving optimization results across runs (persistent cache)
    - Automatically finding optimal configurations (autotuning)
    - Providing sensible defaults (heuristics) as fallback

Example Usage:
    >>> cache = ConfigCache()
    >>> policy = AutotunePolicy(allow_autotune=True)
    >>> selector = ConfigSelectorChain(cache, policy)
    >>>
    >>> # Will autotune on first call, cache result for subsequent calls
    >>> config = selector.choose(invocation, kernel)
    >>>
    >>> # Temporarily disable autotuning
    >>> with policy_override(selector, allow_autotune=False):
    ...     config = selector.choose(invocation, kernel)  # Uses heuristics
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

import jax

from ..core import Invocation, Kernel
from ..utils.fingerprint import device_fingerprint
from .cache import ConfigCache, _cache_overlay
from .persistent import PersistentCache

Cfg = TypeVar("Cfg")
Out = TypeVar("Out")


@dataclass
class AutotunePolicy:
    """Configuration policy for autotuning behavior.

    Attributes:
        allow_autotune: Whether autotuning is permitted
        allow_heuristics: Whether heuristic configurations are allowed
        cache_miss_fallback: Strategy when no cached config is found ("heuristics" or "autotune")
    """

    allow_autotune: bool = True
    allow_heuristics: bool = True
    cache_miss_fallback: str = "heuristics"


class policy_override:
    """Context manager for temporarily overriding autotuning policy settings.

    Allows temporary modification of AutotunePolicy attributes within a context,
    automatically restoring the original values when exiting the context.

    This is useful for:
    - Disabling autotuning for specific operations
    - Forcing use of heuristics during debugging
    - Testing different policy configurations

    Args:
        selector: ConfigSelectorChain instance to modify
        **updates: Policy attributes to override with new values

    Example:
        >>> with policy_override(selector, allow_autotune=False):
        ...     result = executor(kernel, *args)  # Uses heuristics only
        >>> # allow_autotune is restored to original value here

        >>> with policy_override(selector, cache_miss_fallback="heuristics"):
        ...     config = selector.choose(inv, kernel)  # Skip autotuning
    """

    def __init__(self, selector: ConfigSelectorChain, **updates):
        """Initialize policy override context manager.

        Args:
            selector: ConfigSelectorChain to modify
            **updates: Policy attributes to override
        """
        self.selector = selector
        self.updates = updates
        self._prev = {}

    def __enter__(self):
        """Enter context and apply policy overrides.

        Returns:
            Self for use in with statements
        """
        for k, v in self.updates.items():
            self._prev[k] = getattr(self.selector.policy, k)
            setattr(self.selector.policy, k, v)
        return self

    def __exit__(self, *exc):
        """Exit context and restore original policy values.

        Args:
            *exc: Exception information (ignored)
        """
        for k, v in self._prev.items():
            setattr(self.selector.policy, k, v)


class Tuner(Generic[Cfg]):
    """Performance benchmarking and autotuning for kernel configurations.

    Measures execution time of different configurations and selects the fastest one.

    Attributes:
        warmup: Number of warmup iterations before timing
        iters: Number of timing iterations to average over
    """

    def __init__(self, warmup=1, iters=3):
        self.warmup, self.iters = warmup, iters

    def measure(self, fn, *args, **kwargs) -> float:
        """Measure the average execution time of a function over multiple iterations.

        Compiles the function with JAX and measures execution time with
        proper warmup to ensure stable performance measurements.

        Args:
            fn: Function to measure
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Average execution time per iteration in seconds
        """
        c = jax.jit(fn).lower(*args, **kwargs).compile()
        for _ in range(self.warmup):
            _ = c(*args, **kwargs).block_until_ready()
        import time

        t0 = time.perf_counter()
        for _ in range(self.iters):
            _ = c(*args, **kwargs).block_until_ready()
        return (time.perf_counter() - t0) / self.iters

    def autotune(self, make_fn, args, kwargs, candidates: Iterable[Cfg]) -> Cfg:
        """Benchmark all candidate configurations and return the fastest one.

        Tests each candidate configuration by measuring its execution time
        and selects the configuration with the lowest average execution time.

        Args:
            make_fn: Factory function that creates a function given a config
            args: Positional arguments for the function being benchmarked
            kwargs: Keyword arguments for the function being benchmarked
            candidates: Iterable of candidate configurations to test

        Returns:
            The configuration that achieved the fastest execution time

        Raises:
            RuntimeError: If no candidates are provided for testing
        """
        best_cfg, best_t = None, float("inf")
        for cfg in candidates:
            t = self.measure(make_fn(cfg), *args, **kwargs)
            if t < best_t:
                best_cfg, best_t = cfg, t
        if best_cfg is None:
            raise RuntimeError("No candidates provided for autotuning.")
        return best_cfg


class ConfigSelectorChain(Generic[Cfg, Out]):
    """Multi-tier configuration selection system with fallback chain.

    Selection order:
    1. Override (explicit configuration provided)
    2. Overlay (temporary context-specific overrides)
    3. In-memory cache (fast lookup for recently used configs)
    4. Persistent cache (disk-based storage across runs)
    5. Autotune (benchmark candidates to find optimal config)
    6. Heuristics (kernel-provided default configuration)
    7. Error (no configuration available)

    Attributes:
        cache: In-memory configuration cache
        policy: Autotuning behavior policy
        tuner: Performance benchmarking tool
        persistent: Optional disk-based cache
        persist_autotune: Whether to save autotuned configs to persistent storage
        on_event: Optional callback for selection events
        forbid_reautotune: Prevent re-autotuning the same operation
    """

    def __init__(
        self,
        cache: ConfigCache[Cfg],
        policy: AutotunePolicy | None = None,
        tuner: Tuner[Cfg] | None = None,
        persistent: PersistentCache[Cfg] | None = None,
        persist_autotune: bool = True,
        on_event: callable | None = None,
        forbid_reautotune: bool = True,
    ):
        self.cache = cache
        self.policy = policy or AutotunePolicy()
        self.tuner = tuner or Tuner()
        self.persistent = persistent
        self.persist_autotune = persist_autotune
        self.on_event = on_event
        self.forbid_reautotune = forbid_reautotune
        self._autotuned_keys: set[tuple[str, str, str]] = set()

    def choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Select optimal configuration using the fallback hierarchy.

        Implements the complete configuration selection algorithm, trying
        each method in order until a suitable configuration is found.

        Selection Priority (highest to lowest):
        1. Override: Explicit configuration in invocation
        2. Overlay: Temporary context-specific overrides
        3. Memory Cache: Previously computed optimal configurations
        4. Persistent Cache: Disk-stored configurations from previous runs
        5. Autotune: Benchmark candidates to find optimal configuration
        6. Heuristics: Kernel-provided default configuration

        Args:
            inv: Function invocation containing arguments and context
            kernel: Kernel implementation with candidate configurations

        Returns:
            Optimal configuration for this invocation

        Raises:
            RuntimeError: If no configuration can be determined
        """
        dev = device_fingerprint()
        op_id = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        call_key = inv.make_key(kernel.key_builder)
        # 1) explicit override
        if inv.override_cfg is not None:
            cfg = inv.override_cfg
            self._emit("override", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

        # 2) overlay cache
        for overlay in reversed(_cache_overlay.get()):
            if (cfg := overlay.get((dev, op_id, call_key))) is not None:
                self._emit("overlay_hit", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
                return cfg

        # 3) in-memory cache
        if (cfg := self.cache.get(dev, op_id, call_key)) is not None:
            self._emit("cache_hit", level="memory", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
            return cfg

        # 4) persistent cache (optional)
        if self.persistent is not None:
            if (cfg := self.persistent.get(dev, op_id, call_key)) is not None:
                self._emit("cache_hit", level="persistent", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
                # hydrate in-memory cache
                self.cache.put(dev, op_id, call_key, cfg)
                return cfg

        # 5) autotune
        if self.policy.cache_miss_fallback == "autotune" and self.policy.allow_autotune:
            if self.forbid_reautotune and (dev, op_id, call_key) in self._autotuned_keys:
                raise RuntimeError(f"Re-autotune requested for {(dev, op_id, call_key)}")
            candidates = tuple(kernel.candidate_cfgs(inv))
            self._emit("autotune_start", device=dev, op_id=op_id, call_key=call_key, candidates=len(candidates))

            kw = dict(inv.kwargs)
            static_fun_kwargs = {k: v for k, v in kw.items() if callable(v)}
            dyn_kwargs = {k: v for k, v in kw.items() if k not in static_fun_kwargs}

            def mk(c, _run=kernel.run, _static=static_fun_kwargs):
                def f(*a, **k):
                    return _run(*a, cfg=c, **(k | _static))

                return f

            best = self.tuner.autotune(mk, inv.args, dyn_kwargs, candidates)
            self._autotuned_keys.add((dev, op_id, call_key))
            self.cache.put(dev, op_id, call_key, best)
            if self.persistent is not None and self.persist_autotune:
                self.persistent.put(dev, op_id, call_key, best)
            self._emit("autotune_finish", device=dev, op_id=op_id, call_key=call_key, cfg=best)
            return best

        # 6) heuristics
        if self.policy.allow_heuristics:
            cfg = kernel.heuristic_cfg(inv)
            self._emit("heuristics", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
            # Optionally also populate caches for determinism across runs.
            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None and self.persist_autotune:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

        # 7) error
        self._emit("error", device=dev, op_id=op_id, call_key=call_key, reason="no_config")
        raise RuntimeError("No config found: override/overlay/cache/persistent/autotune/heuristics all unavailable.")

    def _emit(self, event: str, **data):
        """Emit selection event for monitoring and debugging.

        Calls the configured event callback with selection information.

        Args:
            event: Event type (e.g., 'cache_hit', 'autotune_start', 'error')
            **data: Additional event data (device, op_id, call_key, etc.)
        """
        if self.on_event:
            self.on_event(event, **data)
