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
    >>>
    >>> config = selector.choose(invocation, kernel)
    >>>
    >>>
    >>> with policy_override(selector, allow_autotune=False):
    ...     config = selector.choose(invocation, kernel)
"""

from __future__ import annotations

import os
import pprint
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jcore
from jax import tree_util as jtu

from ..core import Invocation, Kernel, _get_platform_method
from ..utils.fingerprint import device_fingerprint, get_device_platform
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
    cache_miss_fallback: Literal["autotune", "heuristics"] = "autotune"
    validate_backward: bool = False


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
        ...     result = executor(kernel, *args)
        >>>

        >>> with policy_override(selector, cache_miss_fallback="heuristics"):
        ...     config = selector.choose(inv, kernel)
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
        """Measure average execution time with optional backward validation.

        Deep-flatten (args, kwargs) so only array-like leaves are dynamic:
        - Arrays or JAX tracers become dynamic parameters to the jitted function.
        - Everything else (dtype, strings, bools, callables, nested containers)
          is captured as Python constants in the closure.
        - Tracer-like arrays (e.g., ShardMapTracer, DynamicJaxprTracer) are converted
          to concrete zeros of the same shape/dtype before compile and timing.
        - If _ejk_validate_backward=True, we differentiate a scalar loss w.r.t.
          float/complex array leaves only; others are treated as non-diff.
        - If a kernel uses precompiled functions that can't be transformed, we fall back
          to forward-only timing, and if needed, to non-jitted forward timing.

        Args:
            fn: Function to measure (possibly tagged with _ejk_validate_backward)
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Average execution time per iteration in seconds
        """

        def _is_arrayish(x) -> bool:
            return isinstance(x, jax.Array | np.ndarray) or isinstance(x, jcore.Tracer)

        def _to_concrete(x):
            if isinstance(x, jax.Array | np.ndarray):
                return x
            shape = getattr(x, "shape", None)
            dtype = getattr(x, "dtype", None)
            aval = getattr(x, "aval", None)
            if (shape is None or dtype is None) and aval is not None:
                shape = getattr(aval, "shape", None)
                dtype = getattr(aval, "dtype", None)
            if shape is not None and dtype is not None:
                return jnp.zeros(shape, dtype)
            return jnp.asarray(x)

        def _block_all(x):
            return jtu.tree_map(lambda t: t.block_until_ready() if hasattr(t, "block_until_ready") else t, x)

        leaves, treedef = jtu.tree_flatten((args, kwargs))
        is_arr = [_is_arrayish(x) for x in leaves]
        const_leaves = [None if m else x for m, x in zip(is_arr, leaves, strict=False)]
        arr_leaves = [x for m, x in zip(is_arr, leaves, strict=False) if m]
        arr0 = tuple(_to_concrete(x) for x in arr_leaves)

        def _restore_args_kwargs(array_leaves):
            it = iter(array_leaves)
            merged = [next(it) if m else v for m, v in zip(is_arr, const_leaves, strict=False)]
            return jtu.tree_unflatten(treedef, merged)

        method = getattr(fn, "_ejk_method", "regular")
        validate_bwd = bool(getattr(fn, "_ejk_validate_backward", False))

        if method == "shard_map" and not getattr(fn, "_ejk_validate_backward", False):
            validate_bwd = False

        def _time_forward(jitted: bool = True) -> float:
            def core(*arrs):
                (aa, kk) = _restore_args_kwargs(arrs)
                return fn(*aa, **kk)

            if jitted:
                c = jax.jit(core).lower(*arr0).compile()
                for _ in range(self.warmup):
                    _block_all(c(*arr0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(c(*arr0))
                return (time.perf_counter() - t0) / self.iters
            else:
                for _ in range(self.warmup):
                    _block_all(core(*arr0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(core(*arr0))
                return (time.perf_counter() - t0) / self.iters

        if validate_bwd:

            def _is_diff(x):
                try:
                    dt = np.dtype(getattr(x, "dtype", None))
                    return np.issubdtype(dt, np.inexact)
                except Exception:
                    return False

            diff_mask = [_is_diff(x) for x in arr0]
            has_diff = any(diff_mask)
            if not has_diff:
                try:
                    return _time_forward(jitted=True)
                except Exception:
                    return _time_forward(jitted=False)

            def _split(arrs):
                theta, nondiff = [], []
                for m, v in zip(diff_mask, arrs, strict=False):
                    (theta if m else nondiff).append(v)
                return tuple(theta), tuple(nondiff)

            def _merge(theta, nondiff):
                it_t, it_n = iter(theta), iter(nondiff)
                return tuple(next(it_t) if m else next(it_n) for m in diff_mask)

            def loss(theta, nondiff):
                arrs = _merge(theta, nondiff)
                (aa, kk) = _restore_args_kwargs(arrs)
                y = fn(*aa, **kk)
                return jnp.sum(y)

            try:
                grad_core = jax.jit(jax.grad(loss, argnums=0))
                theta0, nondiff0 = _split(arr0)
                c = grad_core.lower(theta0, nondiff0).compile()
                for _ in range(self.warmup):
                    _block_all(c(theta0, nondiff0))
                t0 = time.perf_counter()
                for _ in range(self.iters):
                    _block_all(c(theta0, nondiff0))
                return (time.perf_counter() - t0) / self.iters
            except Exception as e:
                msg = str(e)
                if ("Cannot apply JAX transformations" in msg) or ("Leaked trace" in msg):
                    try:
                        return _time_forward(jitted=True)
                    except Exception:
                        return _time_forward(jitted=False)

                raise

        try:
            return _time_forward(jitted=True)
        except Exception:
            return _time_forward(jitted=False)

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
        last_err = None
        for cfg in candidates:
            try:
                t = self.measure(make_fn(cfg), *args, **kwargs)
                if os.getenv("EJKERNEL_LOG_AUTOTUNE", "0") == "1":
                    pprint.pprint({"config": cfg, "time": t})
            except Exception as e:
                last_err = e
                continue
            if t < best_t:
                best_cfg, best_t = cfg, t
        if best_cfg is None:
            traceback.print_exception(last_err)
            raise RuntimeError(f"All candidates failed during autotune: {last_err}")
        if os.getenv("EJKERNEL_LOG_AUTOTUNE", "0") == "1":
            pprint.pprint({"best_config": best_cfg, "best_time": best_t})
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

        if inv.override_cfg is not None:
            cfg = inv.override_cfg
            self._emit("override", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

        for overlay in reversed(_cache_overlay.get()):
            if (cfg := overlay.get((dev, op_id, call_key))) is not None:
                self._emit("overlay_hit", device=dev, op_id=op_id, call_key=call_key, cfg=cfg)
                return cfg

        if (cfg := self.cache.get(dev, op_id, call_key)) is not None:
            self._emit(
                "cache_hit",
                level="memory",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                cfg=cfg,
            )
            return cfg

        if self.persistent is not None:
            if (cfg := self.persistent.get(dev, op_id, call_key)) is not None:
                self._emit(
                    "cache_hit",
                    level="persistent",
                    device=dev,
                    op_id=op_id,
                    call_key=call_key,
                    cfg=cfg,
                )

                self.cache.put(dev, op_id, call_key, cfg)
                return cfg

        if self.policy.cache_miss_fallback == "autotune" and self.policy.allow_autotune:
            if self.forbid_reautotune and (dev, op_id, call_key) in self._autotuned_keys:
                raise RuntimeError(f"Re-autotune requested for {(dev, op_id, call_key)}")

            platform = get_device_platform()
            context = "shard_map" if inv.method == "shard_map" else None

            candidate_cfgs_method = _get_platform_method(kernel, "candidate_cfgs", platform, context)
            if candidate_cfgs_method:
                candidates = tuple(candidate_cfgs_method(inv))
            else:
                candidates = tuple(kernel.candidate_cfgs(inv))

            self._emit(
                "autotune_start",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                candidates=len(candidates),
                platform=platform,
                method=inv.method,
            )

            kw = dict(inv.kwargs)

            def _is_arrayish(x) -> bool:
                return isinstance(x, jax.Array | np.ndarray) or isinstance(x, jcore.Tracer)

            static_fun_kwargs = {k: v for k, v in kw.items() if callable(v)}
            dyn_kwargs = kw

            if inv.method == "shard_map":
                if not hasattr(kernel, "create_shard_map_wrapper"):
                    raise RuntimeError(
                        f"Kernel {kernel.op_id} does not implement create_shard_map_wrapper for shard_map benchmarking"
                    )

                def mk(c, _static=static_fun_kwargs):
                    def f(*a, **k):
                        callback = None
                        eagers = kernel.create_shard_map_wrapper(
                            *a,
                            cfg=c,
                            mesh=inv.mesh,
                            in_specs=inv.in_specs,
                            out_specs=inv.out_specs,
                            check_vma=inv.check_vma,
                            **(k | _static),
                        )
                        if len(eagers) == 2:
                            shard_map_fn, call_args = eagers
                        elif len(eagers) == 3:
                            shard_map_fn, call_args, callback = eagers

                        outs = shard_map_fn(*call_args)
                        if callback is not None:
                            outs = callback(outs, cfg=c)
                        return outs

                    f._ejk_method = "shard_map"
                    if self.policy.validate_backward and getattr(kernel, "supports_grad_validation", False):
                        f._ejk_validate_backward = True
                    return f
            else:
                run_method = _get_platform_method(kernel, "run", platform, context) or kernel.run

                def mk(c, _run=run_method, _static=static_fun_kwargs):
                    def f(*a, **k):
                        return _run(*a, cfg=c, **(k | _static))

                    f._ejk_method = "regular"
                    if self.policy.validate_backward:
                        f._ejk_validate_backward = True
                    return f

            best = self.tuner.autotune(mk, inv.args, dyn_kwargs, candidates)
            self._autotuned_keys.add((dev, op_id, call_key))
            self.cache.put(dev, op_id, call_key, best)
            if self.persistent is not None and self.persist_autotune:
                self.persistent.put(dev, op_id, call_key, best)
            self._emit(
                "autotune_finish",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                cfg=best,
                platform=platform,
                method=inv.method,
            )
            return best

        if self.policy.allow_heuristics:
            platform = get_device_platform()
            context = "shard_map" if inv.method == "shard_map" else None

            heuristic_cfg_method = _get_platform_method(kernel, "heuristic_cfg", platform, context)
            if heuristic_cfg_method:
                cfg = heuristic_cfg_method(inv)
            else:
                cfg = kernel.heuristic_cfg(inv)
            self._emit(
                "heuristics",
                device=dev,
                op_id=op_id,
                call_key=call_key,
                cfg=cfg,
                platform=platform,
                method=inv.method,
            )

            self.cache.put(dev, op_id, call_key, cfg)
            if self.persistent is not None and self.persist_autotune:
                self.persistent.put(dev, op_id, call_key, cfg)
            return cfg

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
