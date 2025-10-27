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


"""Main execution engine for kernels with configuration management.

This module provides the Executor class, which serves as the central orchestrator
for running kernel operations with automatic configuration selection, custom
gradient support, and comprehensive profiling capabilities.

Key Components:
    Executor: Main execution engine coordinating the entire execution pipeline
    ConfigChooser: Protocol defining configuration selection interface

The Executor handles:
    - Argument preprocessing via kernel.prepare()
    - Configuration selection through ConfigChooser strategies
    - Custom VJP (Vector-Jacobian Product) implementation for gradients
    - Profiling metadata injection for performance analysis
    - Invocation recording for batch optimization
    - JAX compilation with pre-selected configurations

Execution Flow:
    1. Preprocess arguments using kernel.prepare()
    2. Create Invocation object with argument metadata
    3. Select configuration via ConfigChooser.choose()
    4. Set up custom VJP if kernel implements it
    5. Add profiling metadata based on environment settings
    6. Execute kernel with chosen configuration
    7. Record invocation for future optimization (if enabled)

Environment Variables:
    EJKERNEL_OPS_RECORD: Set to "1" to enable invocation recording
    EJKERNEL_OPS_STAMP: Controls profiling metadata format:
        - "hash": Use operation hash for labeling (default)
        - "json": Use full JSON payload for labeling
        - "none": Disable profiling metadata

Example Usage:
    >>> cache = ConfigCache()
    >>> selector = ConfigSelectorChain(cache)
    >>> executor = Executor(selector)
    >>>
    >>>
    >>> result = executor(my_kernel, input_data)
    >>>
    >>>
    >>> compiled_fn = executor.compile(my_kernel, example_input)
    >>> result = compiled_fn(actual_input)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Generic, Literal, Protocol

import jax
import jax.numpy as jnp
import jax.sharding
import jax.tree_util as jtu

from ..config.cache import _cache_overlay
from ..core import Invocation, Kernel, _get_platform_method, _has_custom_vjp
from ..core.types import Cfg, Out
from ..utils.fingerprint import abstractify, device_fingerprint, get_device_platform, stable_json


class ConfigChooser(Protocol):
    """Protocol for configuration selection strategies.

    Defines the interface that configuration selection strategies must implement.
    The primary implementer is ConfigSelectorChain, which provides a sophisticated
    multi-tier selection system with caching and autotuning.

    Methods:
        choose: Select optimal configuration for the given invocation and kernel
    """

    def choose(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Select optimal configuration for the given invocation.

        Args:
            inv: Invocation object containing arguments and metadata
            kernel: Kernel implementation requiring configuration

        Returns:
            Configuration object suitable for the kernel and invocation
        """
        ...


class Executor(Generic[Cfg, Out]):
    """Main execution engine for kernels with automatic configuration selection.

    The Executor coordinates the entire execution process:
    1. Preprocess arguments via kernel.prepare()
    2. Select configuration via the ConfigChooser
    3. Handle custom VJP if implemented by the kernel
    4. Add profiling metadata if requested
    5. Execute the kernel with the chosen configuration

    Supports both regular operations and custom gradient implementations.

    Attributes:
        chooser: Configuration selection strategy (typically ConfigSelectorChain)
        stamp_prefix: Prefix for profiling metadata labels
    """

    def __init__(self, chooser: ConfigChooser, stamp_prefix: str = "ejkernel_ops"):
        """Initialize executor with configuration chooser and profiling settings.

        Args:
            chooser: Configuration selection strategy (typically ConfigSelectorChain)
            stamp_prefix: Prefix for profiling metadata labels in compiled code
        """
        self.chooser = chooser
        self.stamp_prefix = stamp_prefix

    def _stamp_hash(self, kernel, inv, fn, cfg):
        """Add hash-based profiling metadata to function.

        Creates a compact label using operation ID and call signature hash
        for performance profiling and debugging.

        Args:
            kernel: Kernel being executed
            inv: Invocation object
            fn: Function to wrap with profiling metadata
            cfg: Configuration being used

        Returns:
            Function wrapped with hash-based profiling label
        """
        call_key = inv.make_key(kernel.key_builder)
        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        label = f"{self.stamp_prefix}#{op_id_v}:{call_key}"
        return self._stamp(label, fn)

    def _stamp_json(self, kernel, inv, fn, cfg):
        """Add JSON-based profiling metadata to function.

        Creates detailed profiling metadata including full operation context,
        arguments, and configuration for comprehensive debugging.

        Args:
            kernel: Kernel being executed
            inv: Invocation object
            fn: Function to wrap with profiling metadata
            cfg: Configuration being used

        Returns:
            Function wrapped with JSON profiling metadata

        Note:
            This mode provides more detailed information but may impact
            performance due to larger metadata payloads.
        """

        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        payload = stable_json(
            dict(
                op_id=op_id_v,
                args=abstractify(inv.args),
                kwargs=abstractify(dict(inv.kwargs)),
                cfg=cfg,
            )
        )

        def wrapped(*a, **k):
            with jax.named_scope(f"{self.stamp_prefix}:{payload}"):
                return fn(*a, **k)

        return wrapped

    def _stamp(self, name: str, fn: Callable) -> Callable:
        """Add profiling metadata to function using JAX naming primitives.

        Uses JAX's named_call if available, otherwise falls back to named_scope
        for adding operation labels to compiled code.

        Args:
            name: Label to attach to the operation
            fn: Function to wrap with profiling metadata

        Returns:
            Function wrapped with profiling label
        """
        if hasattr(jax, "named_call"):
            return jax.named_call(fn, name=name)

        def wrapped(*a, **k):
            with jax.named_scope(name):
                return fn(*a, **k)

        return wrapped

    def __call__(
        self,
        kernel: Kernel[Cfg, Out],
        *args,
        cfg: Cfg | None = None,
        stamp: bool = True,
        method: Literal["shard_map"] | None = None,
        mesh: jax.sharding.Mesh | None = None,
        in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None,
        out_specs: jax.sharding.PartitionSpec | None = None,
        check_vma: bool = False,
        **kwargs,
    ) -> Out:
        """Execute kernel with automatic configuration selection and management.

        This is the main execution method that orchestrates the complete execution
        pipeline including preprocessing, configuration selection, custom gradients,
        profiling, and invocation recording.

        Args:
            kernel: Kernel implementation to execute
            *args: Positional arguments for the kernel
            cfg: Optional configuration override (bypasses selection if provided)
            stamp: Whether to add profiling metadata to the operation
            method: Execution method - "shard_map" for distributed execution
            mesh: JAX device mesh for shard_map (required if method="shard_map")
            in_specs: Input partition specs for shard_map (required if method="shard_map")
            out_specs: Output partition spec for shard_map (required if method="shard_map")
            check_vma: Whether to check replication for shard_map
            **kwargs: Keyword arguments for the kernel

        Returns:
            Result of kernel execution with optimal configuration

        Note:
            This method handles both regular operations and kernels with custom
            VJP implementations. Custom gradients are automatically detected and
            properly integrated with JAX's differentiation system.

            When method="shard_map", the execution will be wrapped with shard_map
            for distributed computation across the specified mesh.
        """

        if "_cfg" in kwargs:
            cfg = kwargs.pop("_cfg")

        if method == "shard_map":
            if mesh is None:
                raise ValueError("mesh must be provided when method='shard_map'")
            if in_specs is None:
                raise ValueError("in_specs must be provided when method='shard_map'")
            if out_specs is None:
                raise ValueError("out_specs must be provided when method='shard_map'")

        args2, kwargs2 = kernel.prepare(*args, **kwargs)
        inv = Invocation(
            op_id=kernel.op_id,
            args=args2,
            kwargs=kwargs2,
            override_cfg=cfg,
            stamp=stamp,
            method=method,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        policy = getattr(self.chooser, "policy", None)
        if policy is not None and getattr(policy, "cache_miss_fallback", "heuristics") == "heuristics":
            chosen = self._choose_heuristics_only(inv, kernel)
        else:
            chosen = self.chooser.choose(inv, kernel)

        platform = get_device_platform()
        context = "shard_map" if method == "shard_map" else None

        if _has_custom_vjp(kernel, platform, context):
            fwd_method = (
                _get_platform_method(kernel, "fwd_with_residuals", platform, context) or kernel.fwd_with_residuals
            )
            vjp_method = _get_platform_method(kernel, "vjp", platform, context) or kernel.vjp

            full_leaves, treedef = jtu.tree_flatten((args2, kwargs2))
            is_arr = [isinstance(x, jax.Array) for x in full_leaves]

            const_leaves = [None if m else x for m, x in zip(is_arr, full_leaves, strict=False)]

            def _restore_args_kwargs(array_leaves):
                """Rebuild (args, kwargs) by merging dynamic array leaves into closed constants."""
                it = iter(array_leaves)
                merged = [next(it) if m else v for m, v in zip(is_arr, const_leaves, strict=False)]
                return jtu.tree_unflatten(treedef, merged)

            def fwd_arrays(*array_leaves):
                """Forward rule: takes only array leaves, rebuilds args/kwargs inside."""
                (a, k) = _restore_args_kwargs(array_leaves)
                y, res = fwd_method(*a, cfg=chosen, **k)
                return y, (tuple(array_leaves), res)

            def bwd_arrays(payload, dy):
                """Backward rule: rebuild args/kwargs, call kernel.vjp, and map grads to array inputs."""
                array_leaves, res = payload
                (a, k) = _restore_args_kwargs(array_leaves)

                grads = vjp_method(res, dy, *a, cfg=chosen, **k)
                if isinstance(grads, dict):
                    raise TypeError("kernel.vjp must return a tuple of grads for positional args.")
                grads = tuple(grads)
                if len(grads) != len(a):
                    raise TypeError(
                        f"kernel.vjp must return one grad per positional arg; got {len(grads)} for {len(a)} args."
                    )

                def align_arg_grad(x, g):
                    if g is None:
                        return jtu.tree_map(lambda t: None, x)
                    return jtu.tree_map(lambda _t, gg: gg, x, g)

                aligned_args_grads = tuple(align_arg_grad(x, g) for x, g in zip(a, grads, strict=False))

                zeros_kwargs = {
                    name: jtu.tree_map(lambda t: jnp.zeros_like(t) if isinstance(t, jax.Array) else None, val)
                    for name, val in k.items()
                }

                full_grads = (aligned_args_grads, zeros_kwargs)
                flat_grads, _ = jtu.tree_flatten(full_grads)

                grad_out = []
                itg = iter(flat_grads)
                for m in is_arr:
                    gleaf = next(itg)
                    if m:
                        if gleaf is None:
                            gleaf = 0.0
                        grad_out.append(gleaf)
                return tuple(grad_out)

            def primal_only_arrays(*array_inputs):
                return fwd_arrays(*array_inputs)[0]

            g = jax.custom_vjp(primal_only_arrays)
            g.defvjp(fwd_arrays, bwd_arrays)

            def fn(*a, **k):
                flat_call, _ = jtu.tree_flatten((a, k))
                array_in = [x for x, m in zip(flat_call, is_arr, strict=False) if m]
                return g(*array_in)
        else:
            run_method = _get_platform_method(kernel, "run", platform, context) or kernel.run

            def fn(*a, **k):
                return run_method(*a, cfg=chosen, **k)

        if os.getenv("EJKERNEL_OPS_RECORD", "0") == "1":
            try:
                from ..registry import record_invocation

                call_key = inv.make_key(kernel.key_builder)
                op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
                record_invocation(device_fingerprint(), op_id_v, call_key, kernel, args2, kwargs2)
            except Exception:
                pass
        if stamp:
            mode = os.getenv("EJKERNEL_OPS_STAMP", "none").lower()
            if mode == "json":
                fn = self._stamp_json(kernel, inv, fn, chosen)
            elif mode == "hash":
                fn = self._stamp_hash(kernel, inv, fn, chosen)
            elif mode == "none":
                pass

        if method == "shard_map":
            if not hasattr(kernel, "create_shard_map_wrapper"):
                raise AttributeError(f"Kernel {kernel.op_id} does not implement create_shard_map_wrapper")

            callback = None
            eagers = kernel.create_shard_map_wrapper(
                *args2,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=check_vma,
                cfg=chosen,
                **kwargs2,
            )

            if len(eagers) == 2:
                shard_map_fn, call_args = eagers
            elif len(eagers) == 3:
                shard_map_fn, call_args, callback = eagers
            outs = shard_map_fn(*call_args)
            if callback is not None:
                outs = callback(outs, cfg=chosen)
            return outs

        return fn(*args2, **kwargs2)

    def choose_config(self, kernel: Kernel[Cfg, Out], *args, cfg: Cfg | None = None, **kwargs) -> Cfg:
        """Select configuration for kernel without executing it.

        Useful for inspecting what configuration would be chosen for given
        arguments, or for pre-selecting configurations for compilation.

        Args:
            kernel: Kernel implementation requiring configuration
            *args: Positional arguments for the kernel
            cfg: Optional configuration override
            **kwargs: Keyword arguments for the kernel

        Returns:
            Configuration that would be selected for this invocation
        """

        if "_cfg" in kwargs:
            cfg = kwargs.pop("_cfg")

        args2, kwargs2 = kernel.prepare(*args, **kwargs)

        method = kwargs2.pop("method", None)
        mesh = kwargs2.pop("mesh", None)
        in_specs = kwargs2.pop("in_specs", None)
        out_specs = kwargs2.pop("out_specs", None)
        check_vma = kwargs2.pop("check_vma", False)

        inv = Invocation(
            op_id=kernel.op_id,
            args=args2,
            kwargs=kwargs2,
            override_cfg=cfg,
            stamp=False,
            method=method,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )
        policy = getattr(self.chooser, "policy", None)
        if policy is not None and getattr(policy, "cache_miss_fallback", "heuristics") == "heuristics":
            return self._choose_heuristics_only(inv, kernel)
        return self.chooser.choose(inv, kernel)

    def _choose_heuristics_only(self, inv: Invocation[Cfg, Out], kernel: Kernel[Cfg, Out]) -> Cfg:
        """Fast path: overlay -> memory cache -> persistent -> heuristics.

        Never calls the full selector.choose and never autotunes.
        Writes back the chosen config to memory/persistent caches if needed.
        """
        dev = device_fingerprint()
        op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
        call_key = inv.make_key(kernel.key_builder)

        for overlay in reversed(_cache_overlay.get()):
            if (cfg := overlay.get((dev, op_id_v, call_key))) is not None:
                return cfg

        if (cfg := self.chooser.cache.get(dev, op_id_v, call_key)) is not None:
            return cfg

        if self.chooser.persistent is not None:
            if (cfg := self.chooser.persistent.get(dev, op_id_v, call_key)) is not None:
                self.chooser.cache.put(dev, op_id_v, call_key, cfg)
                return cfg

        platform = get_device_platform()
        context = "shard_map" if getattr(inv, "method", None) == "shard_map" else None
        heuristic_cfg_method = _get_platform_method(kernel, "heuristic_cfg", platform, context) or kernel.heuristic_cfg
        cfg = heuristic_cfg_method(inv)

        self.chooser.cache.put(dev, op_id_v, call_key, cfg)
        if self.chooser.persistent is not None and self.chooser.persist_autotune:
            self.chooser.persistent.put(dev, op_id_v, call_key, cfg)
        return cfg

    def compile(self, kernel: Kernel[Cfg, Out], *example_args, cfg: Cfg | None = None, **example_kwargs):
        """Compile kernel with pre-selected configuration.

        Selects optimal configuration based on example arguments, then returns
        a JAX-compiled function that uses that configuration for all subsequent
        calls. This avoids configuration selection overhead during execution.

        Args:
            kernel: Kernel implementation to compile
            *example_args: Example positional arguments for configuration selection
            cfg: Optional configuration override
            **example_kwargs: Example keyword arguments for configuration selection

        Returns:
            JAX-compiled function with pre-selected configuration

        Example:
            >>> compiled_matmul = executor.compile(matmul_kernel, x_example, y_example)
            >>>
            >>> result = compiled_matmul(x_actual, y_actual)
        """

        if "_cfg" in example_kwargs:
            cfg = example_kwargs.pop("_cfg")

        chosen = self.choose_config(kernel, *example_args, cfg=cfg, **example_kwargs)

        def run(*a, **k):
            return kernel.run(*a, cfg=chosen, **k)

        return jax.jit(run)
