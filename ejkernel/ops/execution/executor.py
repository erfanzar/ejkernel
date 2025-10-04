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
    EFORMER_OPS_RECORD: Set to "1" to enable invocation recording
    EFORMER_OPS_STAMP: Controls profiling metadata format:
        - "hash": Use operation hash for labeling (default)
        - "json": Use full JSON payload for labeling
        - "none": Disable profiling metadata

Example Usage:
    >>> cache = ConfigCache()
    >>> selector = ConfigSelectorChain(cache)
    >>> executor = Executor(selector)
    >>>
    >>> # Execute kernel with automatic config selection
    >>> result = executor(my_kernel, input_data)
    >>>
    >>> # Pre-compile kernel with chosen configuration
    >>> compiled_fn = executor.compile(my_kernel, example_input)
    >>> result = compiled_fn(actual_input)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Generic, Protocol

import jax

from ..core import Invocation, Kernel, _has_custom_vjp
from ..core.types import Cfg, Out
from ..utils.fingerprint import device_fingerprint, stable_json


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

    def __init__(self, chooser: ConfigChooser, stamp_prefix: str = "eformer_ops"):
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
                args=inv.args,
                kwargs=dict(inv.kwargs),
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

    def __call__(self, kernel: Kernel[Cfg, Out], *args, cfg: Cfg | None = None, stamp=True, **kwargs) -> Out:
        """Execute kernel with automatic configuration selection and management.

        This is the main execution method that orchestrates the complete execution
        pipeline including preprocessing, configuration selection, custom gradients,
        profiling, and invocation recording.

        Args:
            kernel: Kernel implementation to execute
            *args: Positional arguments for the kernel
            cfg: Optional configuration override (bypasses selection if provided)
            stamp: Whether to add profiling metadata to the operation
            **kwargs: Keyword arguments for the kernel

        Returns:
            Result of kernel execution with optimal configuration

        Note:
            This method handles both regular operations and kernels with custom
            VJP implementations. Custom gradients are automatically detected and
            properly integrated with JAX's differentiation system.
        """
        args2, kwargs2 = kernel.prepare(*args, **kwargs)
        inv = Invocation(op_id=kernel.op_id, args=args2, kwargs=kwargs2, override_cfg=cfg, stamp=stamp)
        chosen = self.chooser.choose(inv, kernel)

        if _has_custom_vjp(kernel):
            closed_kwargs = dict(kwargs2)

            def fwd_pos(*a):
                return kernel.fwd_with_residuals(*a, cfg=chosen, **closed_kwargs)

            def base(*a):
                y, _ = fwd_pos(*a)
                return y

            def fwd_rule(*a):
                y, res = fwd_pos(*a)
                return y, (res, y, a)

            def bwd_rule(payload, dy):
                res, y, a = payload
                grads = kernel.vjp(res, y, dy, *a, cfg=chosen, **closed_kwargs)
                if isinstance(grads, dict):
                    raise TypeError("kernel.vjp must return a tuple of grads for positional args.")
                return grads

            g = jax.custom_vjp(base)
            g.defvjp(fwd_rule, bwd_rule, optimize_remat=True)

            def fn(*a, **_ignored_kwargs):
                return g(*a)
        else:
            # Inline simple case for better performance
            def fn(*a, **k):
                return kernel.run(*a, cfg=chosen, **k)

        if os.getenv("EFORMER_OPS_RECORD", "0") == "1":
            try:
                from ..registry import record_invocation

                call_key = inv.make_key(kernel.key_builder)
                op_id_v = f"{kernel.op_id}@v{getattr(kernel, 'version', '0')}"
                record_invocation(device_fingerprint(), op_id_v, call_key, kernel, args2, kwargs2)
            except Exception:
                pass
        if stamp:
            mode = os.getenv("EFORMER_OPS_STAMP", "none").lower()
            if mode == "json":
                fn = self._stamp_json(kernel, inv, fn, chosen)
            elif mode == "hash":
                fn = self._stamp_hash(kernel, inv, fn, chosen)
            elif mode == "none":
                pass

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
        args2, kwargs2 = kernel.prepare(*args, **kwargs)
        inv = Invocation(op_id=kernel.op_id, args=args2, kwargs=kwargs2, override_cfg=cfg, stamp=False)
        return self.chooser.choose(inv, kernel)

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
            >>> # Fast execution with pre-selected config
            >>> result = compiled_matmul(x_actual, y_actual)
        """
        chosen = self.choose_config(kernel, *example_args, cfg=cfg, **example_kwargs)

        def run(*a, **k):
            return kernel.run(*a, cfg=chosen, **k)

        return jax.jit(run)
