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

"""Core kernel infrastructure for configurable JAX operations.

This module provides the foundational classes for implementing high-performance
JAX operations with automatic configuration optimization and caching.

Key Classes:
    Invocation: Represents a specific call to a kernel with arguments and metadata
    Kernel: Abstract base class for implementing configurable operations

The kernel system enables:
    - Automatic hyperparameter optimization through configuration testing
    - Caching of optimal configurations for performance
    - Custom gradient implementations with VJP support
    - Flexible argument preprocessing and transformation
    - Device-aware configuration management

Kernel Implementation Pattern:
    1. Inherit from Kernel[ConfigType, OutputType]
    2. Implement run() method for the core operation
    3. Implement heuristic_cfg() for default configuration
    4. Optionally implement candidate_cfgs() for autotuning
    5. Optionally implement custom VJP methods for gradients

Example Implementation:
    >>> @dataclass
    >>> class MatMulConfig:
    ...     precision: str = 'default'
    ...     transpose_a: bool = False
    >>>
    >>> class MatMulKernel(Kernel[MatMulConfig, jax.Array]):
    ...     def run(self, a, b, cfg: MatMulConfig) -> jax.Array:
    ...         return jnp.dot(a, b, precision=cfg.precision)
    ...
    ...     def heuristic_cfg(self, inv) -> MatMulConfig:
    ...         return MatMulConfig()
    ...
    ...     def candidate_cfgs(self, inv):
    ...         return [MatMulConfig(p) for p in ['float32', 'bfloat16']]

Invocation Usage:
    The Invocation class captures all information needed for a kernel call:
    - Arguments and their shapes/types
    - Optional configuration overrides
    - Batching information for vmapping
    - Profiling and caching metadata

This design enables seamless integration with the autotuning and caching
system while providing a clean interface for operation implementers.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Generic

from ..utils.fingerprint import abstractify, short_hash
from .types import Cfg, Out


@dataclasses.dataclass(frozen=True)
class Invocation(Generic[Cfg, Out]):
    """Represents a specific call to a kernel with arguments and metadata.

    This dataclass captures all the information needed to execute a kernel,
    including arguments, configuration overrides, and execution metadata.

    Attributes:
        op_id: Unique identifier for the operation
        args: Positional arguments for the kernel
        kwargs: Keyword arguments for the kernel
        batch_axes: Optional mapping of parameter names to batch axes for vmapping
        override_cfg: Optional configuration to use instead of cached/computed ones
        stamp: Whether to add profiling metadata to the operation
    """

    op_id: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    batch_axes: Mapping[str, int] | None = None
    override_cfg: Cfg | None = None
    stamp: bool = True

    @property
    def call_key(self) -> str:
        """Generate a stable hash key for this invocation based on argument shapes and types.

        Creates a 16-character hash that uniquely identifies this invocation
        based on the abstract shapes and types of arguments, not their values.
        This enables caching of configurations based on operation signature.

        Returns:
            16-character hexadecimal hash string representing the call signature

        Note:
            The hash includes argument shapes/types, keyword argument shapes/types,
            and batch axes information. Array values are not included, allowing
            the same configuration to be reused for arrays with the same structure.
        """
        spec = dict(
            args_spec=abstractify(self.args),
            kwargs_spec=abstractify(dict(self.kwargs)),
            batch_axes=self.batch_axes,
        )
        return short_hash(spec)

    def make_key(self, key_builder=None) -> str:
        """Generate a cache key for this invocation, optionally using a custom key builder.

        Provides flexibility in cache key generation by allowing custom key builders
        while falling back to the default implementation.

        Args:
            key_builder: Optional function that takes an Invocation and returns a key

        Returns:
            Cache key string for this invocation

        Note:
            Custom key builders can include additional information like sharding
            or device placement for more sophisticated caching strategies.
        """
        if key_builder is not None:
            return key_builder(self)
        spec = dict(
            args_spec=abstractify(self.args),
            kwargs_spec=abstractify(dict(self.kwargs)),
            batch_axes=self.batch_axes,
        )
        return short_hash(spec)


class Kernel(Generic[Cfg, Out]):
    """Abstract base class for implementing custom JAX operations with configuration management.

    A Kernel encapsulates the logic for a specific operation, including how to execute it
    with different configurations, what configurations are available, and optionally how
    to compute custom gradients.

    Required methods to implement:
        run: Execute the operation with a given configuration
        heuristic_cfg: Provide a reasonable default configuration

    Optional methods:
        prepare: Preprocess arguments before execution
        candidate_cfgs: Provide alternative configurations for autotuning
        fwd_with_residuals: Forward pass with residuals for custom VJP
        vjp: Backward pass for custom VJP

    Attributes:
        op_id: Unique identifier for this operation
        key_builder: Optional custom function to generate cache keys
        version: Version string for cache invalidation
    """

    op_id: str
    key_builder: Callable[[Invocation[Cfg, Out]], str] | None = None
    version: str = "0"

    def __init__(self, op_id: str | None = None):
        if op_id is not None:
            self.op_id = op_id
        elif getattr(self, "op_id", None):
            pass
        else:
            self.op_id = f"{type(self).__module__}.{type(self).__name__}"

    def prepare(self, *args, **kwargs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Preprocess arguments before execution. Override to modify args/kwargs.

        This method is called before the run() method to allow transformation
        of arguments. Common use cases include shape validation, type conversion,
        or argument reordering.

        Args:
            *args: Positional arguments to preprocess
            **kwargs: Keyword arguments to preprocess

        Returns:
            Tuple of (processed_args, processed_kwargs)

        Example:
            >>> def prepare(self, x, y, **kwargs):
            ...     # Ensure inputs are JAX arrays
            ...     x = jnp.asarray(x)
            ...     y = jnp.asarray(y)
            ...     return (x, y), kwargs
        """
        return args, kwargs

    def run(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute the operation with the given configuration. Must be implemented.

        This is the core method that performs the actual computation. It receives
        the preprocessed arguments and a configuration object, and must return
        the operation result.

        Args:
            *args: Positional arguments (after prepare() preprocessing)
            cfg: Configuration object specifying how to execute the operation
            **kwargs: Keyword arguments (after prepare() preprocessing)

        Returns:
            Result of the operation

        Raises:
            NotImplementedError: Must be overridden in subclasses

        Example:
            >>> def run(self, x, y, cfg: MatMulConfig) -> jax.Array:
            ...     if cfg.transpose_a:
            ...         x = x.T
            ...     return jnp.dot(x, y, precision=cfg.precision)
        """
        raise NotImplementedError

    def heuristic_cfg(self, inv: Invocation[Cfg, Out]) -> Cfg:
        """Return a reasonable default configuration for this invocation. Must be implemented.

        Provides a sensible default configuration based on the invocation context.
        This configuration should work correctly for the given arguments, though
        it may not be optimal for performance.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Default configuration object

        Raises:
            NotImplementedError: Must be overridden in subclasses

        Example:
            >>> def heuristic_cfg(self, inv) -> MatMulConfig:
            ...     # Choose precision based on input dtypes
            ...     dtype = inv.args[0].dtype
            ...     precision = 'bfloat16' if dtype == jnp.bfloat16 else 'float32'
            ...     return MatMulConfig(precision=precision)
        """
        raise NotImplementedError

    def candidate_cfgs(self, inv: Invocation[Cfg, Out]) -> Iterable[Cfg]:
        """Return alternative configurations for autotuning. Defaults to just heuristic_cfg.

        Provides a set of configurations to test during autotuning. The autotuning
        system will benchmark each configuration and select the fastest one.

        Args:
            inv: Invocation object containing arguments and metadata

        Returns:
            Iterable of configuration objects to test

        Example:
            >>> def candidate_cfgs(self, inv):
            ...     return [
            ...         MatMulConfig(precision='float32', transpose_a=False),
            ...         MatMulConfig(precision='bfloat16', transpose_a=False),
            ...         MatMulConfig(precision='float32', transpose_a=True),
            ...     ]

        Note:
            The default implementation returns only the heuristic configuration.
            Override this method to enable autotuning with multiple options.
        """
        return [self.heuristic_cfg(inv)]

    def fwd_with_residuals(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass that returns residuals for custom VJP. Implement for custom gradients.

        When implementing custom gradients, this method performs the forward pass
        and returns both the result and any residual values needed for the
        backward pass.

        Args:
            *args: Positional arguments for the operation
            cfg: Configuration object
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, residuals)
            - operation_result: Same as run() method output
            - residuals: Any values needed for the backward pass

        Raises:
            NotImplementedError: Only implement if providing custom gradients

        Example:
            >>> def fwd_with_residuals(self, x, y, cfg):
            ...     result = jnp.dot(x, y)
            ...     residuals = (x, y, cfg)  # Save inputs for backward pass
            ...     return result, residuals

        Note:
            Must be implemented together with vjp() method for custom gradients.
        """
        raise NotImplementedError

    def vjp(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for custom VJP. Return gradients for positional args only.

        Computes vector-Jacobian products (gradients) for the custom operation.
        This method is called during backpropagation to compute gradients with
        respect to the positional arguments.

        Args:
            residuals: Values returned from fwd_with_residuals()
            y: Forward pass output (from fwd_with_residuals())
            dy: Incoming gradients (cotangents) with respect to y
            *args: Original positional arguments
            cfg: Configuration object
            **kwargs: Original keyword arguments

        Returns:
            Tuple of gradients for each positional argument
            (None for arguments that don't need gradients)

        Raises:
            NotImplementedError: Only implement if providing custom gradients

        Example:
            >>> def vjp(self, residuals, y, dy, *args, cfg, **kwargs):
            ...     x, y_orig, _ = residuals
            ...     dx = jnp.dot(dy, y_orig.T)
            ...     dy_orig = jnp.dot(x.T, dy)
            ...     return dx, dy_orig

        Note:
            Must be implemented together with fwd_with_residuals() method.
        """
        raise NotImplementedError


def _has_custom_vjp(k: Kernel) -> bool:
    """Check if a kernel has implemented custom VJP (vector-Jacobian product) methods.

    Returns True if both fwd_with_residuals and vjp methods have been overridden
    from the base Kernel class.
    """
    try:
        return type(k).fwd_with_residuals is not Kernel.fwd_with_residuals and type(k).vjp is not Kernel.vjp
    except AttributeError:
        return False
