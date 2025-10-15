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

import jax
import jax.sharding
from jax import shard_map

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
        method: Execution method (e.g., "shard_map" or None for standard)
        mesh: JAX mesh for shard_map execution
        in_specs: Input partition specs for shard_map
        out_specs: Output partition spec for shard_map
        check_vma: Whether to check for valid memory access in shard_map
    """

    op_id: str
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    batch_axes: Mapping[str, int] | None = None
    override_cfg: Cfg | None = None
    stamp: bool = True
    method: str | None = None
    mesh: jax.sharding.Mesh | None = None
    in_specs: tuple[jax.sharding.PartitionSpec, ...] | None = None
    out_specs: jax.sharding.PartitionSpec | None = None
    check_vma: bool = False

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
            batch axes information, and execution method. Array values are not included,
            allowing the same configuration to be reused for arrays with the same structure.
        """
        spec = dict(
            args_spec=abstractify(self.args),
            kwargs_spec=abstractify(dict(self.kwargs)),
            batch_axes=self.batch_axes,
            method=self.method,
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
            method=self.method,
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
        run_shard_map: Specialized execution for shard_map contexts
        fwd_with_residuals_shard_map: Forward pass with residuals for shard_map
        vjp_shard_map: Backward pass for shard_map

    Method Naming Convention:
        - Platform-specific: {method}_{platform} (e.g., run_gpu, run_tpu)
        - Context-specific: {method}_{context} (e.g., run_shard_map)
        - Composite: {method}_{context}_{platform} (e.g., run_shard_map_gpu)

        Platforms: 'gpu', 'tpu', 'cpu' (hardware backends)
        Contexts: 'shard_map' (execution modes/environments)

        Priority: composite > context > platform > generic

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
            ...
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
            ...
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
            ...     residuals = (x, y, cfg)
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

    def run_shard_map(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute the operation within a shard_map context. Optional override.

        This method can be implemented to provide a specialized version of the
        operation that runs efficiently within JAX's shard_map context. If not
        implemented, the regular run() method will be used as fallback.

        Args:
            *args: Positional arguments (after prepare() preprocessing)
            cfg: Configuration object specifying how to execute the operation
            **kwargs: Keyword arguments (after prepare() preprocessing)

        Returns:
            Result of the operation

        Raises:
            NotImplementedError: Only implement if providing shard_map-specific execution

        Example:
            >>> def run_shard_map(self, x, y, cfg: MatMulConfig) -> jax.Array:
            ...
            ...     return custom_sharded_matmul(x, y, cfg)

        Note:
            This method is automatically detected by _get_platform_method() and
            used when 'shard_map' is specified as the platform.
        """
        raise NotImplementedError

    def fwd_with_residuals_shard_map(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass with residuals for shard_map context. Optional override.

        Specialized forward pass for shard_map contexts that returns both the
        result and residuals needed for the backward pass.

        Args:
            *args: Positional arguments for the operation
            cfg: Configuration object
            **kwargs: Keyword arguments for the operation

        Returns:
            Tuple of (operation_result, residuals)
            - operation_result: Same as run_shard_map() method output
            - residuals: Any values needed for the backward pass

        Raises:
            NotImplementedError: Only implement if providing custom gradients for shard_map

        Note:
            Must be implemented together with vjp_shard_map() method for custom
            gradients in shard_map contexts.
        """
        raise NotImplementedError

    def vjp_shard_map(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for shard_map context. Optional override.

        Specialized backward pass for shard_map contexts that computes gradients
        with respect to positional arguments.

        Args:
            residuals: Values returned from fwd_with_residuals_shard_map()
            y: Forward pass output (from fwd_with_residuals_shard_map())
            dy: Incoming gradients (cotangents) with respect to y
            *args: Original positional arguments
            cfg: Configuration object
            **kwargs: Original keyword arguments

        Returns:
            Tuple of gradients for each positional argument
            (None for arguments that don't need gradients)

        Raises:
            NotImplementedError: Only implement if providing custom gradients for shard_map

        Note:
            Must be implemented together with fwd_with_residuals_shard_map() method.
        """
        raise NotImplementedError

    def run_shard_map_gpu(self, *args, cfg: Cfg, **kwargs) -> Out:
        """Execute operation in shard_map context on GPU. Optional override.

        Most specific implementation combining shard_map execution context with
        GPU platform optimizations.

        Example:
            >>> def run_shard_map_gpu(self, x, y, cfg: MyConfig) -> jax.Array:
            ...
            ...     return gpu_sharded_operation(x, y, cfg)
        """
        raise NotImplementedError

    def fwd_with_residuals_shard_map_gpu(self, *args, cfg: Cfg, **kwargs) -> tuple[Out, Any]:
        """Forward pass with residuals for shard_map on GPU. Optional override."""
        raise NotImplementedError

    def vjp_shard_map_gpu(self, residuals: Any, y: Out, dy: Out, *args, cfg: Cfg, **kwargs):
        """Backward pass for shard_map on GPU. Optional override."""
        raise NotImplementedError

    def create_shard_map_wrapper(
        self,
        fn: Callable,
        *args,
        mesh: jax.sharding.Mesh,
        in_specs: tuple[jax.sharding.PartitionSpec, ...],
        out_specs: jax.sharding.PartitionSpec,
        check_vma: bool = False,
        **kwargs,
    ) -> tuple[Callable, tuple]:
        """Create a shard_map wrapper around a function with fixed kwargs.

        This helper method simplifies the pattern of wrapping functions with shard_map
        by handling the functools.partial and shard_map setup automatically.

        Args:
            fn: Function to wrap with shard_map (e.g., flash_attention)
            *args: Positional arguments that will be passed through shard_map
            mesh: JAX device mesh for distributed execution
            in_specs: Partition specs for input arguments (must match len(args))
            out_specs: Partition spec for output
            check_vma: Whether to check replication in shard_map
            **kwargs: Keyword arguments to fix via functools.partial

        Returns:
            Tuple of (shard_map_wrapped_function, call_arguments)
            - shard_map_wrapped_function: Function that takes positional args
            - call_arguments: Tuple of args to pass to the wrapped function

        Example:
            >>> from ejkernel.modules.operations import flash_attention
            >>> attn = FlashAttention()
            >>>
            >>>
            >>> shard_map_fn, call_args = attn.create_shard_map_wrapper(
            ...     flash_attention,
            ...     query, key, value,
            ...     mesh=my_mesh,
            ...     in_specs=(q_spec, k_spec, v_spec),
            ...     out_specs=out_spec,
            ...     causal=True,
            ...     softmax_scale=0.125
            ... )
            >>>
            >>>
            >>> output = shard_map_fn(*call_args)

        Note:
            This follows the EasyDeL pattern where attention operations are wrapped
            with shard_map for distributed execution. The function `fn` is called
            with the positional args and fixed kwargs inside the shard_map context.
        """

        def _wrapped(*call_args):
            return fn(*call_args, **kwargs)

        shard_map_fn = shard_map(
            _wrapped,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=check_vma,
        )

        return shard_map_fn, args


def _has_custom_vjp(
    k: Kernel,
    platform: str | None = None,
    context: str | None = None,
) -> bool:
    """Check if a kernel has implemented custom VJP (vector-Jacobian product) methods.

    Returns True if both fwd_with_residuals and vjp methods have been overridden
    from the base Kernel class. Supports context and platform-specific methods
    (e.g., fwd_with_residuals_shard_map_gpu).

    Args:
        k: Kernel instance to check
        platform: Optional platform identifier (e.g., 'gpu', 'tpu', 'cpu')
        context: Optional execution context (e.g., 'shard_map')

    Returns:
        True if kernel has custom VJP implementation (generic, context-specific,
        platform-specific, or composite)
    """
    try:
        if context and platform:
            composite_fwd = f"fwd_with_residuals_{context}_{platform}"
            composite_vjp = f"vjp_{context}_{platform}"

            has_composite_fwd = hasattr(type(k), composite_fwd) and getattr(type(k), composite_fwd) is not getattr(
                Kernel, composite_fwd, None
            )
            has_composite_vjp = hasattr(type(k), composite_vjp) and getattr(type(k), composite_vjp) is not getattr(
                Kernel, composite_vjp, None
            )

            if has_composite_fwd and has_composite_vjp:
                return True

        if context:
            context_fwd = f"fwd_with_residuals_{context}"
            context_vjp = f"vjp_{context}"

            has_context_fwd = hasattr(type(k), context_fwd) and getattr(type(k), context_fwd) is not getattr(
                Kernel, context_fwd, None
            )
            has_context_vjp = hasattr(type(k), context_vjp) and getattr(type(k), context_vjp) is not getattr(
                Kernel, context_vjp, None
            )

            if has_context_fwd and has_context_vjp:
                return True

        if platform:
            platform_fwd = f"fwd_with_residuals_{platform}"
            platform_vjp = f"vjp_{platform}"

            has_platform_fwd = hasattr(type(k), platform_fwd) and getattr(type(k), platform_fwd) is not getattr(
                Kernel, platform_fwd, None
            )
            has_platform_vjp = hasattr(type(k), platform_vjp) and getattr(type(k), platform_vjp) is not getattr(
                Kernel, platform_vjp, None
            )

            if has_platform_fwd and has_platform_vjp:
                return True

        return type(k).fwd_with_residuals is not Kernel.fwd_with_residuals and type(k).vjp is not Kernel.vjp
    except AttributeError:
        return False


def _get_platform_method(
    k: Kernel,
    method_name: str,
    platform: str | None = None,
    context: str | None = None,
) -> Callable | None:
    """Get context and platform-specific method from kernel, with fallback hierarchy.

    Supports execution contexts (like 'shard_map') combined with hardware platforms
    (like 'gpu', 'tpu', 'cpu'). The lookup follows this priority:
    1. {method}_{context}_{platform} (e.g., run_shard_map_gpu)
    2. {method}_{context} (e.g., run_shard_map)
    3. {method}_{platform} (e.g., run_gpu)
    4. {method} (e.g., run)

    Args:
        k: Kernel instance
        method_name: Base method name (e.g., 'run', 'candidate_cfgs', 'fwd_with_residuals')
        platform: Optional platform identifier (e.g., 'gpu', 'tpu', 'cpu')
        context: Optional execution context (e.g., 'shard_map')

    Returns:
        Most specific available method, or None if no override exists

    Example:
        >>>
        >>> method = _get_platform_method(kernel, 'run', platform='gpu', context='shard_map')
    """

    if context and platform:
        name = f"{method_name}_{context}_{platform}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if context:
        name = f"{method_name}_{context}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if platform:
        name = f"{method_name}_{platform}"
        if hasattr(k, name):
            method = getattr(k, name)
            base = getattr(Kernel, name, None)
            if method is not base:
                return method

    if hasattr(k, method_name):
        method = getattr(k, method_name)
        base = getattr(Kernel, method_name, None)
        if method is not base:
            return method
    return None
