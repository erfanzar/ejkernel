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


"""Kernel registry system for managing multi-platform implementations."""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypeVar, overload

import jax

F = TypeVar("F", bound=Callable)


def _normalize_type_string(type_annotation: Any) -> str:
    """Normalize type annotation string for comparison.

    Handles cases where the same type is imported differently:
    - 'jaxtyping.Float' -> 'Float'
    - 'ejkernel.ops.utils.datacarrier.FwdParams' -> 'FwdParams'

    Args:
        type_annotation: The type annotation to normalize

    Returns:
        Normalized string representation of the type
    """
    if type_annotation is inspect._empty:
        return "inspect._empty"

    type_str = str(type_annotation)

    type_str = re.sub(r"<class '(.+)'>", r"\1", type_str)

    type_str = re.sub(r"\bjaxtyping\.", "", type_str)

    type_str = re.sub(r"\bejkernel\.[\w\.]+\.(\w+)", r"\1", type_str)

    return type_str


def _types_are_equivalent(type1: Any, type2: Any) -> bool:
    """Check if two type annotations are equivalent.

    This handles cases where the same type might be imported differently
    in different modules, e.g., 'Float' vs 'jaxtyping.Float'.

    Args:
        type1: First type annotation
        type2: Second type annotation

    Returns:
        True if types are equivalent, False otherwise
    """

    if type1 is inspect._empty and type2 is inspect._empty:
        return True

    if (type1 is inspect._empty) != (type2 is inspect._empty):
        return False

    normalized1 = _normalize_type_string(type1)
    normalized2 = _normalize_type_string(type2)

    return normalized1 == normalized2


class Platform(str, Enum):
    """Supported kernel implementation platforms."""

    TRITON = "triton"
    PALLAS = "pallas"
    CUDA = "cuda"
    XLA = "xla"


class Backend(str, Enum):
    """Target hardware backends for kernel execution."""

    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"
    ANY = "any"


@dataclass(frozen=True)
class KernelSpec:
    """Specification for a registered kernel implementation.

    Attributes:
        platform: The implementation platform (triton, pallas, cuda, xla)
        backend: Target hardware backend (gpu, tpu, cpu, any)
        algorithm: Algorithm name (e.g., 'flash_attention')
        implementation: The actual kernel function
        priority: Selection priority (higher values preferred)
    """

    platform: Platform
    backend: Backend
    algorithm: str
    implementation: Callable
    priority: int = 0


class KernelRegistry:
    """Registry for managing kernel implementations across platforms and backends.

    Supports registering multiple implementations of the same algorithm for different
    platforms and backends, with priority-based selection.

    Example:
        >>> registry = KernelRegistry()
        >>> @registry.register("flash_attention", Platform.TRITON, Backend.GPU)
        ... def flash_attention_triton(q, k, v): ...
        >>>
        >>>
        >>> impl = registry.get("flash_attention", platform="triton", backend="gpu")
    """

    def __init__(self) -> None:
        """Initialize an empty kernel registry."""
        self._registry: dict[str, list[KernelSpec]] = {}

    @overload
    def register(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "xla"],
        backend: Backend | Literal["gpu", "tpu", "cpu", "any"],
        priority: int = 0,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "xla"],
        backend: Backend | Literal["gpu", "tpu", "cpu", "any"],
        priority: int = 0,
    ) -> Callable[[F], F]:
        """Decorator to register a kernel implementation.

        Args:
            algorithm: Name of the algorithm (e.g., 'flash_attention')
            platform: Implementation platform
            backend: Target hardware backend
            priority: Selection priority (default: 0). Higher values are preferred.

        Returns:
            Decorator function that registers the kernel and returns it unchanged

        Example:
            >>> @registry.register("flash_attention", Platform.TRITON, Backend.GPU, priority=10)
            ... def flash_attention_impl(q, k, v):
            ...     return compute_attention(q, k, v)
        """

        def decorator(func: F) -> F:
            key = algorithm.lower()
            if key not in self._registry:
                self._registry[key] = []

            spec = KernelSpec(
                platform=Platform(platform) if isinstance(platform, str) else platform,
                backend=Backend(backend) if isinstance(backend, str) else backend,
                algorithm=algorithm,
                implementation=func,
                priority=priority,
            )
            self._registry[key].append(spec)

            self._registry[key].sort(key=lambda x: x.priority, reverse=True)
            return func

        return decorator

    def get(
        self,
        algorithm: str,
        platform: Platform | Literal["triton", "pallas", "cuda", "xla", "auto"] | None = None,
        backend: Backend | Literal["gpu", "tpu", "cpu", "any"] | None = None,
    ) -> Callable:
        """Retrieve the best matching kernel implementation.

        Searches for implementations matching the specified algorithm, platform,
        and backend. Returns the highest priority match. If backend is not specified,
        any backend will match. Backend.ANY implementations match all backend queries.

        Args:
            algorithm: Algorithm name to look up
            platform: Optional platform filter
            backend: Optional backend filter

        Returns:
            The matching kernel implementation function

        Raises:
            ValueError: If no matching implementation is found

        Example:
            >>> impl = registry.get("flash_attention", platform="triton", backend="gpu")
            >>> result = impl(q, k, v)
        """
        key = algorithm.lower()
        if key not in self._registry:
            raise ValueError(f"No implementation found for algorithm: {algorithm}")

        candidates = self._registry[key]

        if isinstance(platform, str):
            platform = Platform(platform)
        if isinstance(backend, str):
            backend = Backend(backend)

        for spec in candidates:
            if platform is not None and spec.platform != platform:
                continue
            if backend is not None and spec.backend != backend and spec.backend != Backend.ANY:
                continue
            return spec.implementation

        if platform == Platform.XLA:
            return self.get(algorithm=algorithm, platform=platform, backend=Backend.ANY)
        if backend == Backend.ANY:
            return self.get(algorithm=algorithm, platform=platform, backend=jax.default_backend())
        raise ValueError(f"No implementation found for algorithm={algorithm}, platform={platform}, backend={backend}")

    def list_algorithms(self) -> list[str]:
        """List all registered algorithm names.

        Returns:
            Sorted list of algorithm names
        """
        return sorted(self._registry.keys())

    def list_implementations(self, algorithm: str) -> list[KernelSpec]:
        """List all implementations for a given algorithm.

        Args:
            algorithm: Algorithm name to query

        Returns:
            List of KernelSpec objects, sorted by priority (descending)
        """
        key = algorithm.lower()
        return self._registry.get(key, []).copy()

    def validate_signatures(self, algorithm: str | None, verbose: bool = False) -> bool:
        """Validate that all implementations of an algorithm have matching signatures.

        Compares parameter names, order, and defaults across all implementations.
        Issues warnings for any mismatches found.

        Args:
            algorithm: Algorithm name to validate
            verbose: If True, log all parameter signatures before comparison

        Returns:
            True if all signatures match, False otherwise

        Raises:
            ValueError: If algorithm is not registered
        """
        if algorithm is None:
            for algo in self.list_algorithms():
                self.validate_signatures(algo)
            return
        key = algorithm.lower()
        if key not in self._registry:
            raise ValueError(f"No implementation found for algorithm: {algorithm}")

        specs = self._registry[key]
        if len(specs) < 2:
            return True

        reference_spec = specs[0]
        reference_sig = inspect.signature(reference_spec.implementation)
        reference_params = list(reference_sig.parameters.values())

        if verbose:
            print(f"\n{'=' * 80}")
            print(f"Algorithm: {algorithm}")
            print(f"{'=' * 80}")
            for spec in specs:
                sig = inspect.signature(spec.implementation)
                print(f"\n{spec.platform}/{spec.backend} (priority={spec.priority}):")
                print(f"  Signature: {sig}")
                for param_name, param in sig.parameters.items():
                    print(f"    {param_name}:")
                    print(f"      kind: {param.kind.name}")
                    print(f"      default: {param.default}")
                    print(f"      annotation: {param.annotation}")
            print(f"{'=' * 80}\n")

        all_match = True

        for spec in specs[1:]:
            sig = inspect.signature(spec.implementation)
            params = list(sig.parameters.values())

            if len(params) != len(reference_params):
                warnings.warn(
                    f"Signature mismatch for algorithm '{algorithm}':\n"
                    f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                    f"{len(reference_params)} parameters\n"
                    f"  Implementation ({spec.platform}/{spec.backend}): {len(params)} parameters",
                    UserWarning,
                    stacklevel=2,
                )
                all_match = False
                continue

            for ref_param, param in zip(reference_params, params, strict=False):
                if ref_param.name != param.name:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"parameter '{ref_param.name}'\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): parameter '{param.name}'",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if ref_param.kind != param.kind:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): {ref_param.kind.name}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): {param.kind.name}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if ref_param.default != param.default:
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"default={ref_param.default}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): default={param.default}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

                if not _types_are_equivalent(ref_param.annotation, param.annotation):
                    warnings.warn(
                        f"Signature mismatch for algorithm '{algorithm}' parameter '{ref_param.name}':\n"
                        f"  Reference ({reference_spec.platform}/{reference_spec.backend}): "
                        f"type={ref_param.annotation} = {ref_param.default}\n"
                        f"  Implementation ({spec.platform}/{spec.backend}): type={param.annotation} = {param.default}",
                        UserWarning,
                        stacklevel=2,
                    )
                    all_match = False

        return all_match


kernel_registry = KernelRegistry()
