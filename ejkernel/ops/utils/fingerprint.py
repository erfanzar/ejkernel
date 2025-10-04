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

"""Device fingerprinting and object hashing utilities for caching.

This module provides functions for generating stable, deterministic identifiers
for JAX devices, array shapes, and complex Python objects. These identifiers
are essential for the caching system to correctly match configurations across
different runs and environments.

Key Functions:
    device_fingerprint: Generate stable device identifiers
    short_hash: Create short hashes from complex objects
    stable_json: Deterministic JSON serialization
    abstractify: Convert arrays to abstract shape/dtype specs
    sharding_fingerprint: Extract sharding information for caching

The fingerprinting system handles:
    - JAX devices with platform version information
    - Complex nested data structures (PyTrees)
    - Functions and partial functions
    - Dataclasses and Pydantic models
    - JAX arrays with sharding information
    - NumPy arrays and dtypes

These utilities ensure that:
    - Cache keys are consistent across program runs
    - Device-specific optimizations are properly isolated
    - Complex objects can be reliably serialized and hashed
    - Sharding information is preserved for distributed computation

Example Usage:
    >>> device_id = device_fingerprint()  # 'gpu|cuda_12.0'
    >>> config_hash = short_hash(my_config)  # '1a2b3c4d5e6f7g8h'
    >>> abstract_tree = abstractify(data_with_arrays)
    >>> cache_key = default_key_builder_with_sharding(invocation)
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import json
from typing import Any

import jax
import numpy as np
from pydantic import BaseModel


def sharding_fingerprint(x: Any) -> Any:
    """Extract sharding information from a JAX array for fingerprinting.

    Creates a stable representation of how an array is sharded across devices,
    which is essential for device-aware caching in distributed computation.

    Args:
        x: Object to extract sharding information from

    Returns:
        String representation of sharding for JAX arrays, None otherwise

    Note:
        The sharding representation is kept stable and compact to ensure
        consistent cache keys across different program executions.
    """
    if isinstance(x, jax.Array):
        try:
            return repr(x.sharding)  # small, stable
        except Exception:
            return None
    return None


def default_key_builder_with_sharding(inv) -> str:
    """Generate cache key that includes sharding information for device-aware caching.

    Creates a comprehensive cache key that incorporates argument shapes, types,
    sharding information, and batch axes to ensure optimal cache matching in
    distributed computation environments.

    Args:
        inv: Function invocation object containing args, kwargs, and batch_axes

    Returns:
        Short hash string representing the complete invocation signature

    Note:
        This key builder is more comprehensive than basic builders as it includes
        sharding information, making it suitable for distributed workloads where
        the same logical operation may have different optimal configurations
        depending on how data is sharded across devices.
    """

    shard_sig = jax.tree.map(sharding_fingerprint, inv.args)  # ignore kwargs by default
    spec = dict(
        args_spec=abstractify(inv.args),
        kwargs_spec=abstractify(dict(inv.kwargs)),
        batch_axes=inv.batch_axes,
        sharding=shard_sig,
    )
    return short_hash(spec)


def device_fingerprint(dev: jax.Device | None = None) -> str:  # type:ignore
    """Generate a stable identifier for a JAX device including platform version.

    Creates a unique, stable identifier for JAX devices that includes both the
    device type and platform version information. This ensures that cached
    configurations are specific to the exact hardware and software environment.

    Args:
        dev: JAX device to fingerprint, uses default device if None

    Returns:
        String identifier like 'gpu|cuda_12.0', 'tpu|v4', or 'cpu|'

    Examples:
        >>> device_fingerprint()  # Using default device
        'gpu|cuda_12.0'
        >>> device_fingerprint(jax.devices('cpu')[0])
        'cpu|'

    Note:
        The format is 'device_kind|platform_version' where platform_version
        may be empty for some device types. This fingerprint is used as a
        key component in cache storage to ensure device-specific optimization.
    """
    d = dev or jax.devices()[0]
    kind = getattr(d, "device_kind", getattr(d, "platform", "unknown"))
    client = getattr(d, "client", None)
    plat_ver = getattr(client, "platform_version", "") if client else ""
    return f"{kind}|{plat_ver}"


def device_kind() -> str:
    """Get the device kind (gpu, cpu, tpu) for the default device.

    Returns a simple string identifier for the type of the default JAX device,
    without platform version information.

    Returns:
        Device kind string: 'gpu', 'cpu', 'tpu', or 'unknown'

    Examples:
        >>> device_kind()
        'gpu'

    Note:
        This is a simplified version of device_fingerprint() that only
        returns the device type without platform version details.
    """
    d = jax.devices()[0]
    return getattr(d, "device_kind", getattr(d, "platform", "unknown"))


def stable_json(obj: Any) -> str:
    """Deterministic JSON serialization that handles JAX/NumPy types and dataclasses.

    Provides stable, deterministic JSON serialization for complex Python objects
    including JAX arrays, NumPy types, dataclasses, Pydantic models, and functions.
    The serialization is designed to produce identical output for equivalent objects
    across different program runs.

    Args:
        obj: Object to serialize (can be arbitrarily nested)

    Returns:
        Deterministic JSON string representation

    Supported Types:
        - Functions and methods (with module, name, and position info)
        - functools.partial objects (with function and bound arguments)
        - Callable objects (with class information)
        - Pydantic models (using model_dump())
        - Dataclasses (using asdict())
        - JAX ShapeDtypeStruct objects
        - NumPy dtypes and scalar types
        - Standard Python types

    Note:
        The JSON output uses sorted keys and compact separators to ensure
        consistent formatting. Function objects are serialized with their
        module, qualified name, and source location for stability.
    """

    def default(o):
        if inspect.isfunction(o) or inspect.ismethod(o):
            mod = getattr(o, "__module__", None)
            qn = getattr(o, "__qualname__", getattr(o, "__name__", "anon"))
            co = getattr(o, "__code__", None)
            pos = (getattr(co, "co_filename", None), getattr(co, "co_firstlineno", None)) if co else None
            return {"__callable__": True, "module": mod, "name": qn, "pos": pos}
        if isinstance(o, functools.partial):
            func = default(o.func)
            kws = tuple(sorted((o.keywords or {}).items()))
            return {"__partial__": True, "func": func, "args": o.args, "kwargs": kws}
        if callable(o):
            cls = o.__class__
            return {"__callable_obj__": f"{cls.__module__}.{cls.__qualname__}"}
        if isinstance(o, BaseModel):
            return o.model_dump()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, jax.ShapeDtypeStruct):
            dtype = getattr(o.dtype, "name", str(o.dtype))
            return {"shape": tuple(int(s) for s in o.shape), "dtype": dtype}
        if isinstance(o, np.dtype):  # dtype
            return o.name
        if isinstance(o, np.integer | np.floating | np.bool_):  # scalars
            return o.item()
        return repr(o)

    return json.dumps(obj, default=default, sort_keys=True, separators=(",", ":"))


def short_hash(obj: Any) -> str:
    """Generate a short (16-character) hash from an object using stable JSON serialization.

    Creates a compact, deterministic hash of any Python object by first
    converting it to stable JSON and then computing a SHA-256 hash.

    Args:
        obj: Object to hash (can be arbitrarily complex)

    Returns:
        16-character hexadecimal hash string

    Examples:
        >>> short_hash({'a': 1, 'b': [2, 3]})
        '1a2b3c4d5e6f7g8h'
        >>> short_hash(MyDataclass(x=1, y=2))
        'a1b2c3d4e5f6g7h8'

    Note:
        Uses SHA-256 internally but truncates to 16 characters for compactness.
        The hash is deterministic across program runs for equivalent objects.
    """
    return hashlib.sha256(stable_json(obj).encode()).hexdigest()[:16]


def abstractify(pytree: Any) -> Any:
    """Convert a PyTree containing arrays to abstract shape/dtype specifications.

    Transforms a nested data structure containing JAX or NumPy arrays into an
    abstract representation using ShapeDtypeStruct objects. This allows for
    consistent hashing and comparison based on array shapes and dtypes rather
    than actual array values.

    Args:
        pytree: Nested data structure potentially containing arrays

    Returns:
        PyTree with same structure but arrays replaced by ShapeDtypeStruct

    Examples:
        >>> import jax.numpy as jnp
        >>> data = {'x': jnp.array([1, 2, 3]), 'y': 'scalar'}
        >>> abstract = abstractify(data)
        >>> # abstract['x'] is now ShapeDtypeStruct(shape=(3,), dtype=int32)
        >>> # abstract['y'] is still 'scalar'

    Note:
        This is essential for creating cache keys based on array structure
        rather than values, allowing the same optimized configuration to be
        reused for arrays with the same shape and dtype but different values.
    """

    def leaf(x):
        """Convert array leaf to ShapeDtypeStruct, leave other types unchanged."""
        if isinstance(x, jax.Array | np.ndarray):
            return jax.ShapeDtypeStruct(np.shape(x), getattr(x, "dtype", None))
        return x

    return jax.tree.map(leaf, pytree)
