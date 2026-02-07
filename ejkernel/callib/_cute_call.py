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

"""CuTe DSL kernel integration helpers for JAX.

The wrapper mirrors the placement checks used by :mod:`ejkernel.callib._triton_call`.
It enforces that all array arguments are on one device, and when multiple
accelerators are present it requires execution under ``jax.shard_map``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util
from jax._src import core

from ._utils import ShapeDtype

CAN_USE_CUTE = False
try:
    import cutlass  # noqa: F401
    import cutlass.cute as cute  # noqa: F401

    CAN_USE_CUTE = True
except ModuleNotFoundError:
    pass

_NAMED_CALL_CACHE: dict[tuple[Any, ...], Callable[..., Any]] = {}
_NAMED_CALL_CACHE_LOCK = threading.Lock()


def _device_set_from_sharding(sharding: Any) -> set | None:
    """Extract the participating devices from a sharding object."""
    for attr_name in ("device_set", "devices"):
        try:
            attr = getattr(sharding, attr_name, None)
        except Exception:
            continue
        if attr is None:
            continue
        try:
            devices = attr() if callable(attr) else attr
        except Exception:
            continue
        if devices is not None:
            return set(devices)
    return None


def _array_device_set(arg: Any) -> set | None:
    """Return the device set for a JAX array or tracer."""
    if isinstance(arg, jax.Array):
        device_attr = getattr(arg, "device", None)
        if device_attr is not None:
            try:
                device = device_attr() if callable(device_attr) else device_attr
            except Exception:
                device = None
            if device is not None:
                return {device}

        devices_attr = getattr(arg, "devices", None)
        if devices_attr is not None:
            try:
                devices = devices_attr() if callable(devices_attr) else devices_attr
            except Exception:
                devices = None
            if devices is not None:
                return set(devices)

        sharding = getattr(arg, "sharding", None)
        if sharding is not None:
            device_set = _device_set_from_sharding(sharding)
            if device_set is not None:
                return device_set

    if isinstance(arg, core.Tracer):
        aval = getattr(arg, "aval", None)
        sharding = getattr(aval, "sharding", None)
        if sharding is not None:
            device_set = _device_set_from_sharding(sharding)
            if device_set is not None:
                return device_set

    return None


def _assert_single_device_args(
    array_args: Sequence[Any],
    device_index: int | None,
    *,
    allow_sharded_tracers: bool,
) -> None:
    """Validate that array arguments are placed on one logical device."""
    device_sets: list[tuple[int, set, bool]] = []
    for idx, arg in enumerate(array_args):
        devs = _array_device_set(arg)
        if devs is not None:
            device_sets.append((idx, devs, isinstance(arg, core.Tracer)))

    if not device_sets:
        return

    for idx, devs, is_tracer in device_sets:
        if len(devs) != 1:
            if allow_sharded_tracers and is_tracer:
                continue
            raise AssertionError(
                "cute_call requires all array arguments to be on a single device. "
                f"Argument {idx} is sharded across {len(devs)} devices. "
                "Use `jax.shard_map` for multi-device execution."
            )

    single_device_sets = [(idx, devs) for idx, devs, _ in device_sets if len(devs) == 1]
    if not single_device_sets:
        return

    first_device = next(iter(single_device_sets[0][1]))
    for idx, devs in single_device_sets[1:]:
        if next(iter(devs)) != first_device:
            raise AssertionError(
                "cute_call requires all array arguments to be on the same device. "
                f"Argument {idx} is on a different device than argument {single_device_sets[0][0]}."
            )

    if device_index is None:
        return

    try:
        platform = getattr(first_device, "platform", None)
        devices = jax.devices(platform) if platform else jax.devices()
        if 0 <= device_index < len(devices) and devices[device_index] != first_device:
            raise AssertionError(
                "cute_call received inputs on a different device than the requested "
                f"`device={device_index}`. Place inputs on the target device or adjust "
                "the `device` argument."
            )
    except Exception:
        return


def _has_multi_accelerators() -> bool:
    """Check whether more than one non-CPU accelerator is available."""
    try:
        devices = jax.devices()
    except Exception:
        return False
    accelerator_devices = [
        device for device in devices if getattr(device, "platform", None) not in (None, "cpu")
    ]
    return len(accelerator_devices) > 1


def _in_shard_map_context() -> bool:
    """Check whether execution is currently inside a ``jax.shard_map`` context."""
    try:
        from jax._src import mesh as mesh_lib

        env = getattr(mesh_lib, "thread_resources", None)
        env = getattr(env, "env", None)
        physical_mesh = getattr(env, "physical_mesh", None)
        axis_names = getattr(physical_mesh, "axis_names", None)
        if axis_names:
            return True
    except Exception:
        pass

    try:
        axis_env = core.thread_local_state.trace_state.axis_env
        axis_names = getattr(axis_env, "names", None)
        if axis_names:
            return True
    except Exception:
        pass

    return False


def _leaf_shape_dtype(leaf: Any) -> tuple[tuple[int, ...] | None, jnp.dtype | None]:
    """Read ``(shape, dtype)`` from an output leaf or tracer."""
    shape = getattr(leaf, "shape", None)
    dtype = getattr(leaf, "dtype", None)
    if shape is not None and dtype is not None:
        return tuple(shape), jnp.dtype(dtype)

    aval = getattr(leaf, "aval", None)
    shape = getattr(aval, "shape", None)
    dtype = getattr(aval, "dtype", None)
    if shape is not None and dtype is not None:
        return tuple(shape), jnp.dtype(dtype)

    return None, None


def _validate_out_leaves(
    flat_out: Sequence[Any],
    flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None,
) -> None:
    """Validate explicit output leaves and optional output shape contracts."""
    if not flat_out:
        raise ValueError("`out` must contain at least one output array.")

    for i, leaf in enumerate(flat_out):
        if not isinstance(leaf, (jax.Array, core.Tracer)):
            raise TypeError(
                "`out` leaves must be JAX arrays/tracers. "
                f"Got type {type(leaf)!r} at output index {i}."
            )
        shape, dtype = _leaf_shape_dtype(leaf)
        if shape is None or dtype is None:
            raise TypeError(
                "Could not infer shape/dtype from an `out` leaf. "
                f"Output index: {i}."
            )

    if flat_out_shapes is None:
        return

    if len(flat_out) != len(flat_out_shapes):
        raise ValueError(
            "Mismatch between number of output leaves and `out_shape` leaves: "
            f"{len(flat_out)} vs {len(flat_out_shapes)}."
        )

    for i, (leaf, spec) in enumerate(zip(flat_out, flat_out_shapes, strict=False)):
        leaf_shape, leaf_dtype = _leaf_shape_dtype(leaf)
        if leaf_shape != tuple(spec.shape):
            raise ValueError(
                "Output shape mismatch at index "
                f"{i}: out has shape {leaf_shape}, expected {tuple(spec.shape)}."
            )
        if leaf_dtype != jnp.dtype(spec.dtype):
            raise ValueError(
                "Output dtype mismatch at index "
                f"{i}: out has dtype {leaf_dtype}, expected {jnp.dtype(spec.dtype)}."
            )


def _shape_specs_from_out_leaves(flat_out: Sequence[Any]) -> list[jax.ShapeDtypeStruct]:
    """Build shape/dtype structs from explicit ``out`` leaves."""
    specs: list[jax.ShapeDtypeStruct] = []
    for i, leaf in enumerate(flat_out):
        shape, dtype = _leaf_shape_dtype(leaf)
        if shape is None or dtype is None:
            raise TypeError(
                "Could not infer shape/dtype from an `out` leaf. "
                f"Output index: {i}."
            )
        specs.append(jax.ShapeDtypeStruct(shape, dtype))
    return specs


def _shape_key(shape: Any) -> tuple[str, ...]:
    """Normalize shape values to a stable, hashable key."""
    return tuple(str(d) for d in tuple(shape))


def _arg_contract_key(arg: Any) -> tuple[Any, ...]:
    """Build a cache-key fragment for an argument."""
    if isinstance(arg, (jax.Array, core.Tracer)):
        shape, dtype = _leaf_shape_dtype(arg)
        return ("array", _shape_key(shape or ()), str(dtype))
    if isinstance(arg, (bool, int, float, str, bytes)):
        return ("scalar", type(arg).__name__, arg)
    if isinstance(arg, jnp.dtype):
        return ("dtype", str(arg))
    return ("object", type(arg).__module__, type(arg).__qualname__)


def _out_contract_key(output_contract_shapes: Sequence[jax.ShapeDtypeStruct] | None) -> tuple[Any, ...]:
    """Build a stable key for expected output contracts."""
    if output_contract_shapes is None:
        return ()
    return tuple((_shape_key(spec.shape), str(jnp.dtype(spec.dtype))) for spec in output_contract_shapes)


def cute_call(
    *args: Any,
    call: Callable[..., Any] | None = None,
    out_shape: ShapeDtype | Sequence[ShapeDtype] | None = None,
    out: Any | None = None,
    name: str | None = None,
    device: int | None = None,
) -> Any:
    """Execute a CuTe DSL kernel and return its output(s).

    The callable is expected to return output arrays directly. Callers can pass
    ``out_shape`` or ``out`` to define/validate the expected output contract.

    The provided ``call`` must be primitive-backed and return output arrays.
    ``out`` is accepted for compatibility and treated as output metadata
    contract (shape/dtype/tree), not as a destination buffer.
    """
    if not CAN_USE_CUTE:
        raise ValueError("`cute_call` is only available when CUTLASS CuTe is installed.")
    if call is None:
        raise ValueError(
            "Provide `call` to `cute_call`. "
            "For JIT/tracing, pass a CuTe primitive callable "
            "(e.g. from `build_cute_ffi_call` using TVM-FFI)."
        )

    flat_args, _ = tree_util.tree_flatten(args)
    flat_out: list[Any] = []
    out_tree = None
    flat_out_shapes: Sequence[jax.ShapeDtypeStruct] | None = None

    if out_shape is not None:
        out_shape = tree_util.tree_map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), out_shape)
        flat_out_shapes, out_tree_from_shape = tree_util.tree_flatten(out_shape)
    else:
        out_tree_from_shape = None

    if out is not None:
        flat_out, out_tree = tree_util.tree_flatten(out)
        _validate_out_leaves(flat_out, flat_out_shapes)
    else:
        if flat_out_shapes is None:
            raise ValueError("Provide either `out` or `out_shape` to `cute_call`.")
        out_tree = out_tree_from_shape

    output_contract_shapes = list(flat_out_shapes) if flat_out_shapes is not None else None
    if output_contract_shapes is None and flat_out:
        output_contract_shapes = _shape_specs_from_out_leaves(flat_out)

    array_args = [arg for arg in flat_args if isinstance(arg, (jax.Array, core.Tracer))]

    def _coerce_function_output(function_out: Any) -> Any:
        if function_out is None:
            raise ValueError(
                "`cute_call` expected `call` to return output arrays. "
                "Pass a primitive-backed callable instead of a runtime launch object."
            )
        flat_function_out, _ = tree_util.tree_flatten(function_out)
        _validate_out_leaves(flat_function_out, output_contract_shapes)

        if out is None:
            return function_out

        if len(flat_function_out) != len(flat_out):
            raise ValueError(
                "Mismatch between callable output leaves and provided `out` leaves: "
                f"{len(flat_function_out)} vs {len(flat_out)}."
            )

        assert out_tree is not None
        return tree_util.tree_unflatten(out_tree, flat_function_out)

    in_shard_map_context = _in_shard_map_context()
    _assert_single_device_args(array_args, device, allow_sharded_tracers=in_shard_map_context)
    if _has_multi_accelerators() and not in_shard_map_context:
        raise AssertionError(
            "Multiple accelerator devices detected. "
            "cute_call must be invoked under `jax.shard_map` in multi-accelerator setups."
        )

    call_to_run = call
    if name is not None:
        call_key = (
            str(name),
            tuple(_arg_contract_key(arg) for arg in flat_args),
            _out_contract_key(output_contract_shapes),
            device,
        )
        with _NAMED_CALL_CACHE_LOCK:
            cached_call = _NAMED_CALL_CACHE.get(call_key)
            if cached_call is None:
                _NAMED_CALL_CACHE[call_key] = call_to_run
            else:
                call_to_run = cached_call

    scope_name = "cute_call" if name is None else str(name)
    with jax.named_scope(scope_name):
        return _coerce_function_output(call_to_run(*args))


__all__ = ["cute_call"]
