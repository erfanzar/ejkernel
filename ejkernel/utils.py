# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import functools
import typing as tp
from typing import Literal, overload

import jax
import numpy
import numpy as np
import triton
from eformer.loggings import get_logger
from jax import Array
from jax import numpy as jnp

logger = get_logger("ejgpu-utils")

F = tp.TypeVar("F", bound=tp.Callable[..., tp.Any])

DEBUG_GLOBAL_RNG = None

CDNA_ARCHS = ["gfx940", "gfx941", "gfx942", "gfx90a", "gfx908"]
RDNA_ARCHS = ["gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"]
Layouts: tp.TypeAlias = Literal["bhsd", "bshd", "thd"]


@overload
def cdiv(a: int, b: int) -> int: ...


@overload
def cdiv(a: int, b: jax.Array) -> jax.Array: ...


@overload
def cdiv(a: jax.Array, b: int) -> jax.Array: ...


@overload
def cdiv(a: jax.Array, b: jax.Array) -> jax.Array: ...


def cdiv(a: int | jax.Array, b: int | jax.Array) -> int | jax.Array:
    """Ceiling division operation.

    Computes the ceiling division of a by b, which is equivalent to (a + b - 1) // b.

    Args:
            a: Dividend, can be an integer or a JAX array.
            b: Divisor, can be an integer or a JAX array.

    Returns:
            The ceiling division result with the same type as inputs.
    """
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    return jax.lax.div(a + b - 1, b)


def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Calculate the strides for a contiguous array with the given shape.

    Args:
            shape: A tuple of integers representing the dimensions of an array.

    Returns:
            A tuple of integers representing the strides of a contiguous array.
    """
    size = np.prod(shape)
    strides = []
    for s in shape:
        size = size // s
        strides.append(int(size))
    return tuple(strides)


def get_stride(shape: tuple[int, ...] | jax.Array, index=0) -> int:
    return get_strides(shape)[index]


def next_power_of_2(x: int) -> int:
    """Returns the next power of two greater than or equal to `x`.

    Args:
            x: A non-negative integer.

    Returns:
            The smallest power of 2 greater than or equal to x.

    Raises:
            ValueError: If x is negative.
    """
    if x < 0:
        raise ValueError("`next_power_of_2` requires a non-negative integer.")
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def safe_autotune(
    configs,
    key,
    prune_configs_by=None,
    reset_to_zero=None,
    restore_value=None,
    pre_hook=None,
    post_hook=None,
    warmup=None,
    rep=None,
    use_cuda_graph=False,
    do_bench=None,
) -> tp.Callable[[F], F]:
    """
    Applies `triton.autotune` safely. Falls back to the original function if autotuning fails.
    """
    try:
        from triton.runtime.autotuner import Autotuner

        def decorator(fn):
            try:
                return Autotuner(
                    fn,
                    fn.arg_names,
                    configs,
                    key,
                    reset_to_zero,
                    restore_value,
                    pre_hook=pre_hook,
                    post_hook=post_hook,
                    prune_configs_by=prune_configs_by,
                    warmup=warmup,
                    rep=rep,
                    use_cuda_graph=use_cuda_graph,
                )
            except Exception:
                return fn

        return decorator
    except (Exception, RuntimeError) as err:
        print(f"Couldn't autotune given function due to {err}")

        def decorator(fn):
            return fn

        return decorator


def dtype_index(x: jnp.array) -> int:
    if x.dtype == jnp.float16:
        return 1
    if x.dtype == jnp.bfloat16:
        return 2
    if x.dtype == jnp.float32:
        return 3
    raise ValueError(x.dtype)


def get_sharding(arr: jax.Array):
    """Gets the sharding of an array.

    Args:
            arr: Array to get sharding from.

    Returns:
            Sharding of the array.
    """
    return getattr(arr, "sharding", None)


def get_strides(shape: tuple[int, ...] | jax.Array) -> tuple[int, ...]:
    """Calculates strides for a given shape.

    Args:
            shape: Shape of the array.

    Returns:
            Tuple of strides.
    """
    if hasattr(shape, "shape"):
        shape = shape.shape
    size = numpy.prod(shape)
    strides = []
    for s in shape:
        size = int(size // s)
        strides.append(size)
    return tuple(strides)


def get_padded_headsize(size):
    padded_d_model = 1 << (size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def kw_strides(x: Array | None, *stride_names: str):
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": get_stride(x, i) for i, s in enumerate(stride_names)}


def narrow(x, dim: int, start: int, length: int):
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, start + length)
    return x[tuple(slices)]


def get_input_shapes():
    cases = [(max(1, 2 ** (16 - i)), 1, 2**i, 16, 1, 128) for i in range(8, 18)] + [
        (max(1, 2 ** (16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)
    ]
    return cases


@functools.cache
def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


@functools.cache
def is_cdna():
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in CDNA_ARCHS
    except Exception:
        return False


@functools.cache
def is_rdna():
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in RDNA_ARCHS
    except Exception:
        return False


def calculate_blocksize_and_wraps(n):
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError()
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


def numeric_gen(*shape, dtype: str | jnp.dtype = jnp.float16, method: str = "normal"):
    global DEBUG_GLOBAL_RNG
    if DEBUG_GLOBAL_RNG is None:
        DEBUG_GLOBAL_RNG = jax.random.PRNGKey(0)
    DEBUG_GLOBAL_RNG, key = jax.random.split(DEBUG_GLOBAL_RNG, 2)
    method = getattr(jax.random, method, None)
    assert method is not None, "unsupported method in `jax.random`."
    return method(key=key, shape=shape, dtype=dtype)


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    logger.info(msg)
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    if warning or (error_rate < 0.01 or abs_atol <= 0.3):
        if error_rate > ratio:
            import warnings

            warnings.warn(msg, stacklevel=1)
    else:
        assert error_rate < ratio, msg


def is_fp8(x):
    if x.dtype in {jnp.float8_e4m3fnuz, jnp.float8_e4m3fn, jnp.float8_e5m2, jnp.float8_e5m2fnuz}:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device does not support fp8")
    else:
        return False


@functools.cache
def get_gpu_arch() -> str:
    """Get current GPU architecture."""
    try:
        return triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return ""


def arch_supports_fp8():
    return is_hip() and get_gpu_arch() in ("gfx942")
