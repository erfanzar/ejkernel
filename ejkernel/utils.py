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


"""Utility functions for ejKernel library.

This module provides a comprehensive collection of utility functions for kernel
development, including mathematical operations, array manipulation, hardware
detection, performance testing, and distributed synchronization utilities.

The utilities are designed to support both Triton and JAX-based kernel implementations
with focus on GPU architectures (CDNA, RDNA) and distributed training scenarios.
"""

import functools
import typing as tp
from typing import Literal, overload

import jax
import numpy
import numpy as np
import triton
from jax import Array
from jax import numpy as jnp

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
    """Get the stride at a specific dimension index.

    Args:
        shape: Shape of the array or the array itself.
        index: The dimension index to get the stride for. Defaults to 0.

    Returns:
        The stride value at the specified index.
    """
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
) -> tp.Callable[[F], F]:
    """Safely apply Triton autotuning with fallback on failure.

    Wraps a function with Triton's autotuning capability, gracefully falling back
    to the original function if autotuning fails. This ensures kernel execution
    continues even if autotuning encounters issues.

    Args:
        configs: List of triton.Config objects to test during autotuning.
        key: List of argument names whose values define the autotuning key.
        prune_configs_by: Optional dict mapping metric names to pruning functions.
        reset_to_zero: List of argument names to reset to zero between runs.
        restore_value: List of argument names to restore after autotuning.
        pre_hook: Optional function to call before each autotuning run.
        post_hook: Optional function to call after each autotuning run.
        warmup: Number of warmup runs before measuring performance.
        rep: Number of repetitions for each configuration.

    Returns:
        A decorator that applies autotuning to the wrapped function.

    Example:
        >>> @safe_autotune(
        ...     configs=[triton.Config({'BLOCK_SIZE': 128})],
        ...     key=['n_elements']
        ... )
        ... def kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        ...     pass
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
    """Get numeric index for array dtype.

    Maps JAX array dtypes to numeric indices for use in kernel dispatch
    and configuration.

    Args:
        x: JAX array whose dtype to index.

    Returns:
        Integer index corresponding to the dtype:
            1 for float16
            2 for bfloat16
            3 for float32

    Raises:
        ValueError: If the dtype is not supported.
    """
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
    """Calculate padded head size for optimal memory alignment.

    Rounds up the head size to the next power of 2 with a minimum of 16
    for better memory access patterns in attention kernels.

    Args:
        size: Original head size.

    Returns:
        Padded head size as the next power of 2, minimum 16.

    Example:
        >>> get_padded_headsize(13)
        >>> get_padded_headsize(20)
    """
    padded_d_model = 1 << (size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    return padded_d_model


def kw_strides(x: Array | None, *stride_names: str):
    """Generate stride keyword arguments for kernel calls.

    Creates a dictionary mapping stride names to their corresponding values
    for use as keyword arguments in kernel invocations.

    Args:
        x: JAX array to get strides from, or None for zero strides.
        *stride_names: Names for each dimension's stride.

    Returns:
        Dictionary mapping "stride_{name}" to stride values.

    Example:
        >>> arr = jnp.ones((2, 3, 4))
        >>> kw_strides(arr, 'batch', 'seq', 'head')
        {'stride_batch': 12, 'stride_seq': 4, 'stride_head': 1}
    """
    if x is None:
        return {f"stride_{s}": 0 for i, s in enumerate(stride_names)}

    assert x.ndim == len(stride_names)
    return {f"stride_{s}": get_stride(x, i) for i, s in enumerate(stride_names)}


def narrow(x, dim: int, start: int, length: int):
    """Narrow a tensor along a specific dimension.

    Extracts a contiguous slice of length `length` starting at `start`
    along the specified dimension, similar to PyTorch's narrow operation.

    Args:
        x: Input array to narrow.
        dim: Dimension along which to narrow.
        start: Starting index of the slice.
        length: Length of the slice to extract.

    Returns:
        Narrowed array with reduced size along the specified dimension.

    Example:
        >>> x = jnp.arange(20).reshape(4, 5)
        >>> narrow(x, dim=1, start=1, length=3).shape
        (4, 3)
    """
    slices = [slice(None)] * x.ndim
    slices[dim] = slice(start, start + length)
    return x[tuple(slices)]


def get_input_shapes():
    """Generate test input shapes for benchmarking and testing.

    Creates a list of input shape configurations with varying batch sizes
    and sequence lengths for comprehensive kernel testing.

    Returns:
        List of tuples containing (batch, ?, seq_len, ?, ?, ?) dimensions
        for testing various input configurations.
    """
    cases = [(max(1, 2 ** (16 - i)), 1, 2**i, 16, 1, 128) for i in range(8, 18)] + [
        (max(1, 2 ** (16 - i)), 1, 2**i, 16, 2, 128) for i in range(8, 18)
    ]
    return cases


@functools.cache
def is_hip():
    """Check if running on AMD HIP backend.

    Returns:
        True if the current Triton target uses HIP backend, False otherwise.
    """
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


@functools.cache
def is_cdna():
    """Check if running on AMD CDNA architecture.

    CDNA (Compute DNA) architectures include MI100, MI200 series GPUs.

    Returns:
        True if running on CDNA architecture (gfx940, gfx941, etc.), False otherwise.
    """
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in CDNA_ARCHS
    except Exception:
        return False


@functools.cache
def is_rdna():
    """Check if running on AMD RDNA architecture.

    RDNA (Radeon DNA) architectures include RX 6000, 7000 series GPUs.

    Returns:
        True if running on RDNA architecture (gfx1030, gfx1100, etc.), False otherwise.
    """
    try:
        return is_hip() and triton.runtime.driver.active.get_current_target().arch in RDNA_ARCHS
    except Exception:
        return False


def calculate_blocksize_and_wraps(n):
    """Calculate optimal block size and number of warps for Triton kernels.

    Determines the appropriate block size (as power of 2) and number of warps
    based on the input size, with architecture-specific adjustments for HIP.

    Args:
        n: Input size to calculate block configuration for.

    Returns:
        Tuple of (block_size, num_warps) optimized for the input size.

    Raises:
        RuntimeError: If the required block size exceeds MAX_FUSED_SIZE (65536).

    Example:
        >>> calculate_blocksize_and_wraps(1024)
        (1024, 4)
        >>> calculate_blocksize_and_wraps(10000)
        (16384, 16)
    """
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
    """Generate random numeric arrays for testing and debugging.

    Creates random arrays using JAX's random number generation with a global
    debug RNG state for reproducibility.

    Args:
        *shape: Dimensions of the array to generate.
        dtype: Data type of the generated array. Defaults to float16.
        method: Random generation method from jax.random. Defaults to "normal".

    Returns:
        Random JAX array with specified shape and dtype.

    Raises:
        AssertionError: If the specified method is not available in jax.random.

    Example:
        >>> arr = numeric_gen(2, 3, 4, dtype=jnp.float32, method="uniform")
        >>> arr.shape
        (2, 3, 4)
    """
    global DEBUG_GLOBAL_RNG
    if DEBUG_GLOBAL_RNG is None:
        DEBUG_GLOBAL_RNG = jax.random.PRNGKey(0)
    DEBUG_GLOBAL_RNG, key = jax.random.split(DEBUG_GLOBAL_RNG, 2)
    method = getattr(jax.random, method, None)
    assert method is not None, "unsupported method in `jax.random`."
    return method(key=key, shape=shape, dtype=dtype)


def random_dense(
    *shape,
    dtype: str | jnp.dtype = jnp.float16,
    limit: int | None = 1,
) -> jnp.ndarray:
    global DEBUG_GLOBAL_RNG
    if DEBUG_GLOBAL_RNG is None:
        DEBUG_GLOBAL_RNG = jax.random.PRNGKey(0)
    DEBUG_GLOBAL_RNG, key = jax.random.split(DEBUG_GLOBAL_RNG, 2)
    if limit is None:
        limit = 1 / np.prod(shape)
    x = jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)
    return x.astype(jnp.bfloat16).astype(dtype)


def get_abs_err(x, y):
    """Calculate maximum absolute error between two arrays.

    Args:
        x: First array.
        y: Second array.

    Returns:
        Maximum absolute difference between the arrays.
    """
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    """Calculate relative error ratio between two arrays.

    Computes the root mean square error normalized by the RMS of the reference.

    Args:
        x: Reference array.
        y: Array to compare against reference.

    Returns:
        Relative error ratio (RMSE / RMS of reference).
    """
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    """Assert that two arrays are close within tolerance.

    Compares arrays using both absolute and relative error thresholds,
    with options for warnings vs assertions.

    Args:
        prefix: Message prefix for error reporting.
        ref: Reference array for comparison.
        tri: Array to test against reference.
        ratio: Maximum allowed error ratio.
        warning: If True, issue warning instead of assertion on failure.
        err_atol: Absolute tolerance threshold. Defaults to 1e-6.

    Raises:
        AssertionError: If arrays differ beyond tolerance and warning=False.

    Example:
        >>> ref = jnp.ones((10,))
        >>> test = ref + 1e-7
        >>> assert_close("Test", ref, test, ratio=0.01)
    """
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    print(msg)
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
    """Check if an array uses FP8 dtype and if hardware supports it.

    Args:
        x: Array to check for FP8 dtype.

    Returns:
        True if array is FP8 and hardware supports it, False if not FP8.

    Raises:
        RuntimeError: If array is FP8 but hardware doesn't support it.
    """
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
    """Check if current GPU architecture supports FP8 operations.

    Returns:
        True if running on AMD gfx942 architecture with FP8 support, False otherwise.
    """
    return is_hip() and get_gpu_arch() in ("gfx942")


def generate_block_indices(
    batch: int,
    num_query_blocks: int,
    heads: int,
    selected_blocks: int,
    block_size: int,
    seed: int = 42,
) -> jax.Array:
    """Generate random block indices for sparse attention benchmarks.

    This function generates a tensor of block indices where each token attends to
    a random selection of previous key blocks. The indices are sorted in ascending order.
    Returns per-token format: each token in a query block gets the same block indices.

    Args:
        batch: Batch size.
        num_query_blocks: Number of query blocks.
        heads: Number of attention heads (typically kv_heads for GQA).
        selected_blocks: Number of key blocks each query block should attend to.
        block_size: Size of each block.
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (batch, seq_len, heads, selected_blocks) containing sorted
        block indices in per-token format. Positions beyond available blocks are filled with -1.

    Example:
        >>>
        >>> indices = generate_block_indices(batch=2, num_query_blocks=4, heads=8, selected_blocks=2, block_size=64)
        >>> indices.shape
        (2, 256, 8, 2)
    """
    seq_len = num_query_blocks * block_size
    block_indices = jnp.full((batch, seq_len, heads, selected_blocks), -1, dtype=jnp.int32)

    key = jax.random.PRNGKey(seed)

    for b in range(batch):
        for qb in range(num_query_blocks):
            for h in range(heads):
                key, subkey = jax.random.split(key)

                num_available_blocks = qb + 1

                perm = jax.random.permutation(subkey, num_available_blocks)[:selected_blocks]

                if num_available_blocks < selected_blocks:
                    perm = jnp.pad(perm, (0, selected_blocks - num_available_blocks), constant_values=-1)

                perm_sorted = jnp.sort(perm)

                token_start = qb * block_size
                token_end = (qb + 1) * block_size
                for t in range(token_start, token_end):
                    block_indices = block_indices.at[b, t, h, :].set(perm_sorted)

    return block_indices


_sync_counter = 0


def barrier_sync(timeout: float = 200):
    """Synchronize all JAX processes at a barrier point.

    Blocks execution until all processes in the distributed JAX runtime reach
    this barrier. This is essential for ensuring consistency across distributed
    training, especially before/after collective operations or checkpointing.

    The function uses a global counter to create unique barrier names, allowing
    multiple barriers to be used sequentially without conflicts.

    Args:
        timeout: Maximum time to wait for all processes to reach the barrier,
            in seconds. Defaults to 200 seconds (3.33 minutes). If the timeout
            is exceeded, a RuntimeError will be raised by the underlying JAX
            distributed client.

    Returns:
        None

    Raises:
        RuntimeError: If the JAX distributed client is not initialized. This
            typically means JAX was not started in distributed mode or the
            distributed runtime failed to initialize.

    Note:
        - This function is a no-op when running with a single process
          (jax.process_count() == 1), allowing code to work seamlessly
          in both single and multi-process environments.
        - Each call increments a global counter to ensure unique barrier names,
          preventing conflicts when multiple barriers are used in sequence.
        - The timeout is converted to milliseconds for the underlying JAX API.

    Example:
        >>>
        >>> model = train_step(model, batch)
        >>> barrier_sync()
        >>> if jax.process_index() == 0:
        ...     save_checkpoint(model)
        >>> barrier_sync()

        >>>
        >>> barrier_sync(timeout=600)

    Warning:
        Ensure all processes call barrier_sync() the same number of times and
        in the same order, or deadlocks may occur. Conditional barriers based
        on process rank should be avoided.
    """
    global _sync_counter
    if jax.process_count() == 1:
        return
    import jax._src.distributed as distributed

    client = distributed.global_state.client

    if client is None:
        raise RuntimeError("barrier_sync requires jax distributed client to be initialized")

    _sync_counter += 1
    client.wait_at_barrier(f"easy_barrier_sync_{_sync_counter}", timeout_in_ms=int(timeout * 1000.0))
