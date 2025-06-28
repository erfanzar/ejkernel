# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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

import typing as tp
from typing import Literal, overload

import jax
import numpy
import numpy as np
import triton
from jax import Array
from jax import numpy as jnp
from jax import random as jrnd

from .logging_utils import get_logger

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
    if isinstance(shape, jax.Array):
        shape = shape.shape
    return strides_from_shape(shape=shape)[index]


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


class AttentionMetadata:
    cu_seqlens_q: list[int] | None = None
    cu_seqlens_k: list[int] | None = None
    max_seqlens_q: int = 0
    max_seqlens_k: int = 0
    bias: Array | None = None
    alibi_slopes: Array | None = None
    causal: bool = False
    num_contexts: int = 0
    varlen: bool = False
    layout: Layouts | None = None
    cache_seqlens: list[int] | None = None
    cache_batch_idx: list[int] | None = None
    new_kv: bool = False
    seqlen_new: int | None = None
    k_new: Array | None = None
    v_new: Array | None = None
    dropout_p: float = 0.0

    def __repr__(self) -> str:
        """Return a string representation of the metadata."""
        return (
            f"AttentionMetadata(\n"
            f"  sm_scale={self.sm_scale},\n"
            f"  cu_seqlens_q={self.cu_seqlens_q},\n"
            f"  cu_seqlens_k={self.cu_seqlens_k},\n"
            f"  max_seqlens_q={self.max_seqlens_q},\n"
            f"  max_seqlens_k={self.max_seqlens_k},\n"
            f"  bias={self.bias},\n"
            f"  alibi_slopes={self.alibi_slopes},\n"
            f"  causal={self.causal},\n"
            f"  num_contexts={self.num_contexts},\n"
            f"  varlen={self.varlen},\n"
            f"  layout={self.layout},\n"
            f"  cache_seqlens={self.cache_seqlens},\n"
            f"  cache_batch_idx={self.cache_batch_idx},\n"
            f"  new_kv={self.new_kv},\n"
            f"  seqlen_new={self.seqlen_new},\n"
            f"  k_new={self.k_new},\n"
            f"  v_new={self.v_new},\n"
            f"  dropout_p={self.dropout_p},\n"
            f")"
        )

    def __init__(self, sm_scale: float = 1.0):
        """Initialize attention metadata with scale factor."""
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q: list[int] | Array, cu_seqlens_k: list[int] | Array) -> None:
        """Configure metadata for variable-length sequences."""
        self.varlen = True
        self.layout = "thd"
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias: Array, batch: int, nheads: int, seqlen_q: int, seqlen_k: int) -> None:
        """Configure bias for attention computation."""
        assert bias.is_cuda
        assert bias.ndim == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes: Array, batch: int, nheads: int) -> None:
        """Configure ALiBi slopes for attention computation."""
        assert alibi_slopes.is_cuda
        assert alibi_slopes.ndim == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self) -> None:
        """Enable causal attention pattern."""
        self.causal = True

    def need_dropout(self, dropout_p: float, return_scores: bool) -> None:
        """Configure dropout and score return parameters."""
        self.dropout_p = dropout_p
        self.return_scores = return_scores

    def check_args(self, q: Array, k: Array, v: Array, o: Array) -> None:
        """Validate input tensors against metadata configuration."""
        assert q.ndim == k.ndim and q.ndim == v.ndim
        batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(
            q,
            k,
            self.layout,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.max_seqlens_q,
            self.max_seqlens_k,
        )
        if self.varlen:
            assert q.ndim == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            assert self.bias is None
            assert self.dropout_p == 0.0
        else:
            assert q.ndim == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == "thd" or not self.varlen

    @staticmethod
    def convert_to_torch(q: jax.Array, k: jax.Array, v: jax.Array, host: str = "cuda:0"):
        import torch

        device = jax.devices("cpu")[0]
        q, k, v = jax.device_put(q, device), jax.device_put(k, device), jax.device_put(v, device)
        q, k, v = np.asarray(q.tolist()), np.asarray(k.tolist()), np.asarray(v.tolist())
        return (
            torch.from_numpy(q).to(host).to(torch.float16),
            torch.from_numpy(k).to(host).to(torch.float16),
            torch.from_numpy(v).to(host).to(torch.float16),
        )


def input_helper(
    batch_size: int,
    qheads: int,
    kvheads: int,
    qseqlen: int,
    kseqlen: int,
    head_dim: int,
    dtype: str | jnp.dtype = "f2",
    layout: Literal["bhsd", "bshd"] = "bshd",
    debug_mode: bool = False,
) -> tuple[Array, Array, Array, AttentionMetadata]:
    if layout == "bhsd":
        q_tensor_shape = (batch_size, qheads, qseqlen, head_dim)
        k_tensor_shape = (batch_size, kvheads, kseqlen, head_dim)
    elif layout == "bshd":
        q_tensor_shape = (batch_size, qseqlen, qheads, head_dim)
        k_tensor_shape = (batch_size, kseqlen, kvheads, head_dim)
    else:
        raise NotImplementedError(f"Got unsupported tensor layout: {layout}")

    if debug_mode:
        if layout == "bhsd":
            q = jnp.broadcast_to(jnp.arange(qseqlen, dtype=dtype).reshape(1, 1, qseqlen, 1), q_tensor_shape)
            k = jnp.broadcast_to(jnp.arange(kseqlen, dtype=dtype).reshape(1, 1, kseqlen, 1), k_tensor_shape)
            v = jnp.broadcast_to(jnp.arange(kseqlen, dtype=dtype).reshape(1, 1, kseqlen, 1), k_tensor_shape)
        elif layout == "bshd":
            q = jnp.broadcast_to(jnp.arange(qseqlen, dtype=dtype).reshape(1, qseqlen, 1, 1), q_tensor_shape)
            k = jnp.broadcast_to(jnp.arange(kseqlen, dtype=dtype).reshape(1, kseqlen, 1, 1), k_tensor_shape)
            v = jnp.broadcast_to(jnp.arange(kseqlen, dtype=dtype).reshape(1, kseqlen, 1, 1), k_tensor_shape)
    else:
        q = jrnd.normal(jrnd.key(1), q_tensor_shape, dtype=dtype)
        k = jrnd.normal(jrnd.key(2), k_tensor_shape, dtype=dtype)
        v = jrnd.normal(jrnd.key(3), k_tensor_shape, dtype=dtype)

    if debug_mode:
        sm_scale = 1
    else:
        sm_scale = head_dim**-0.5
    input_metadata = AttentionMetadata(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = qseqlen
    input_metadata.max_seqlens_k = kseqlen
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(
    batch_size: int,
    qheads: int,
    kvheads: int,
    qseqlen: int,
    kseqlen: int,
    head_dim: int,
    dtype: str | jnp.dtype = "f2",
    equal_seqlens: bool = False,
    debug_mode: bool = False,
) -> tuple[Array, Array, Array, AttentionMetadata]:
    if not equal_seqlens:
        max_seqlens_q = qseqlen // batch_size
        max_seqlens_k = kseqlen // batch_size
        seqlens_q = jrnd.randint(jrnd.key(0), (batch_size,), 1, max_seqlens_q + 1, dtype="i4")
        seqlens_k = jrnd.randint(jrnd.key(1), (batch_size,), 1, max_seqlens_k + 1, dtype="i4")
    else:
        seqlens_q = jnp.full((batch_size,), qseqlen // batch_size, dtype="i4")
        seqlens_k = jnp.full((batch_size,), kseqlen // batch_size, dtype="i4")

    cu_seqlens_q = jnp.concatenate([jnp.array([0], dtype="i4"), seqlens_q.cumsum(axis=0)])
    cu_seqlens_k = jnp.concatenate([jnp.array([0], dtype="i4"), seqlens_k.cumsum(axis=0)])
    cu_seqlens_q = cu_seqlens_q.astype("i4")
    cu_seqlens_k = cu_seqlens_k.astype("i4")
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()

    if debug_mode:
        q = jnp.arange(total_q, dtype=dtype).reshape(total_q, 1, 1)
        q = jnp.broadcast_to(q, (total_q, qheads, head_dim))
        k = jnp.arange(total_k, dtype=dtype).reshape(total_k, 1, 1)
        k = jnp.broadcast_to(k, (total_k, kvheads, head_dim))
        v = jnp.arange(total_k, dtype=dtype).reshape(total_k, 1, 1)
        v = jnp.broadcast_to(v, (total_k, kvheads, head_dim))
        sm_scale = 1
    else:
        q = jrnd.normal(jrnd.key(5), (total_q, qheads, head_dim), dtype=dtype)
        k = jrnd.normal(jrnd.key(6), (total_k, kvheads, head_dim), dtype=dtype)
        v = jrnd.normal(jrnd.key(7), (total_k, kvheads, head_dim), dtype=dtype)
        sm_scale = head_dim**-0.5

    input_metadata = AttentionMetadata(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


def generate_block_indices(batch_size, sequence_length, kv_heads, num_blocks_per_token, block_size, seed=0):
    """
    Generate sparse attention block indices.
    """
    block_indices = jnp.full(
        (batch_size, sequence_length, kv_heads, num_blocks_per_token),
        sequence_length,
        dtype=jnp.int32,
    )
    key = jax.random.PRNGKey(seed)

    for b in range(batch_size):
        for t in range(sequence_length):
            for h in range(kv_heads):
                key, subkey = jax.random.split(key)
                max_blocks = max(1, cdiv(t, block_size))
                selected_blocks = jax.random.permutation(subkey, max_blocks)[:num_blocks_per_token]
                block_indices = block_indices.at[b, t, h, : len(selected_blocks)].set(selected_blocks)

    return jnp.sort(block_indices, axis=-1)


def get_shape_from_layout(
    q: Array,
    k: Array,
    layout: Layouts,
    cu_seqlens_q: int | list[int] | Array | None = None,
    cu_seqlens_k: int | list[int] | Array | None = None,
    max_seqlen_q: int | list[int] | Array | None = None,
    max_seqlen_k: int | list[int] | Array | None = None,
):
    if layout == "bhsd":
        batch_q, nheads_q, max_seqlen_q, head_size_q = q.shape
        batch_k, nheads_k, max_seqlen_k, head_size_k = k.shape
    elif layout == "bshd":
        batch_q, max_seqlen_q, nheads_q, head_size_q = q.shape
        batch_k, max_seqlen_k, nheads_k, head_size_k = k.shape
    elif layout == "thd":
        batch_q, max_seqlen_q, nheads_q, head_size_q = (len(cu_seqlens_q) - 1, max_seqlen_q, q.shape[1], q.shape[2])
        batch_k, max_seqlen_k, nheads_k, head_size_k = (len(cu_seqlens_k) - 1, max_seqlen_k, k.shape[1], k.shape[2])
    else:
        raise NotImplementedError("Got unsupported layout.")

    assert batch_q == batch_k
    assert head_size_q == head_size_k

    return batch_q, nheads_q, nheads_k, head_size_q, max_seqlen_q, max_seqlen_k


def get_strides_from_layout(q: Array, k: Array, v: Array, o: Array, layout: Layouts):
    if layout == "thd":
        q_strides = (0, get_stride(q, 1), get_stride(q, 0), get_stride(q, 2))
        k_strides = (0, get_stride(k, 1), get_stride(k, 0), get_stride(k, 2))
        v_strides = (0, get_stride(v, 1), get_stride(v, 0), get_stride(v, 2))
        o_strides = (0, get_stride(o, 1), get_stride(o, 0), get_stride(o, 2))
    elif layout == "bhsd":
        q_strides = (get_stride(q, 0), get_stride(q, 1), get_stride(q, 2), get_stride(q, 3))
        k_strides = (get_stride(k, 0), get_stride(k, 1), get_stride(k, 2), get_stride(k, 3))
        v_strides = (get_stride(v, 0), get_stride(v, 1), get_stride(v, 2), get_stride(v, 3))
        o_strides = (get_stride(o, 0), get_stride(o, 1), get_stride(o, 2), get_stride(o, 3))
    elif layout == "bshd":
        q_strides = (get_stride(q, 0), get_stride(q, 2), get_stride(q, 1), get_stride(q, 3))
        k_strides = (get_stride(k, 0), get_stride(k, 2), get_stride(k, 1), get_stride(k, 3))
        v_strides = (get_stride(v, 0), get_stride(v, 2), get_stride(v, 1), get_stride(v, 3))
        o_strides = (get_stride(o, 0), get_stride(o, 2), get_stride(o, 1), get_stride(o, 3))
    else:
        raise NotImplementedError("Got unsupported layout.")
    return q_strides, k_strides, v_strides, o_strides


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


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in CDNA_ARCHS


def is_rdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in RDNA_ARCHS


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
