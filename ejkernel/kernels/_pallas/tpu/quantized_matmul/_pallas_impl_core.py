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

"""Shared TPU Pallas core for quantized matmul forward and input-grad paths."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.quantization._utils.fp_tables import (
    _get_e2m1_table,
    _get_e4m3_table,
    _get_nf4_table,
)

_QMM_PATHS = frozenset(("hybrid", "packed", "predecode"))
_DEFAULT_PREDECODE_CACHE_MAX_ITEMS = 2
_DEFAULT_PREDECODE_MAX_BYTES = 256 * 1024 * 1024

_PREDECODE_CACHE: OrderedDict[tuple, jax.Array] = OrderedDict()
_PREDECODE_CACHE_LOCK = threading.Lock()


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pad_2d(x: jax.Array, pad0: int, pad1: int) -> jax.Array:
    if pad0 == 0 and pad1 == 0:
        return x
    return jnp.pad(x, ((0, pad0), (0, pad1)))


def _pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int) -> jax.Array | None:
    if x is None:
        return None
    return _pad_2d(x, pad0, pad1)


def _normalize_tpu_blocks(block_m: int, block_n: int, block_k: int) -> tuple[int, int, int]:
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m/block_n/block_k must be positive.")
    block_m = max(8, _ceil_div(block_m, 8) * 8)
    block_n = max(128, _ceil_div(block_n, 128) * 128)
    block_k = max(128, _ceil_div(block_k, 128) * 128)
    return block_m, block_n, block_k


def _is_2d_blockspec_legal(block0: int, block1: int, dim0: int, dim1: int) -> bool:
    # Mosaic TPU lowering requires: trailing dim % 128 == 0 (or equals full dim),
    # second-to-trailing dim % 8 == 0 (or equals full dim).
    return (block1 == dim1 or block1 % 128 == 0) and (block0 == dim0 or block0 % 8 == 0)


def is_packed_tpu_legal_forward(
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    if bits not in (4, 8):
        return False
    try:
        block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)
    except ValueError:
        return False

    values_per_word = 32 // bits
    if block_n % values_per_word != 0 or block_n % group_size != 0:
        return False

    m, k = x.shape
    if w_q.shape[0] != k or scales.shape[0] != k:
        return False
    n = scales.shape[-1] * group_size
    if n <= 0:
        return False

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    words_pad = n_pad // values_per_word
    groups_pad = n_pad // group_size
    if w_q.shape[1] > words_pad or scales.shape[1] > groups_pad:
        return False

    block_words = block_n // values_per_word
    block_groups = block_n // group_size
    return (
        _is_2d_blockspec_legal(block_m, block_k, m_pad, k_pad)
        and _is_2d_blockspec_legal(block_k, block_words, k_pad, words_pad)
        and _is_2d_blockspec_legal(block_k, block_groups, k_pad, groups_pad)
        and _is_2d_blockspec_legal(block_m, block_n, m_pad, n_pad)
    )


def is_packed_tpu_legal_input_grad(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    if bits not in (4, 8):
        return False
    try:
        block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)
    except ValueError:
        return False

    values_per_word = 32 // bits
    if block_n % values_per_word != 0 or block_n % group_size != 0:
        return False

    m, n = dy.shape
    k = w_q.shape[0]
    if scales.shape[0] != k:
        return False
    n_expected = scales.shape[-1] * group_size
    if n != n_expected:
        return False

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    words_pad = n_pad // values_per_word
    groups_pad = n_pad // group_size
    if w_q.shape[1] > words_pad or scales.shape[1] > groups_pad:
        return False

    block_words = block_n // values_per_word
    block_groups = block_n // group_size
    return (
        _is_2d_blockspec_legal(block_m, block_n, m_pad, n_pad)
        and _is_2d_blockspec_legal(block_k, block_words, k_pad, words_pad)
        and _is_2d_blockspec_legal(block_k, block_groups, k_pad, groups_pad)
        and _is_2d_blockspec_legal(block_m, block_k, m_pad, k_pad)
    )


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def get_qmm_tpu_path() -> str:
    path = os.getenv("EJKERNEL_QMM_TPU_PATH", "hybrid").strip().lower()
    if path not in _QMM_PATHS:
        return "hybrid"
    return path


def get_predecode_cache_enabled() -> bool:
    return _parse_bool_env("EJKERNEL_QMM_TPU_PREDECODE_CACHE", True)


def get_predecode_cache_max_items() -> int:
    return _parse_int_env("EJKERNEL_QMM_TPU_PREDECODE_CACHE_MAX_ITEMS", _DEFAULT_PREDECODE_CACHE_MAX_ITEMS)


def get_predecode_max_bytes() -> int:
    return _parse_int_env("EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES", _DEFAULT_PREDECODE_MAX_BYTES)


def _decode_e2m1(code: jax.Array) -> jax.Array:
    table, _ = _get_e2m1_table()
    return table[code.astype(jnp.int32)]


def _decode_e4m3(code: jax.Array) -> jax.Array:
    table, _ = _get_e4m3_table()
    return table[code.astype(jnp.int32)]


def _decode_nf4(code: jax.Array) -> jax.Array:
    table = _get_nf4_table()
    return table[code.astype(jnp.int32)]


def _unpack_bits_4_8(words: jax.Array, bits: int) -> jax.Array:
    words = words.astype(jnp.uint32)
    if bits == 4:
        q = jnp.stack(
            [
                (words >> jnp.uint32(0)) & jnp.uint32(0xF),
                (words >> jnp.uint32(4)) & jnp.uint32(0xF),
                (words >> jnp.uint32(8)) & jnp.uint32(0xF),
                (words >> jnp.uint32(12)) & jnp.uint32(0xF),
                (words >> jnp.uint32(16)) & jnp.uint32(0xF),
                (words >> jnp.uint32(20)) & jnp.uint32(0xF),
                (words >> jnp.uint32(24)) & jnp.uint32(0xF),
                (words >> jnp.uint32(28)) & jnp.uint32(0xF),
            ],
            axis=-1,
        )
        return q.reshape(words.shape[0], words.shape[1] * 8)
    if bits == 8:
        q = jnp.stack(
            [
                (words >> jnp.uint32(0)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(8)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(16)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(24)) & jnp.uint32(0xFF),
            ],
            axis=-1,
        )
        return q.reshape(words.shape[0], words.shape[1] * 4)
    raise ValueError("Only bits in {4, 8} are supported by _unpack_bits_4_8.")


def _expand_groups(values: jax.Array, group_size: int, width: int) -> jax.Array:
    groups = values.shape[-1]
    expanded = jnp.broadcast_to(values[..., :, None], (*values.shape, group_size))
    return expanded.reshape(values.shape[0], groups * group_size)[:, :width]


def _dequantize_tile(
    q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    mode: str,
    group_size: int,
) -> jax.Array:
    width = q.shape[-1]
    if mode == "affine":
        vals = q.astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(scales.astype(jnp.float32), group_size, width)
        w = vals * s
        if biases is not None:
            b = _expand_groups(biases.astype(jnp.float32), group_size, width)
            w = w + b
        return w
    if mode == "nf4":
        vals = _decode_nf4(q)
        s = _expand_groups(scales.astype(jnp.float32), group_size, width)
        return vals * s
    if mode == "mxfp4":
        vals = _decode_e2m1(q)
        exp = scales.astype(jnp.int8).astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(jnp.exp2(exp), group_size, width)
        return vals * s
    if mode == "mxfp8":
        vals = _decode_e4m3(q)
        exp = scales.astype(jnp.int8).astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(jnp.exp2(exp), group_size, width)
        return vals * s
    if mode == "nvfp4":
        vals = _decode_e2m1(q)
        s_decoded = _decode_e4m3(scales.astype(jnp.uint32))
        s = _expand_groups(s_decoded, group_size, width)
        return vals * s
    if mode == "nvfp8":
        vals = _decode_e4m3(q)
        s_decoded = _decode_e4m3(scales.astype(jnp.uint32))
        s = _expand_groups(s_decoded, group_size, width)
        return vals * s
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _predecode_dense_weight(
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> jax.Array:
    if bits not in (4, 8):
        raise ValueError("TPU predecode path supports bits in {4, 8}.")
    n = scales.shape[-1] * group_size
    q_full = _unpack_bits_4_8(w_q, bits)
    if q_full.shape[-1] < n:
        raise ValueError("Packed weight width is smaller than scales-implied output width.")
    q = q_full[:, :n]
    w = _dequantize_tile(q, scales, biases, mode, group_size)
    # TPU fused path computes using bf16 inputs with fp32 accumulation.
    return w.astype(jnp.bfloat16)


def _device_key(arr: jax.Array) -> tuple | None:
    try:
        dev = arr.device()
    except Exception:
        try:
            devices = list(arr.devices())
            dev = devices[0] if devices else None
        except Exception:
            dev = None
    if dev is None:
        return None
    return (getattr(dev, "platform", None), getattr(dev, "id", None), str(dev))


def _is_tracer(x: object) -> bool:
    return isinstance(x, jax_core.Tracer)


def _estimate_predecode_bytes(k: int, n: int) -> int:
    return int(k) * int(n) * jnp.dtype(jnp.bfloat16).itemsize


def get_predecoded_dense_weight(
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> jax.Array:
    k = int(w_q.shape[0])
    n = int(scales.shape[-1]) * int(group_size)
    est_bytes = _estimate_predecode_bytes(k, n)
    max_bytes = get_predecode_max_bytes()
    if max_bytes > 0 and est_bytes > max_bytes:
        raise ValueError(
            f"Predecode buffer exceeds cap ({est_bytes} bytes > {max_bytes} bytes). "
            "Increase EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES or use packed/XLA path."
        )

    cache_allowed = get_predecode_cache_enabled()
    cache_allowed = cache_allowed and not _is_tracer(w_q) and not _is_tracer(scales)
    cache_allowed = cache_allowed and (biases is None or not _is_tracer(biases))
    if not cache_allowed:
        return _predecode_dense_weight(
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    key = (
        _device_key(w_q),
        id(w_q),
        id(scales),
        id(biases) if biases is not None else None,
        w_q.shape,
        w_q.dtype,
        scales.shape,
        scales.dtype,
        biases.shape if biases is not None else None,
        biases.dtype if biases is not None else None,
        group_size,
        bits,
        mode,
    )

    with _PREDECODE_CACHE_LOCK:
        cached = _PREDECODE_CACHE.get(key)
        if cached is not None:
            _PREDECODE_CACHE.move_to_end(key)
            return cached

    decoded = _predecode_dense_weight(
        w_q,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    max_items = get_predecode_cache_max_items()
    if max_items <= 0:
        return decoded

    with _PREDECODE_CACHE_LOCK:
        _PREDECODE_CACHE[key] = decoded
        _PREDECODE_CACHE.move_to_end(key)
        while len(_PREDECODE_CACHE) > max_items:
            _PREDECODE_CACHE.popitem(last=False)
    return decoded


def pallas_dense_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    transpose_rhs: bool,
    block_m: int,
    block_n: int,
    block_k: int,
) -> jax.Array:
    block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)

    m, k = lhs.shape
    if transpose_rhs:
        n = rhs.shape[0]
        k_rhs = rhs.shape[1]
    else:
        k_rhs = rhs.shape[0]
        n = rhs.shape[1]
    if k != k_rhs:
        raise ValueError("Dense Pallas matmul dimension mismatch on contraction axis.")

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    lhs_pad = _pad_2d(lhs, m_pad - m, k_pad - k).astype(jnp.bfloat16)
    if transpose_rhs:
        rhs_pad = _pad_2d(rhs, n_pad - n, k_pad - k).astype(jnp.bfloat16)
        dot_dims = (((1,), (1,)), ((), ()))
        rhs_spec = pl.BlockSpec((block_n, block_k), lambda m_i, n_i, k_i: (n_i, k_i))
    else:
        rhs_pad = _pad_2d(rhs, k_pad - k, n_pad - n).astype(jnp.bfloat16)
        dot_dims = (((1,), (0,)), ((), ()))
        rhs_spec = pl.BlockSpec((block_k, block_n), lambda m_i, n_i, k_i: (k_i, n_i))

    num_m = m_pad // block_m
    num_n = n_pad // block_n
    num_k = k_pad // block_k

    def _kernel(lhs_ref, rhs_ref, out_ref, acc_ref):
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] += jax.lax.dot_general(
            lhs_ref[...].astype(jnp.bfloat16),
            rhs_ref[...].astype(jnp.bfloat16),
            dot_dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(k_i == num_k - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    lhs_spec = pl.BlockSpec((block_m, block_k), lambda m_i, n_i, k_i: (m_i, k_i))
    out_spec = pl.BlockSpec((block_m, block_n), lambda m_i, n_i, k_i: (m_i, n_i))
    grid = (num_m, num_n, num_k)

    flops = 2 * m_pad * k_pad * n_pad
    lhs_bytes = m_pad * k_pad * jnp.dtype(jnp.bfloat16).itemsize
    rhs_bytes = n_pad * k_pad * jnp.dtype(jnp.bfloat16).itemsize
    out_bytes = m_pad * n_pad * jnp.dtype(jnp.float32).itemsize
    cost_estimate = pl.CostEstimate(
        flops=flops,
        bytes_accessed=lhs_bytes + rhs_bytes + out_bytes,
        transcendentals=0,
    )

    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m_pad, n_pad), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[lhs_spec, rhs_spec],
            out_specs=out_spec,
            grid=grid,
            scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary")),
        cost_estimate=cost_estimate,
    )(lhs_pad, rhs_pad)
    return out[:m, :n]


__all__ = (
    "_ceil_div",
    "_decode_e2m1",
    "_decode_e4m3",
    "_decode_nf4",
    "_dequantize_tile",
    "_normalize_tpu_blocks",
    "_pad_2d",
    "_pad_2d_optional",
    "_unpack_bits_4_8",
    "get_predecoded_dense_weight",
    "get_qmm_tpu_path",
    "is_packed_tpu_legal_forward",
    "is_packed_tpu_legal_input_grad",
    "pallas_dense_matmul",
)
