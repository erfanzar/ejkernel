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

"""Backward TPU Pallas kernels for quantized matmul."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.callib._ejit import ejit
from ejkernel.quantization import dequantize

from ...._xla.quantized_matmul import quantized_matmul as _xla_quantized_matmul
from ._pallas_impl_core import (
    _ceil_div,
    _dequantize_tile,
    _normalize_tpu_blocks,
    _pad_2d,
    _pad_2d_optional,
    _unpack_bits_4_8,
    get_predecoded_dense_weight,
    pallas_dense_matmul,
)

_PACKED_SUPPORTED_MODES = frozenset(("affine", "nf4"))


def _pallas_qmm_input_grad_transpose_false_packed(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> jax.Array:
    """Packed fused TPU Pallas path for dX when forward transpose=False."""
    del use_bf16  # TPU fused path always computes in bfloat16.
    if bits not in (4, 8):
        raise ValueError("TPU packed fused path supports bits in {4, 8}.")
    if mode not in _PACKED_SUPPORTED_MODES:
        raise ValueError(f"TPU packed fused path currently supports modes {sorted(_PACKED_SUPPORTED_MODES)}.")

    block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)
    values_per_word = 32 // bits
    if block_n % values_per_word != 0:
        raise ValueError("block_n must be a multiple of values_per_word for TPU packed path.")
    if block_n % group_size != 0:
        raise ValueError("block_n must be a multiple of group_size for TPU packed path.")

    m, n = dy.shape
    k = w_q.shape[0]
    n_expected = scales.shape[-1] * group_size
    if n != n_expected:
        raise ValueError("dy.shape[-1] must match scales-implied N.")
    if scales.shape[0] != k:
        raise ValueError("Packed scales leading dimension must match packed weight K.")

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    dy_pad = _pad_2d(dy, m_pad - m, n_pad - n).astype(jnp.bfloat16)
    words_pad = n_pad // values_per_word
    groups_pad = n_pad // group_size
    if w_q.shape[1] > words_pad or scales.shape[1] > groups_pad:
        raise ValueError("Packed/scales trailing dimensions are incompatible with tiled N padding.")

    w_q_pad = _pad_2d(w_q, k_pad - k, words_pad - w_q.shape[-1])
    scales_pad = _pad_2d(scales, k_pad - k, groups_pad - scales.shape[-1])
    biases_pad = _pad_2d_optional(biases, k_pad - k, groups_pad - scales.shape[-1])

    num_m = m_pad // block_m
    num_n = n_pad // block_n
    num_k = k_pad // block_k
    block_words = block_n // values_per_word
    block_groups = block_n // group_size
    # dX = dY @ W^T, contract over N.
    dot_dims = (((1,), (1,)), ((), ()))

    def _kernel_no_bias(dy_ref, w_ref, s_ref, out_ref, acc_ref):
        n_i = pl.program_id(1)

        @pl.when(n_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        q = _unpack_bits_4_8(w_ref[...], bits)
        w_deq = _dequantize_tile(q, s_ref[...], None, mode, group_size).astype(jnp.bfloat16)
        acc_ref[...] += jax.lax.dot_general(
            dy_ref[...].astype(jnp.bfloat16),
            w_deq,
            dot_dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(n_i == num_n - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    def _kernel_with_bias(dy_ref, w_ref, s_ref, b_ref, out_ref, acc_ref):
        n_i = pl.program_id(1)

        @pl.when(n_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        q = _unpack_bits_4_8(w_ref[...], bits)
        w_deq = _dequantize_tile(q, s_ref[...], b_ref[...], mode, group_size).astype(jnp.bfloat16)
        acc_ref[...] += jax.lax.dot_general(
            dy_ref[...].astype(jnp.bfloat16),
            w_deq,
            dot_dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(n_i == num_n - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    dy_spec = pl.BlockSpec((block_m, block_n), lambda m_i, n_i, k_i: (m_i, n_i))
    w_spec = pl.BlockSpec((block_k, block_words), lambda m_i, n_i, k_i: (k_i, n_i))
    s_spec = pl.BlockSpec((block_k, block_groups), lambda m_i, n_i, k_i: (k_i, n_i))
    b_spec = pl.BlockSpec((block_k, block_groups), lambda m_i, n_i, k_i: (k_i, n_i))
    o_spec = pl.BlockSpec((block_m, block_k), lambda m_i, n_i, k_i: (m_i, k_i))
    grid = (num_m, num_n, num_k)

    flops = 2 * m_pad * k_pad * n_pad
    dy_bytes = m_pad * n_pad * jnp.dtype(jnp.bfloat16).itemsize
    w_bytes = k_pad * words_pad * jnp.dtype(w_q.dtype).itemsize
    s_bytes = k_pad * groups_pad * jnp.dtype(scales.dtype).itemsize
    o_bytes = m_pad * k_pad * jnp.dtype(jnp.float32).itemsize
    cost_estimate = pl.CostEstimate(
        flops=flops, bytes_accessed=dy_bytes + w_bytes + s_bytes + o_bytes, transcendentals=0
    )

    if biases_pad is None:
        out = pl.pallas_call(
            _kernel_no_bias,
            out_shape=jax.ShapeDtypeStruct((m_pad, k_pad), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[dy_spec, w_spec, s_spec],
                out_specs=o_spec,
                grid=grid,
                scratch_shapes=[pltpu.VMEM((block_m, block_k), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "parallel")),
            cost_estimate=cost_estimate,
        )(dy_pad, w_q_pad, scales_pad)
    else:
        out = pl.pallas_call(
            _kernel_with_bias,
            out_shape=jax.ShapeDtypeStruct((m_pad, k_pad), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[dy_spec, w_spec, s_spec, b_spec],
                out_specs=o_spec,
                grid=grid,
                scratch_shapes=[pltpu.VMEM((block_m, block_k), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "parallel")),
            cost_estimate=cost_estimate,
        )(dy_pad, w_q_pad, scales_pad, biases_pad)
    return out[:m, :k]


def _pallas_qmm_input_grad_transpose_false_predecode(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> jax.Array:
    """Predecode-to-dense TPU Pallas path for dX when forward transpose=False."""
    del use_bf16
    w_dense = get_predecoded_dense_weight(
        w_q,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    # dX = dY @ W^T.
    return pallas_dense_matmul(
        dy,
        w_dense,
        transpose_rhs=True,
        block_m=block_m,
        block_n=block_k,
        block_k=block_n,
    )


def _prefer_packed_path(n: int, block_n: int, mode: str) -> bool:
    if mode == "nf4":
        return n >= max(512, 2 * block_n)
    return False


def _quantized_matmul_input_grad_hybrid(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    path: str,
    packed_legal: bool,
) -> jax.Array:
    n = dy.shape[-1]
    if path == "packed":
        if not packed_legal:
            raise ValueError("Packed TPU path requested but current dX tiling is illegal.")
        return _pallas_qmm_input_grad_transpose_false_packed(
            dy,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )
    if path == "predecode":
        return _pallas_qmm_input_grad_transpose_false_predecode(
            dy,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )

    if packed_legal and _prefer_packed_path(n, block_n, mode):
        return _pallas_qmm_input_grad_transpose_false_packed(
            dy,
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )
    return _pallas_qmm_input_grad_transpose_false_predecode(
        dy,
        w_q,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
    )


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
        "path",
        "packed_legal",
    ],
)
def quantized_matmul_input_grad(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    path: str,
    packed_legal: bool,
) -> jax.Array:
    """Gradient w.r.t. input for TPU Pallas quantized matmul."""
    del use_bf16
    zeros = None
    if mode == "affine":
        if biases is None:
            raise ValueError("affine input grad requires affine metadata.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        zeros = -biases / safe_scale
    # Forward transpose=True path currently stays on XLA in this backend.
    if transpose:
        return _xla_quantized_matmul(
            dy,
            w_q,
            scales,
            zeros,
            transpose=False,
            group_size=group_size,
            bits=bits,
            mode=mode,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=True,
        )

    if bits in (4, 8):
        try:
            return _quantized_matmul_input_grad_hybrid(
                dy,
                w_q,
                scales,
                biases,
                group_size=group_size,
                bits=bits,
                mode=mode,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                use_bf16=True,
                path=path,
                packed_legal=packed_legal,
            )
        except Exception:
            pass

    w_f = dequantize(w_q, scales, zeros, group_size=group_size, bits=bits, mode=mode)
    return jax.lax.dot_general(dy, w_f, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
