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

"""Forward TPU Pallas kernels for quantized matrix multiplication.

This module implements the forward-pass TPU Pallas kernels for quantized
matrix multiplication (Y = X @ dequant(W)). It provides two execution paths
that are dispatched by a hybrid router:

Execution Paths:
    - **Packed (fused)**: Unpacks quantized codes, dequantizes, and multiplies
      in a single Pallas kernel. Best for large-N NF4 workloads where the
      unpack/dequant overhead is hidden by the matmul.
    - **Predecode (two-stage)**: First materializes a dense bfloat16 weight
      via ``get_predecoded_dense_weight``, then calls ``pallas_dense_matmul``.
      Generally faster for affine mode due to TPU-friendly dense scheduling.

The ``_pallas_qmm_transpose_false`` dispatcher selects between packed and
predecode based on the ``path`` argument or the hybrid heuristic.

Grid Strategy (Packed Path):
    3D grid (num_M, num_N, num_K) where M and N tiles are parallel and K
    is accumulated sequentially with a VMEM scratch buffer. Each K iteration
    unpacks the weight tile, dequantizes it, and accumulates the matmul
    result. The final K iteration stores to HBM.

Supported Modes:
    - affine: ``code * scale + bias``
    - nf4: NormalFloat4 lookup with scale
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._pallas_impl_core import (
    _ceil_div,
    _dequantize_tile,
    _normalize_tpu_blocks,
    _pad_2d,
    _pad_2d_optional,
    _unpack_bits_4_8,
    choose_packed_n_subtile,
    estimate_qmm_tpu_vmem_limit_bytes,
    get_predecoded_dense_weight,
    pallas_dense_matmul,
)

_PACKED_SUPPORTED_MODES = frozenset(("affine", "nf4"))


def _pallas_qmm_transpose_false_packed(
    x: jax.Array,
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
    """Packed fused TPU Pallas path for forward Y = X @ dequant(W).

    Runs a single Pallas kernel that, for each (M, N, K) tile, unpacks
    the quantized weight codes from packed 32-bit words, dequantizes them
    using per-group scales (and optional biases), and accumulates the
    matmul result in a VMEM scratch buffer with fp32 precision.

    Args:
        x: Activation tensor [M, K].
        w_q: Packed quantized weight [K, N // values_per_word].
        scales: Per-group scale tensor [K, N // group_size].
        biases: Optional per-group additive bias [K, N // group_size].
        group_size: Number of output elements per quantization group.
        bits: Quantization bit width (4 or 8).
        mode: Quantization mode ("affine" or "nf4").
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: Ignored (TPU fused path always uses bfloat16).

    Returns:
        Float32 result [M, N].

    Raises:
        ValueError: If bits, mode, or block constraints are invalid.
    """
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

    m, k = x.shape
    n = scales.shape[-1] * group_size
    if w_q.shape[0] != k or scales.shape[0] != k:
        raise ValueError("Packed weight/scales leading dimensions must match input K.")

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    x_pad = _pad_2d(x, m_pad - m, k_pad - k).astype(jnp.bfloat16)
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
    n_subtile = choose_packed_n_subtile(
        block_n=block_n,
        group_size=group_size,
        values_per_word=values_per_word,
    )
    subtile_words = n_subtile // values_per_word
    subtile_groups = n_subtile // group_size
    num_n_subtiles = block_n // n_subtile
    dot_dims = (((1,), (0,)), ((), ()))

    def _kernel_no_bias(x_ref, w_ref, s_ref, out_ref, acc_ref):
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        for n_i in range(num_n_subtiles):
            n_start = n_i * n_subtile
            n_end = n_start + n_subtile
            word_start = n_i * subtile_words
            word_end = word_start + subtile_words
            group_start = n_i * subtile_groups
            group_end = group_start + subtile_groups
            q = _unpack_bits_4_8(w_ref[:, word_start:word_end], bits)
            w_deq = _dequantize_tile(q, s_ref[:, group_start:group_end], None, mode, group_size).astype(jnp.bfloat16)
            acc_ref[:, n_start:n_end] += jax.lax.dot_general(
                x_ref[...].astype(jnp.bfloat16),
                w_deq,
                dot_dims,
                preferred_element_type=jnp.float32,
            )

        @pl.when(k_i == num_k - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    def _kernel_with_bias(x_ref, w_ref, s_ref, b_ref, out_ref, acc_ref):
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        for n_i in range(num_n_subtiles):
            n_start = n_i * n_subtile
            n_end = n_start + n_subtile
            word_start = n_i * subtile_words
            word_end = word_start + subtile_words
            group_start = n_i * subtile_groups
            group_end = group_start + subtile_groups
            q = _unpack_bits_4_8(w_ref[:, word_start:word_end], bits)
            w_deq = _dequantize_tile(
                q,
                s_ref[:, group_start:group_end],
                b_ref[:, group_start:group_end],
                mode,
                group_size,
            ).astype(jnp.bfloat16)
            acc_ref[:, n_start:n_end] += jax.lax.dot_general(
                x_ref[...].astype(jnp.bfloat16),
                w_deq,
                dot_dims,
                preferred_element_type=jnp.float32,
            )

        @pl.when(k_i == num_k - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    x_spec = pl.BlockSpec((block_m, block_k), lambda m_i, n_i, k_i: (m_i, k_i))
    w_spec = pl.BlockSpec((block_k, block_words), lambda m_i, n_i, k_i: (k_i, n_i))
    s_spec = pl.BlockSpec((block_k, block_groups), lambda m_i, n_i, k_i: (k_i, n_i))
    b_spec = pl.BlockSpec((block_k, block_groups), lambda m_i, n_i, k_i: (k_i, n_i))
    o_spec = pl.BlockSpec((block_m, block_n), lambda m_i, n_i, k_i: (m_i, n_i))
    grid = (num_m, num_n, num_k)

    flops = 2 * m_pad * k_pad * n_pad
    x_bytes = m_pad * k_pad * jnp.dtype(jnp.bfloat16).itemsize
    w_bytes = k_pad * words_pad * jnp.dtype(w_q.dtype).itemsize
    s_bytes = k_pad * groups_pad * jnp.dtype(scales.dtype).itemsize
    o_bytes = m_pad * n_pad * jnp.dtype(jnp.float32).itemsize
    tile_x_bytes = block_m * block_k * jnp.dtype(jnp.bfloat16).itemsize
    tile_w_bytes = block_k * block_words * jnp.dtype(w_q.dtype).itemsize
    tile_s_bytes = block_k * block_groups * jnp.dtype(scales.dtype).itemsize
    tile_o_bytes = block_m * block_n * jnp.dtype(jnp.float32).itemsize
    tile_b_bytes = 0 if biases_pad is None else (block_k * block_groups * jnp.dtype(biases_pad.dtype).itemsize)
    vmem_limit_bytes = estimate_qmm_tpu_vmem_limit_bytes(
        io_bytes=tile_x_bytes + tile_w_bytes + tile_s_bytes + tile_b_bytes + tile_o_bytes,
        scratch_bytes=tile_o_bytes,
        has_double_buffer=(num_m > 1 or num_n > 1 or num_k > 1),
    )
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=x_bytes + w_bytes + s_bytes + o_bytes, transcendentals=0)

    if biases_pad is None:
        out = pl.pallas_call(
            _kernel_no_bias,
            out_shape=jax.ShapeDtypeStruct((m_pad, n_pad), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[x_spec, w_spec, s_spec],
                out_specs=o_spec,
                grid=grid,
                scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            cost_estimate=cost_estimate,
        )(x_pad, w_q_pad, scales_pad)
    else:
        out = pl.pallas_call(
            _kernel_with_bias,
            out_shape=jax.ShapeDtypeStruct((m_pad, n_pad), jnp.float32),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[x_spec, w_spec, s_spec, b_spec],
                out_specs=o_spec,
                grid=grid,
                scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
                vmem_limit_bytes=vmem_limit_bytes,
            ),
            cost_estimate=cost_estimate,
        )(x_pad, w_q_pad, scales_pad, biases_pad)
    return out[:m, :n]


def _pallas_qmm_transpose_false_predecode(
    x: jax.Array,
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
    """Predecode-to-dense TPU Pallas path for forward Y = X @ dequant(W).

    First materializes a full dense bfloat16 weight matrix from the quantized
    representation (using cached predecoding when possible), then delegates
    to ``pallas_dense_matmul`` for the actual matmul.

    Args:
        x: Activation tensor [M, K].
        w_q: Packed quantized weight [K, N_packed].
        scales: Per-group scale tensor [K, N // group_size].
        biases: Optional per-group additive bias.
        group_size: Elements per quantization group.
        bits: Quantization bit width (4 or 8).
        mode: Quantization mode string.
        block_m: M-dimension tile size for dense matmul.
        block_n: N-dimension tile size for dense matmul.
        block_k: K-dimension tile size for dense matmul.
        use_bf16: Ignored (TPU always uses bfloat16).

    Returns:
        Float32 result [M, N].
    """
    del use_bf16
    w_dense = get_predecoded_dense_weight(
        w_q,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    return pallas_dense_matmul(
        x,
        w_dense,
        transpose_rhs=False,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )


def _prefer_packed_path(
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    mode: str,
    bits: int,
    group_size: int,
) -> bool:
    """Heuristic deciding whether the packed kernel is preferred over predecode.

    Packed wins most consistently for large-N NF4 workloads; affine is
    typically on-par or faster with predecode due to TPU-friendly dense
    matmul scheduling.

    Args:
        m: Output M dimension.
        n: Output N dimension.
        k: Reduction K dimension.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        mode: Quantization mode string.
        bits: Quantization bit-width.
        group_size: Quantization group size.

    Returns:
        True if the packed path is expected to be faster.
    """
    if bits not in (4, 8):
        return False
    if mode == "nf4":
        enough_n = n >= max(512, 2 * block_n)
        enough_m = m >= max(64, block_m // 2)
        enough_k = k >= max(256, block_k)
        valid_grouping = group_size <= block_n
        return enough_n and enough_m and enough_k and valid_grouping
    return False


def _pallas_qmm_transpose_false(
    x: jax.Array,
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
    """Dispatch forward TPU QMM based on hybrid/packed/predecode path.

    Selects between the packed fused kernel and the predecode-to-dense
    path for the forward pass (Y = X @ dequant(W)):

    - ``path="packed"``: Forces the packed fused kernel (raises if illegal).
    - ``path="predecode"``: Forces the predecode-to-dense path.
    - ``path="hybrid"``: Uses ``_prefer_packed_path`` heuristic to choose.

    Args:
        x: Activation tensor [M, K].
        w_q: Packed quantized weight [K, N_packed].
        scales: Per-group scale tensor [K, N // group_size].
        biases: Optional per-group additive bias.
        group_size: Elements per quantization group.
        bits: Quantization bit width (4 or 8).
        mode: Quantization mode string.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: Whether to use bfloat16 (ignored on TPU).
        path: Execution path (``"packed"``, ``"predecode"``, or ``"hybrid"``).
        packed_legal: Whether the packed path satisfies TPU tiling constraints.

    Returns:
        Float32 result [M, N].

    Raises:
        ValueError: If ``path="packed"`` but ``packed_legal`` is False.
    """
    n = scales.shape[-1] * group_size
    if path == "packed":
        if not packed_legal:
            raise ValueError("Packed TPU path requested but current tiling is illegal.")
        return _pallas_qmm_transpose_false_packed(
            x,
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
        return _pallas_qmm_transpose_false_predecode(
            x,
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

    m, k = x.shape
    if packed_legal and _prefer_packed_path(
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        mode=mode,
        bits=bits,
        group_size=group_size,
    ):
        return _pallas_qmm_transpose_false_packed(
            x,
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
    return _pallas_qmm_transpose_false_predecode(
        x,
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


__all__ = (
    "_pallas_qmm_transpose_false",
    "_pallas_qmm_transpose_false_packed",
    "_pallas_qmm_transpose_false_predecode",
)
