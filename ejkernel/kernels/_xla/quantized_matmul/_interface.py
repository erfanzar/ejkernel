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

"""Quantized matmul interface using XLA.

This module provides a JAX/XLA implementation of quantized matrix
multiplication. It supports the MLX-style packed uint32 layout and the
quantization modes implemented in ejkernel.quantization.

For performance, a blocked decode-and-matmul path is used when the
quantization parameters are compatible with fixed tiling. When the
inputs are not aligned to the tile sizes or use unsupported bit-widths,
this falls back to dequantize+matmul.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit
from ejkernel.quantization._quants.quantizations import dequantize
from ejkernel.quantization._utils.bitpack import _unpack_bits
from ejkernel.quantization._utils.fp_tables import (
    _get_e2m1_table,
    _get_e4m3_table,
    _get_nf4_table,
)
from ejkernel.quantization._utils.grouping import _require_bits

from ..._registry import Backend, Platform, kernel_registry

#: Supported quantization modes for the XLA implementation.
QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve and validate quantization parameters.

    Applies mode-specific defaults and validates that the parameters are
    compatible with the quantization mode.

    Args:
        mode: Quantization mode (affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8).
        group_size: Number of elements per quantization group, or None for default.
        bits: Bit-width per quantized element, or None for default.

    Returns:
        Tuple of (resolved_group_size, resolved_bits).

    Raises:
        ValueError: If mode is unsupported or parameters are invalid for the mode.
    """
    mode = mode.lower()
    if mode == "affine":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else _require_bits(bits, {2, 3, 4, 5, 6, 7, 8})
        if group_size not in {32, 64, 128}:
            raise ValueError("affine mode supports group_size in {32, 64, 128}.")
        return group_size, bits
    if mode == "mxfp4":
        group_size = 32 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 32 or bits != 4:
            raise ValueError("mxfp4 requires group_size=32 and bits=4.")
        return group_size, bits
    if mode == "mxfp8":
        group_size = 32 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 32 or bits != 8:
            raise ValueError("mxfp8 requires group_size=32 and bits=8.")
        return group_size, bits
    if mode == "nvfp4":
        group_size = 16 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if group_size != 16 or bits != 4:
            raise ValueError("nvfp4 requires group_size=16 and bits=4.")
        return group_size, bits
    if mode == "nvfp8":
        group_size = 16 if group_size is None else int(group_size)
        bits = 8 if bits is None else bits
        if group_size != 16 or bits != 8:
            raise ValueError("nvfp8 requires group_size=16 and bits=8.")
        return group_size, bits
    if mode == "nf4":
        group_size = 64 if group_size is None else int(group_size)
        bits = 4 if bits is None else bits
        if bits != 4:
            raise ValueError("nf4 requires bits=4.")
        return group_size, bits
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division of a by b."""
    return (a + b - 1) // b


def _pad_2d(x: jax.Array, pad0: int, pad1: int) -> jax.Array:
    """Pad a 2D array with zeros on the right and bottom edges.

    Args:
        x: Input 2D array.
        pad0: Padding for dimension 0 (rows).
        pad1: Padding for dimension 1 (columns).

    Returns:
        Padded array, or original array if no padding needed.
    """
    if pad0 == 0 and pad1 == 0:
        return x
    return jnp.pad(x, ((0, pad0), (0, pad1)))


def _pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int) -> jax.Array | None:
    """Pad a 2D array if not None, otherwise return None."""
    if x is None:
        return None
    return _pad_2d(x, pad0, pad1)


def _decode_tile_affine(
    q: jax.Array,
    scale_tile: jax.Array,
    bias_tile: jax.Array | None,
) -> jax.Array:
    """Decode affine quantized tile: out = q * scale + bias.

    Args:
        q: Quantized codes tile.
        scale_tile: Per-element scales (broadcast from per-group).
        bias_tile: Per-element biases (broadcast from per-group), or None.

    Returns:
        Dequantized weight tile.
    """
    out = q.astype(scale_tile.dtype) * scale_tile
    if bias_tile is not None:
        out = out + bias_tile
    return out


def _decode_tile_nf4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NF4 quantized tile using codebook lookup.

    Args:
        q: NF4 codes tile (0-15).
        scale_tile: Per-element absmax scales.

    Returns:
        Dequantized weight tile.
    """
    nf4_table = _get_nf4_table()
    vals = nf4_table[q.astype(jnp.int32)]
    return vals * scale_tile


def _decode_tile_mxfp4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode MXFP4 (E2M1) quantized tile with E8M0 shared exponent.

    Args:
        q: E2M1 codes tile (0-15).
        scale_tile: E8M0 shared exponents as uint8.

    Returns:
        Dequantized weight tile.
    """
    e2m1_table, _ = _get_e2m1_table()
    vals = e2m1_table[q.astype(jnp.int32)]
    exp = scale_tile.astype(jnp.int8).astype(jnp.float32)
    scale = jnp.exp2(exp)
    return vals * scale


def _decode_tile_mxfp8(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode MXFP8 (E4M3) quantized tile with E8M0 shared exponent.

    Args:
        q: E4M3 codes tile (0-255).
        scale_tile: E8M0 shared exponents as uint8.

    Returns:
        Dequantized weight tile.
    """
    e4m3_table, _ = _get_e4m3_table()
    vals = e4m3_table[q.astype(jnp.int32)]
    exp = scale_tile.astype(jnp.int8).astype(jnp.float32)
    scale = jnp.exp2(exp)
    return vals * scale


def _decode_tile_nvfp4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NVFP4 (E2M1 with E4M3 scale) quantized tile.

    Args:
        q: E2M1 codes tile (0-15).
        scale_tile: E4M3 per-group scales as uint8.

    Returns:
        Dequantized weight tile.
    """
    e2m1_table, _ = _get_e2m1_table()
    e4m3_table, _ = _get_e4m3_table()
    vals = e2m1_table[q.astype(jnp.int32)]
    scale = e4m3_table[scale_tile.astype(jnp.int32)]
    return vals * scale


def _decode_tile_nvfp8(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NVFP8 (E4M3 with E4M3 scale) quantized tile.

    Args:
        q: E4M3 codes tile (0-255).
        scale_tile: E4M3 per-group scales as uint8.

    Returns:
        Dequantized weight tile.
    """
    e4m3_table, _ = _get_e4m3_table()
    vals = e4m3_table[q.astype(jnp.int32)]
    scale = e4m3_table[scale_tile.astype(jnp.int32)]
    return vals * scale


def _dot_general(a: jax.Array, b: jax.Array) -> jax.Array:
    """Perform matrix multiplication with float32 output accumulation.

    Args:
        a: Left operand of shape (M, K).
        b: Right operand of shape (K, N).

    Returns:
        Result of shape (M, N) in float32.
    """
    return jax.lax.dot_general(
        a,
        b,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _blocked_quantized_matmul(
    x: jax.Array,
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
) -> jax.Array:
    """Execute blocked quantized matmul with fused dequantization.

    This function performs tiled matrix multiplication where each tile's
    weights are dequantized on-the-fly before computing the dot product.
    This avoids materializing the full dequantized weight matrix in memory.

    The algorithm pads inputs to block boundaries, then iterates over tiles
    using nested fori_loops for M, N, and K dimensions.

    Args:
        x: Input activation matrix of shape (M, K).
        w_q: Packed uint32 weights.
        scales: Per-group scales array.
        biases: Per-group biases (affine mode only).
        transpose: If True, weights are in NxK layout; if False, KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element (4 or 8).
        mode: Quantization mode (affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8).
        block_m: Block size for M dimension.
        block_n: Block size for N dimension.
        block_k: Block size for K dimension.
        use_bf16: If True, use BF16 for tile computations; if False, FP16.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If block sizes are incompatible with group_size or bits.
    """
    mode = mode.lower()
    M, K = x.shape

    if transpose:
        N = w_q.shape[0]
    else:
        N = scales.shape[-1] * group_size

    values_per_word = 32 // bits

    if transpose:
        if block_k % values_per_word != 0 or block_k % group_size != 0:
            raise ValueError("block_k must be a multiple of values_per_word and group_size for transpose=True")
    else:
        if block_n % values_per_word != 0 or block_n % group_size != 0:
            raise ValueError("block_n must be a multiple of values_per_word and group_size for transpose=False")

    M_pad = _ceil_div(M, block_m) * block_m
    N_pad = _ceil_div(N, block_n) * block_n
    K_pad = _ceil_div(K, block_k) * block_k

    x_pad = _pad_2d(x, M_pad - M, K_pad - K)

    if transpose:
        words_pad = K_pad // values_per_word
        w_q_pad = _pad_2d(w_q, N_pad - N, words_pad - w_q.shape[-1])
        groups_pad = K_pad // group_size
        scales_pad = _pad_2d(scales, N_pad - N, groups_pad - scales.shape[-1])
        biases_pad = _pad_2d_optional(biases, N_pad - N, groups_pad - scales.shape[-1])
    else:
        words_pad = N_pad // values_per_word
        w_q_pad = _pad_2d(w_q, K_pad - K, words_pad - w_q.shape[-1])
        groups_pad = N_pad // group_size
        scales_pad = _pad_2d(scales, K_pad - K, groups_pad - scales.shape[-1])
        biases_pad = _pad_2d_optional(biases, K_pad - K, groups_pad - scales.shape[-1])

    compute_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
    out = jnp.zeros((M_pad, N_pad), dtype=jnp.float32)

    num_m = M_pad // block_m
    num_n = N_pad // block_n
    num_k = K_pad // block_k

    def load_q_tile(off_k: int, off_n: int) -> jax.Array:
        """Load and unpack a tile of quantized weight codes.

        Slices the packed uint32 weight array and unpacks it into
        individual quantized codes for the tile at (off_k, off_n).

        Args:
            off_k: Offset along the K dimension (reduction axis).
            off_n: Offset along the N dimension (output axis).

        Returns:
            Unpacked quantized codes of shape (block_k, block_n).
        """
        if transpose:
            word_start = off_k // values_per_word
            words_tile = block_k // values_per_word
            wq_tile = jax.lax.dynamic_slice(w_q_pad, (off_n, word_start), (block_n, words_tile))
            q = _unpack_bits(wq_tile, block_k, bits)
            q = jnp.swapaxes(q, 0, 1)
        else:
            word_start = off_n // values_per_word
            words_tile = block_n // values_per_word
            wq_tile = jax.lax.dynamic_slice(w_q_pad, (off_k, word_start), (block_k, words_tile))
            q = _unpack_bits(wq_tile, block_n, bits)
        return q

    def load_group_tile(off_k: int, off_n: int) -> tuple[jax.Array, jax.Array | None]:
        """Load per-group scales and biases for a weight tile.

        Slices the scales (and optionally biases) arrays and repeats
        them to match the tile dimensions for element-wise dequantization.

        Args:
            off_k: Offset along the K dimension.
            off_n: Offset along the N dimension.

        Returns:
            Tuple of (scale_tile, bias_tile) where scale_tile has shape
            (block_k, block_n) and bias_tile is the same shape or None.
        """
        if transpose:
            group_start = off_k // group_size
            groups_tile = block_k // group_size
            scale_tile = jax.lax.dynamic_slice(scales_pad, (off_n, group_start), (block_n, groups_tile))
            scale_tile = jnp.repeat(scale_tile, group_size, axis=1)
            scale_tile = jnp.swapaxes(scale_tile, 0, 1)
            bias_tile = None
            if biases_pad is not None:
                bias_tile = jax.lax.dynamic_slice(biases_pad, (off_n, group_start), (block_n, groups_tile))
                bias_tile = jnp.repeat(bias_tile, group_size, axis=1)
                bias_tile = jnp.swapaxes(bias_tile, 0, 1)
        else:
            group_start = off_n // group_size
            groups_tile = block_n // group_size
            scale_tile = jax.lax.dynamic_slice(scales_pad, (off_k, group_start), (block_k, groups_tile))
            scale_tile = jnp.repeat(scale_tile, group_size, axis=1)
            bias_tile = None
            if biases_pad is not None:
                bias_tile = jax.lax.dynamic_slice(biases_pad, (off_k, group_start), (block_k, groups_tile))
                bias_tile = jnp.repeat(bias_tile, group_size, axis=1)
        return scale_tile, bias_tile

    def decode_tile(off_k: int, off_n: int) -> jax.Array:
        """Dequantize a single weight tile using the configured quantization mode.

        Loads packed codes and group parameters, then applies the
        mode-specific decoding formula (affine, nf4, mxfp4, etc.)
        to produce a float tile.

        Args:
            off_k: Offset along the K dimension.
            off_n: Offset along the N dimension.

        Returns:
            Dequantized weight tile of shape (block_k, block_n) in compute_dtype.
        """
        q = load_q_tile(off_k, off_n)
        scale_tile, bias_tile = load_group_tile(off_k, off_n)
        if mode == "affine":
            w = _decode_tile_affine(q, scale_tile, bias_tile)
        elif mode == "nf4":
            w = _decode_tile_nf4(q, scale_tile)
        elif mode == "mxfp4":
            w = _decode_tile_mxfp4(q, scale_tile)
        elif mode == "mxfp8":
            w = _decode_tile_mxfp8(q, scale_tile)
        elif mode == "nvfp4":
            w = _decode_tile_nvfp4(q, scale_tile)
        elif mode == "nvfp8":
            w = _decode_tile_nvfp8(q, scale_tile)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        return w.astype(compute_dtype)

    def k_loop(k_idx: int, acc: jax.Array, *, off_m: int, off_n: int) -> jax.Array:
        """Accumulate one K-dimension tile into the output accumulator.

        Loads the activation tile and dequantized weight tile, performs
        a dot product, and adds the result to the running accumulator.

        Args:
            k_idx: K-dimension block index.
            acc: Running accumulator of shape (block_m, block_n) in float32.
            off_m: M-dimension offset for the current tile.
            off_n: N-dimension offset for the current tile.

        Returns:
            Updated accumulator with this K-tile's contribution added.
        """
        off_k = k_idx * block_k
        x_tile = jax.lax.dynamic_slice(x_pad, (off_m, off_k), (block_m, block_k)).astype(compute_dtype)
        w_tile = decode_tile(off_k, off_n)
        acc = acc + _dot_general(x_tile, w_tile)
        return acc

    def n_loop(n_idx: int, out_buf: jax.Array) -> jax.Array:
        """Process one N-dimension block column across all M rows.

        Iterates over all M-dimension blocks and accumulates the full
        K-dimension reduction for each (M, N) output tile.

        Args:
            n_idx: N-dimension block index.
            out_buf: Output buffer of shape (M_pad, N_pad) being filled.

        Returns:
            Updated output buffer with this N column computed.
        """
        off_n = n_idx * block_n

        def m_loop(m_idx: int, out_local: jax.Array) -> jax.Array:
            """Process one M-dimension block row for a fixed N column.

            Accumulates the dot product over all K-dimension tiles and
            writes the result to the output buffer.

            Args:
                m_idx: M-dimension block index.
                out_local: Output buffer being updated.

            Returns:
                Updated output buffer with this (M, N) tile written.
            """
            off_m = m_idx * block_m
            acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)

            def k_body(idx: int, carry: jax.Array) -> jax.Array:
                """Inner loop body for K-dimension accumulation.

                Args:
                    idx: K-dimension block index.
                    carry: Running accumulator.

                Returns:
                    Updated accumulator.
                """
                return k_loop(idx, carry, off_m=off_m, off_n=off_n)

            acc = jax.lax.fori_loop(0, num_k, k_body, acc)
            out_local = jax.lax.dynamic_update_slice(out_local, acc, (off_m, off_n))
            return out_local

        out_buf = jax.lax.fori_loop(0, num_m, m_loop, out_buf)
        return out_buf

    out = jax.lax.fori_loop(0, num_n, n_loop, out)
    return out[:M, :N]


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
    ],
)
def _operate(
    x,
    w,
    scales,
    biases,
    transpose,
    group_size,
    bits,
    mode,
    block_m,
    block_n,
    block_k,
    use_bf16,
):
    """Execute quantized matmul with automatic path selection.

    Attempts the blocked fused dequantize-and-matmul path first for
    4-bit and 8-bit modes with valid block sizes. If the fused path
    raises a ValueError (e.g., due to shape or tiling mismatches),
    falls back to a two-step dequantize+matmul approach.

    Args:
        x: Input activation matrix of shape (M, K).
        w: Packed uint32 weights.
        scales: Per-group scale parameters.
        biases: Per-group bias parameters (affine mode) or None.
        transpose: Whether weights are in NxK (transposed) layout.
        group_size: Elements per quantization group.
        bits: Bit-width per quantized element.
        mode: Quantization mode string.
        block_m: Tile size for M dimension.
        block_n: Tile size for N dimension.
        block_k: Tile size for K dimension.
        use_bf16: Whether to use BF16 for intermediate computations.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.
    """
    can_fuse = bits in (4, 8)
    can_fuse = can_fuse and block_m > 0 and block_n > 0 and block_k > 0

    if can_fuse:
        try:
            return _blocked_quantized_matmul(
                x,
                w,
                scales,
                biases,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                use_bf16=use_bf16,
            )
        except ValueError:
            # Shape or tiling mismatch, fall back to dequantize+matmul.
            pass

    w_f = dequantize(w, scales, biases, group_size=group_size, bits=bits, mode=mode)
    return x @ w_f.T if transpose else x @ w_f


@kernel_registry.register("quantized_matmul", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    biases: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Quantized matrix multiplication using XLA.

    This function provides a portable XLA implementation of quantized matmul
    that supports all quantization modes (affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8).
    When possible, it uses a blocked algorithm with fused dequantization for
    better performance. For incompatible shapes or bit-widths, it falls back
    to a simple dequantize+matmul approach.

    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights produced by quantize() or prepack_quantized_weights().
            For transpose=True, shape is (N, ceil(K/values_per_word)).
            For transpose=False, shape is (K, ceil(N/values_per_word)).
        scales: Per-group scales array. Shape depends on mode:
            - affine/nf4: float scales, shape (N, K//group_size) or (K, N//group_size)
            - mxfp4/mxfp8: uint8 E8M0 exponents
            - nvfp4/nvfp8: uint8 E4M3 scale codes
        biases: Per-group biases (required for affine mode only). Must have
            the same shape as scales. Must be None for non-affine modes.
        transpose: If True, weights are in NxK layout (transposed) and the
            computation is x @ w.T. If False, weights are in KxN layout and
            the computation is x @ w. Default is False.
        group_size: Number of elements per quantization group. If None, uses
            mode default (64 for affine/nf4, 32 for mxfp4/mxfp8, 16 for nvfp4/nvfp8).
        bits: Bit-width per quantized element. If None, uses mode default
            (4 for affine/nf4/mxfp4/nvfp4, 8 for mxfp8/nvfp8).
        mode: Quantization mode determining the dequantization formula:
            - "affine": Linear scale+bias (q * scale + bias)
            - "nf4": 4-bit NormalFloat codebook
            - "mxfp4": Microscaling FP4 (E2M1) with E8M0 exponent
            - "mxfp8": Microscaling FP8 (E4M3) with E8M0 exponent
            - "nvfp4": NVIDIA FP4 (E2M1) with E4M3 per-group scale
            - "nvfp8": NVIDIA FP8 (E4M3) with E4M3 per-group scale
        block_m: Tile size for M dimension in blocked algorithm. Default 128.
        block_n: Tile size for N dimension in blocked algorithm. Default 128.
        block_k: Tile size for K dimension in blocked algorithm. Default 64.
        use_bf16: If True, use BF16 for intermediate dot products.
            If False, use FP16. Default is True.
        num_warps: Ignored in XLA path (Triton-only).
        num_stages: Ignored in XLA path (Triton-only).
        split_k: Ignored in XLA path (Triton-only).

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If mode is "affine" but biases is None.
        ValueError: If mode is not "affine" but biases is provided.
        ValueError: If parameters are invalid for the selected mode.

    Notes:
        - The blocked algorithm is used when bits is 4 or 8 and block sizes
          are compatible with group_size. Otherwise, falls back to
          dequantize+matmul.
        - This implementation serves as the fallback for the Triton backend
          when shapes or modes are unsupported.
    """
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires biases.")
    if mode != "affine" and biases is not None:
        raise ValueError("biases must be None for non-affine modes.")

    return _operate(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
    )
