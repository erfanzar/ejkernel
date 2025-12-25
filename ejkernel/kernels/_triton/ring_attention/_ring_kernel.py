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

"""Ring attention kernels for distributed ring topology.

Historically this module implemented ring attention on top of the Triton flash
attention kernel. That approach cannot express global positions/segment masking
for distributed causal/window attention.

The preferred implementation here uses Triton block-sparse attention with
explicit `q_positions/kv_positions` and `q_segment_ids/kv_segment_ids`, enabling
correct masking while still rotating KV blocks with `lax.ppermute`.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.ops import BwdParams, FwdParams

from ..blocksparse_attention._mask import create_sparsity_mask
from ..blocksparse_attention._triton_impl_bwd import _bwd_blocksparse_attn_call
from ..blocksparse_attention._triton_impl_fwd import _fwd_blocksparse_attn_call
from ..flash_attention._triton_impl_bwd import _bwd_attention_kernel_call
from ..flash_attention._triton_impl_fwd import _fwd_attention_kernel_call

# ln(2) constant for converting between log2 and natural log
LN2 = 0.6931471805599453


class RingFlashResiduals(NamedTuple):
    """Residuals saved from forward pass for backward computation."""

    q: jax.Array
    k: jax.Array
    v: jax.Array
    bias: jax.Array | None
    attention_mask: jax.Array | None
    o: jax.Array
    lse: jax.Array  # In natural log space
    dropout_seed: int | None


@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 13))
def ring_flash_attention_call(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    dropout_seed: int | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
) -> jax.Array:
    """Ring flash attention with custom VJP for efficient gradients.

    Args:
        query: Query tensor [batch, seq_len_q, num_heads, head_dim]
        key: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        value: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        attention_mask: Optional attention mask
        bias: Optional attention bias
        softmax_scale: Scale for attention scores
        dropout_prob: Dropout probability
        causal: Whether to use causal masking
        dropout_seed: Random seed for dropout
        fwd_params: Forward pass block size parameters
        bwd_params: Backward pass block size parameters
        sliding_window: Sliding window size
        logits_soft_cap: Soft cap value for logits
        axis_name: Name of axis for ring communication

    Returns:
        Output tensor [batch, seq_len_q, num_heads, head_dim]
    """
    o, _ = _ring_flash_attention_fwd(
        query,
        key,
        value,
        attention_mask,
        bias,
        softmax_scale,
        dropout_prob,
        causal,
        dropout_seed,
        fwd_params,
        bwd_params,
        sliding_window,
        logits_soft_cap,
        axis_name,
    )
    return o


def _ring_flash_attention_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    attention_mask: jax.Array | None,
    bias: jax.Array | None,
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    dropout_seed: int | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
) -> tuple[jax.Array, RingFlashResiduals]:
    """Forward pass of ring flash attention.

    Uses online softmax to combine attention outputs from different ring positions.
    """
    batch = query.shape[0]
    q_seq_len = query.shape[1]
    num_heads = query.shape[2]

    # Get ring size
    if axis_name is not None:
        axis_size = lax.psum(1, axis_name)
    else:
        axis_size = 1

    # Initialize accumulators
    o = jnp.zeros(query.shape, dtype=jnp.float32)
    lse = jnp.full((batch, num_heads, q_seq_len), -jnp.inf, dtype=jnp.float32)

    def scan_ring(carry, idx):
        o_acc, lse_acc, k_curr, v_curr = carry

        # Call flash attention forward kernel
        o_chunk, lse_chunk_log2 = _fwd_attention_kernel_call(
            q=query,
            k=k_curr,
            v=v_curr,
            attention_mask=attention_mask,
            bias=bias,
            softmax_scale=softmax_scale,
            dropout_prob=dropout_prob,
            causal=causal,
            dropout_seed=dropout_seed,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            cum_seqlens_q=None,
            cum_seqlens_k=None,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_aux=None,  # Attention sinks not supported in ring mode yet
        )

        # Convert LSE from log2 to natural log
        lse_chunk = lse_chunk_log2 * LN2
        # Handle padding: lse shape is (batch, heads, max_seqlen_q_rounded)
        lse_chunk = lse_chunk[..., :q_seq_len]

        # Online softmax combination
        lse_max = jnp.maximum(lse_acc, lse_chunk)
        alpha = jnp.exp(lse_acc - lse_max)
        beta = jnp.exp(lse_chunk - lse_max)
        sum_weights = alpha + beta

        # Avoid division by zero
        sum_weights_safe = jnp.where(sum_weights == 0, 1.0, sum_weights)

        # Update output with weighted combination
        # Transpose o_chunk to match lse shape broadcasting [batch, heads, seq] -> [batch, seq, heads]
        alpha_expanded = jnp.transpose(alpha, (0, 2, 1))[..., None]  # [batch, seq, heads, 1]
        beta_expanded = jnp.transpose(beta, (0, 2, 1))[..., None]
        sum_weights_expanded = jnp.transpose(sum_weights_safe, (0, 2, 1))[..., None]

        o_next = (alpha_expanded * o_acc + beta_expanded * o_chunk.astype(jnp.float32)) / sum_weights_expanded

        # Update log-sum-exp
        lse_next = lse_max + jnp.log(jnp.where(sum_weights == 0, 1.0, sum_weights))

        # Rotate K, V to next device in ring
        if axis_name is not None:
            perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
            k_next = lax.ppermute(k_curr, axis_name, perm)
            v_next = lax.ppermute(v_curr, axis_name, perm)
        else:
            k_next, v_next = k_curr, v_curr

        return (o_next, lse_next, k_next, v_next), None

    (o, lse, _, _), _ = lax.scan(scan_ring, (o, lse, key, value), jnp.arange(axis_size))
    o_out = o.astype(query.dtype)

    residuals = RingFlashResiduals(
        q=query,
        k=key,
        v=value,
        bias=bias,
        attention_mask=attention_mask,
        o=o_out,
        lse=lse,
        dropout_seed=dropout_seed,
    )

    return o_out, residuals


def _ring_flash_attention_bwd(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    dropout_seed: int | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
    res: RingFlashResiduals,
    do: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None]:
    """Backward pass of ring flash attention."""
    q, k, v, bias, attention_mask, o, lse, dropout_seed_res = res
    del dropout_seed_res  # Use the one from nondiff_argnums

    if axis_name is not None:
        axis_size = lax.psum(1, axis_name)
    else:
        axis_size = 1

    # Initialize gradient accumulators
    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)

    # Convert LSE back to log2 for backward kernel (it expects log2 space)
    lse_log2 = lse / LN2

    def scan_ring_bwd(carry, idx):
        dq_acc, dk_acc, dv_acc, k_curr, v_curr = carry

        # Compute gradients using flash attention backward kernel
        dq_chunk, dk_chunk, dv_chunk = _bwd_attention_kernel_call(
            dO=do,
            q=q,
            k=k_curr,
            v=v_curr,
            bias=bias,
            attention_mask=attention_mask,
            o=o,
            M=lse_log2,
            dropout_prob=dropout_prob,
            causal=causal,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            dropout_seed=dropout_seed,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            cum_seqlens_k=None,
            cum_seqlens_q=None,
            logits_soft_cap=logits_soft_cap,
        )

        dq_acc = dq_acc + dq_chunk.astype(jnp.float32)
        dk_acc = dk_acc + dk_chunk.astype(jnp.float32)
        dv_acc = dv_acc + dv_chunk.astype(jnp.float32)

        # Rotate K, V and their gradients
        if axis_name is not None:
            perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
            k_next = lax.ppermute(k_curr, axis_name, perm)
            v_next = lax.ppermute(v_curr, axis_name, perm)
            dk_acc = lax.ppermute(dk_acc, axis_name, perm)
            dv_acc = lax.ppermute(dv_acc, axis_name, perm)
        else:
            k_next, v_next = k_curr, v_curr

        return (dq_acc, dk_acc, dv_acc, k_next, v_next), None

    (dq, dk, dv, _, _), _ = lax.scan(scan_ring_bwd, (dq, dk, dv, k, v), jnp.arange(axis_size))

    # Cast back to input dtypes
    dq = dq.astype(q.dtype)
    dk = dk.astype(k.dtype)
    dv = dv.astype(v.dtype)

    return dq, dk, dv, None, None


ring_flash_attention_call.defvjp(_ring_flash_attention_fwd, _ring_flash_attention_bwd)


class RingBlocksparseResiduals(NamedTuple):
    """Residuals saved from forward pass for backward computation."""

    q: jax.Array  # [B, Hq, Tq, D]
    k: jax.Array  # [B, Hkv, Tk, D]
    v: jax.Array  # [B, Hkv, Tk, D]
    q_positions: jax.Array  # [B, Tq]
    q_segment_ids: jax.Array  # [B, Tq]
    kv_positions: jax.Array  # [B, Tk]
    kv_segment_ids: jax.Array  # [B, Tk]
    o: jax.Array  # [B, Hq, Tq, D]
    lse: jax.Array  # [B, Hq, Tq] in natural log space


def _window_to_bounds(sliding_window: int | tuple[int, int] | None) -> tuple[int, int]:
    if sliding_window is None:
        return -1, -1
    if isinstance(sliding_window, int):
        w = int(sliding_window)
        return w, w
    left, right = sliding_window
    return int(left), int(right)


def _build_positions(
    *,
    segment_ids: jax.Array,
    seq_len: int,
    pad_value: int,
    axis_name: str | None,
) -> jax.Array:
    """Build per-token positions (optionally globalized across a ring axis)."""
    segment_ids = jnp.asarray(segment_ids, jnp.int32)
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must be rank-2 (batch, seq_len), got shape {segment_ids.shape}")

    from ejkernel.types.mask import _positions_from_segments_2d

    positions = _positions_from_segments_2d(segment_ids, pad_value=pad_value).astype(jnp.int32)

    if axis_name is None:
        return positions

    axis_idx = lax.axis_index(axis_name)
    offset = jnp.asarray(axis_idx * seq_len, dtype=jnp.int32)
    if pad_value == -1:
        # Query padding positions are -1; keep them unchanged.
        positions = jnp.where(positions >= 0, positions + offset, positions)
    else:
        # KV padding positions are usually int32_max; keep them unchanged based on segment ids.
        valid = segment_ids >= 0
        positions = jnp.where(valid, positions + offset, positions)
    return positions


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14))
def ring_blocksparse_attention_call(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    q_position_ids: jax.Array | None,
    kv_position_ids: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
) -> jax.Array:
    o, _ = _ring_blocksparse_attention_fwd(
        query,
        key,
        value,
        q_segment_ids,
        kv_segment_ids,
        q_position_ids,
        kv_position_ids,
        softmax_aux,
        softmax_scale,
        causal,
        sliding_window,
        logits_soft_cap,
        axis_name,
        fwd_params,
        bwd_params,
    )
    return o


def _ring_blocksparse_attention_fwd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    q_position_ids: jax.Array | None,
    kv_position_ids: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
) -> tuple[jax.Array, RingBlocksparseResiduals]:
    batch, q_seq_len, num_heads, _ = query.shape
    kv_seq_len = key.shape[1]

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=fwd_params.q_blocksize, kv_blocksize=fwd_params.kv_blocksize)

    if q_segment_ids is None:
        q_segment_ids = jnp.zeros((batch, q_seq_len), dtype=jnp.int32)
    else:
        q_segment_ids = jnp.asarray(q_segment_ids, jnp.int32)
    if kv_segment_ids is None:
        kv_segment_ids = jnp.zeros((batch, kv_seq_len), dtype=jnp.int32)
    else:
        kv_segment_ids = jnp.asarray(kv_segment_ids, jnp.int32)

    if q_position_ids is None:
        q_positions = _build_positions(
            segment_ids=q_segment_ids,
            seq_len=q_seq_len,
            pad_value=-1,
            axis_name=axis_name,
        )
    else:
        q_positions = jnp.asarray(q_position_ids, jnp.int32)
        if q_positions.shape != (batch, q_seq_len):
            raise ValueError(
                f"q_position_ids must have shape (batch, seq_len_q)=({batch}, {q_seq_len}), got {q_positions.shape}"
            )

    if kv_position_ids is None:
        kv_positions = _build_positions(
            segment_ids=kv_segment_ids,
            seq_len=kv_seq_len,
            pad_value=jnp.iinfo(jnp.int32).max,
            axis_name=axis_name,
        )
    else:
        kv_positions = jnp.asarray(kv_position_ids, jnp.int32)
        if kv_positions.shape != (batch, kv_seq_len):
            raise ValueError(
                f"kv_position_ids must have shape (batch, seq_len_k)=({batch}, {kv_seq_len}), got {kv_positions.shape}"
            )

    q = jnp.transpose(query, (0, 2, 1, 3))
    k = jnp.transpose(key, (0, 2, 1, 3))
    v = jnp.transpose(value, (0, 2, 1, 3))

    if axis_name is not None:
        axis_size = lax.psum(1, axis_name)
        perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
    else:
        axis_size = 1
        perm = None

    window_left, window_right = _window_to_bounds(sliding_window)

    o_acc = jnp.zeros((batch, num_heads, q_seq_len, query.shape[-1]), dtype=jnp.float32)
    lse_acc = jnp.full((batch, num_heads, q_seq_len), -jnp.inf, dtype=jnp.float32)

    # Attention sinks contribute to normalization mass but not the output.
    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim == 1:
            sink_lse = jax.nn.logsumexp(aux)
            sink_lse = jnp.broadcast_to(sink_lse, (num_heads,))
        elif aux.ndim == 2:
            sink_lse = jax.nn.logsumexp(aux, axis=-1)
            if sink_lse.shape[0] == 1:
                sink_lse = jnp.broadcast_to(sink_lse[0], (num_heads,))
            elif sink_lse.shape[0] != num_heads:
                raise ValueError(f"softmax_aux first dim must be 1 or num_heads ({num_heads}); got {aux.shape[0]}")
        else:
            raise ValueError(f"softmax_aux must be 1D or 2D, got shape {aux.shape}")

        sink_lse = jnp.broadcast_to(sink_lse[None, :, None], (batch, num_heads, q_seq_len))
        lse_acc = jnp.where(jnp.isfinite(sink_lse), sink_lse, lse_acc)

    def scan_ring(carry, _idx):
        o_acc, lse_acc, k_curr, v_curr, kv_pos_curr, kv_seg_curr = carry

        qkv_layouts = create_sparsity_mask(
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            kv_positions=kv_pos_curr,
            kv_segment_ids=kv_seg_curr,
            q_blocksize=int(fwd_params.q_blocksize),
            kv_blocksize=int(fwd_params.kv_blocksize),
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )

        o_chunk, outputs_for_bwd_pass = _fwd_blocksparse_attn_call(
            query=q,
            key=k_curr,
            value=v_curr,
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            kv_positions=kv_pos_curr,
            kv_segment_ids=kv_seg_curr,
            qkv_layouts=qkv_layouts,
            softmax_scale=float(softmax_scale),
            softmax_aux=None,
            bias=None,
            apply_load_balance=True,
            sequence_parallelism_mesh_axis_name=None,
            window_left=window_left,
            window_right=window_right,
            causal=causal,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            logits_soft_cap=logits_soft_cap,
        )

        lse_chunk_log2 = outputs_for_bwd_pass[9]
        lse_chunk = lse_chunk_log2 * LN2

        lse_max = jnp.maximum(lse_acc, lse_chunk)
        alpha = jnp.where(jnp.isfinite(lse_acc), jnp.exp(lse_acc - lse_max), 0.0)
        beta = jnp.where(jnp.isfinite(lse_chunk), jnp.exp(lse_chunk - lse_max), 0.0)
        sum_weights = alpha + beta
        sum_weights_safe = jnp.where(sum_weights == 0.0, 1.0, sum_weights)

        alpha_e = alpha / sum_weights_safe
        beta_e = beta / sum_weights_safe

        o_next = alpha_e[..., None] * o_acc + beta_e[..., None] * o_chunk.astype(jnp.float32)
        lse_next = lse_max + jnp.log(sum_weights_safe)

        if axis_name is not None:
            k_next = lax.ppermute(k_curr, axis_name, perm)
            v_next = lax.ppermute(v_curr, axis_name, perm)
            kv_pos_next = lax.ppermute(kv_pos_curr, axis_name, perm)
            kv_seg_next = lax.ppermute(kv_seg_curr, axis_name, perm)
        else:
            k_next, v_next, kv_pos_next, kv_seg_next = k_curr, v_curr, kv_pos_curr, kv_seg_curr

        return (o_next, lse_next, k_next, v_next, kv_pos_next, kv_seg_next), None

    (o_acc, lse_acc, _, _, _, _), _ = lax.scan(
        scan_ring,
        init=(o_acc, lse_acc, k, v, kv_positions, kv_segment_ids),
        xs=jnp.arange(axis_size),
    )

    o_out = o_acc.astype(query.dtype)
    res = RingBlocksparseResiduals(
        q=q,
        k=k,
        v=v,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        o=o_out,
        lse=lse_acc,
    )
    return jnp.transpose(o_out, (0, 2, 1, 3)), res


def _ring_blocksparse_attention_bwd(
    q_segment_ids: jax.Array | None,
    kv_segment_ids: jax.Array | None,
    q_position_ids: jax.Array | None,
    kv_position_ids: jax.Array | None,
    softmax_aux: jax.Array | None,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    axis_name: str | None,
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
    res: RingBlocksparseResiduals,
    do: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    del q_segment_ids, kv_segment_ids, q_position_ids, kv_position_ids, softmax_aux
    (
        q,
        k,
        v,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        o,
        lse,
    ) = res

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=128, kv_blocksize=128, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=fwd_params.q_blocksize, kv_blocksize=fwd_params.kv_blocksize)

    if axis_name is not None:
        axis_size = lax.psum(1, axis_name)
        perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
    else:
        axis_size = 1
        perm = None

    window_left, window_right = _window_to_bounds(sliding_window)

    dq = jnp.zeros_like(q, dtype=jnp.float32)
    dk = jnp.zeros_like(k, dtype=jnp.float32)
    dv = jnp.zeros_like(v, dtype=jnp.float32)

    do_bhtd = jnp.transpose(do, (0, 2, 1, 3))

    lse_log2 = lse / LN2

    def scan_ring_bwd(carry, _idx):
        dq_acc, dk_acc, dv_acc, k_curr, v_curr, kv_pos_curr, kv_seg_curr = carry

        qkv_layouts = create_sparsity_mask(
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            kv_positions=kv_pos_curr,
            kv_segment_ids=kv_seg_curr,
            q_blocksize=int(bwd_params.q_blocksize),
            kv_blocksize=int(bwd_params.kv_blocksize),
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )

        step_res = (
            q,
            k_curr,
            v_curr,
            q_positions,
            q_segment_ids,
            kv_pos_curr,
            kv_seg_curr,
            qkv_layouts,
            o,
            lse_log2,
            None,
            None,
        )

        dq_chunk, dk_chunk, dv_chunk, *_ = _bwd_blocksparse_attn_call(
            softmax_scale=float(softmax_scale),
            apply_load_balance=True,
            sequence_parallelism_mesh_axis_name=None,
            window_left=window_left,
            window_right=window_right,
            causal=causal,
            fwd_params=fwd_params,
            bwd_params=bwd_params,
            logits_soft_cap=logits_soft_cap,
            res=step_res,
            dout=do_bhtd,
        )

        dq_acc = dq_acc + dq_chunk.astype(jnp.float32)
        dk_acc = dk_acc + dk_chunk.astype(jnp.float32)
        dv_acc = dv_acc + dv_chunk.astype(jnp.float32)

        if axis_name is not None:
            k_next = lax.ppermute(k_curr, axis_name, perm)
            v_next = lax.ppermute(v_curr, axis_name, perm)
            kv_pos_next = lax.ppermute(kv_pos_curr, axis_name, perm)
            kv_seg_next = lax.ppermute(kv_seg_curr, axis_name, perm)
            dk_acc = lax.ppermute(dk_acc, axis_name, perm)
            dv_acc = lax.ppermute(dv_acc, axis_name, perm)
        else:
            k_next, v_next, kv_pos_next, kv_seg_next = k_curr, v_curr, kv_pos_curr, kv_seg_curr

        return (dq_acc, dk_acc, dv_acc, k_next, v_next, kv_pos_next, kv_seg_next), None

    (dq, dk, dv, _, _, _, _), _ = lax.scan(
        scan_ring_bwd,
        init=(dq, dk, dv, k, v, kv_positions, kv_segment_ids),
        xs=jnp.arange(axis_size),
    )

    dq = jnp.transpose(dq.astype(q.dtype), (0, 2, 1, 3))
    dk = jnp.transpose(dk.astype(k.dtype), (0, 2, 1, 3))
    dv = jnp.transpose(dv.astype(v.dtype), (0, 2, 1, 3))

    return dq, dk, dv


ring_blocksparse_attention_call.defvjp(_ring_blocksparse_attention_fwd, _ring_blocksparse_attention_bwd)
