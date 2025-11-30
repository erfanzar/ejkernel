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

"""Ring Flash Attention Kernel - wraps Triton flash attention for distributed ring topology."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from ejkernel.ops import BwdParams, FwdParams

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
    o = jnp.zeros_like(query)
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

        o_next = (alpha_expanded * o_acc + beta_expanded * o_chunk) / sum_weights_expanded

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

    residuals = RingFlashResiduals(
        q=query,
        k=key,
        v=value,
        bias=bias,
        attention_mask=attention_mask,
        o=o,
        lse=lse,
        dropout_seed=dropout_seed,
    )

    return o, residuals


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
