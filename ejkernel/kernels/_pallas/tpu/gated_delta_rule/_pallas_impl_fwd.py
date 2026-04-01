# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Forward TPU Pallas kernels for Gated Delta Rule (GDR).

Implements a fused Pallas kernel that performs both the Neumann-series
preprocessing and the inter-chunk scan step entirely in VMEM, maximizing
MXU utilization and minimizing HBM round-trips.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array, Float

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_with_inv

_P = lax.Precision.DEFAULT


def _dot(a, b):
    """Matrix multiply with the module-level precision setting."""
    return lax.dot(a, b, precision=_P)


def _chunk_blockspec(shape: tuple[int, ...]) -> pl.BlockSpec:
    """Create a Pallas BlockSpec indexed by ``(batch, head)`` with remaining axes at 0."""
    return pl.BlockSpec(shape, lambda b, h: (b, h, *([0] * (len(shape) - 2))))


def _neumann_inv(A, C):
    """Compute (I - A)^{-1} for strict lower triangular A via repeated squaring.

    Uses the Neumann series identity with doubling:
      S = (I + A)(I + A^2)(I + A^4) ... in O(log C) iterations.
    All matmuls use HIGHEST precision for numerical stability.
    """
    num_iters = math.ceil(math.log2(C)) if C > 1 else 0
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), k=-1)
    unit_lower = strict_lower + jnp.eye(C, dtype=jnp.float32)
    S = jnp.eye(C, dtype=jnp.float32)
    P = jnp.where(strict_lower > 0.0, A, 0.0)
    P = jnp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(num_iters):
        S = S + _dot(P, S)
        S = jnp.where(unit_lower > 0.0, S, 0.0)
        S = jnp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        P = _dot(P, P)
        P = jnp.where(strict_lower > 0.0, P, 0.0)
        P = jnp.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    return S


def _gdr_fused_fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    state_ref,
    out_ref,
    final_state_ref,
):
    """Fused forward: Neumann-series preprocessing + scan step in VMEM."""
    C = q_ref.shape[2]

    q = q_ref[0, 0].astype(jnp.float32)
    k = k_ref[0, 0].astype(jnp.float32)
    v = v_ref[0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, :, 0].astype(jnp.float32)
    decay = decay_ref[0, 0, :, 0].astype(jnp.float32)
    state = state_ref[0, 0].astype(jnp.float32)
    state = jnp.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cumsum = _dot(lower_mask, decay)
    g_diff = g_cumsum[:, None] - g_cumsum[None, :]
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C)
    attn_inv = jnp.nan_to_num(attn_inv, nan=0.0, posinf=0.0, neginf=0.0)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    value_local = _dot(attn_inv, v_beta)
    value_local = jnp.nan_to_num(value_local, nan=0.0, posinf=0.0, neginf=0.0)
    k_beta_scaled = k_beta * g_cumsum_exp[:, None]
    k_cumdecay = _dot(attn_inv, k_beta_scaled)
    k_cumdecay = jnp.nan_to_num(k_cumdecay, nan=0.0, posinf=0.0, neginf=0.0)

    attn_qk = _dot(q, k.T) * decay_mask
    attn_qk = jnp.where(lower_mask > 0.0, attn_qk, 0.0)
    attn_qk = jnp.nan_to_num(attn_qk, nan=0.0, posinf=0.0, neginf=0.0)
    q_scaled = q * g_cumsum_exp[:, None]
    v_prime = _dot(k_cumdecay, state)
    v_prime = jnp.nan_to_num(v_prime, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inter = _dot(q_scaled, state)
    attn_inter = jnp.nan_to_num(attn_inter, nan=0.0, posinf=0.0, neginf=0.0)
    v_new = value_local - v_prime
    v_new = jnp.nan_to_num(v_new, nan=0.0, posinf=0.0, neginf=0.0)
    core_out = attn_inter + _dot(attn_qk, v_new)
    core_out = jnp.nan_to_num(core_out, nan=0.0, posinf=0.0, neginf=0.0)

    state_decayed = state * g_end_exp
    state_decayed = jnp.nan_to_num(state_decayed, nan=0.0, posinf=0.0, neginf=0.0)
    k_scaled = k * g_diff_state_exp[:, None]
    state_update = _dot(k_scaled.T, v_new)
    state_update = jnp.nan_to_num(state_update, nan=0.0, posinf=0.0, neginf=0.0)
    new_state = state_decayed + state_update
    new_state = jnp.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)

    out_ref[0, 0] = core_out.astype(out_ref.dtype)
    final_state_ref[0, 0] = new_state.astype(final_state_ref.dtype)


def _run_fused_fwd_step(q_chunk, k_chunk, v_chunk, beta_chunk, decay_chunk, state):
    """Launch the fused Pallas kernel for one chunk and return (output, new_state)."""
    bsz, num_heads, C, qk_dim = q_chunk.shape
    v_dim = v_chunk.shape[-1]
    call = pl.pallas_call(
        _gdr_fused_fwd_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, qk_dim)),
                _chunk_blockspec((1, 1, C, v_dim)),
                _chunk_blockspec((1, 1, C, 1)),
                _chunk_blockspec((1, 1, C, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, C, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, C, v_dim), q_chunk.dtype),
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(q_chunk, k_chunk, v_chunk, beta_chunk, decay_chunk, state)


def _chunk_gdr_fwd_core(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    *,
    save_residual: bool,
):
    """Core chunked GDR forward: chunk inputs, scan over chunks, collect outputs.

    Optionally saves residuals (pre-scan states, chunked inputs, L2-norm
    inverses) for the backward pass.

    Returns:
        ``(output, final_state, residual)`` where *residual* is None when
        ``save_residual=False``.
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    decay_was_none = decay is None
    initial_state_was_none = initial_state is None

    q_inv_norm = k_inv_norm = None
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)

    pad_size = (chunk_size - L % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    num_chunks = (L + pad_size) // chunk_size
    scale = 1.0 / math.sqrt(K_dim)
    query = query * scale

    query_c = query.reshape(B, H, num_chunks, chunk_size, K_dim)
    key_c = key.reshape(B, H, num_chunks, chunk_size, K_dim)
    value_c = value.reshape(B, H, num_chunks, chunk_size, V_dim)
    beta_c = beta.reshape(B, H, num_chunks, chunk_size, 1)
    decay_c = decay.reshape(B, H, num_chunks, chunk_size, 1)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    xs = (
        query_c.transpose(2, 0, 1, 3, 4),
        key_c.transpose(2, 0, 1, 3, 4),
        value_c.transpose(2, 0, 1, 3, 4),
        beta_c.transpose(2, 0, 1, 3, 4),
        decay_c.transpose(2, 0, 1, 3, 4),
    )

    def chunk_step(state, inputs):
        """Single scan step: run fused kernel on one chunk, return new state."""
        q_i, k_i, v_i, b_i, d_i = inputs
        core_out, new_state = _run_fused_fwd_step(q_i, k_i, v_i, b_i, d_i, state)
        return new_state, (core_out, state)

    final_state, (core_out_tm, state_pre_tm) = lax.scan(chunk_step, initial_state, xs)

    core_attn_out = core_out_tm.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)[:, :, :L, :]

    final_state_out = final_state.astype(input_dtype)

    if not save_residual:
        return core_attn_out, final_state_out, None

    state_pre_all = state_pre_tm.transpose(1, 2, 0, 3, 4)

    residual = (
        query_c,
        key_c,
        value_c,
        beta_c[:, :, :, :, 0],
        decay_c[:, :, :, :, 0],
        state_pre_all,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    )
    return core_attn_out, final_state_out, residual


def _chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Inference-only chunked GDR forward (no residuals saved)."""
    output, final_state, _ = _chunk_gdr_fwd_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=False,
    )
    return output, final_state


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 7))
def _chunk_gdr_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    chunk_size: int = 64,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked forward pass for Gated Delta Rule on TPU via fused Pallas kernel."""
    return _chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm)


def _chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Forward rule for ``custom_vjp``: run forward with residual saving."""
    output, final_state, residual = _chunk_gdr_fwd_core(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
        save_residual=True,
    )
    return (output, final_state), residual


def _chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, res, g):
    """Backward rule for ``custom_vjp``: delegates to the XLA analytical backward."""
    from ._pallas_impl_bwd import _chunk_gdr_bwd

    return _chunk_gdr_bwd(chunk_size, use_qk_l2norm, res, g)


_chunk_gdr_fwd.defvjp(_chunk_gdr_fwd_rule, _chunk_gdr_bwd_rule)


def _gdr_single_step_fwd_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, final_state_ref):
    """Pallas kernel body for a single-token GDR update: decay state, apply delta rule, write output."""
    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0, 0].astype(jnp.float32)[0]
    g_exp = jnp.exp(jnp.clip(decay_ref[0, 0, 0].astype(jnp.float32)[0], -20.0, 20.0))
    state_prev = state_ref[0, 0].astype(jnp.float32)
    state_decayed = state_prev * g_exp
    kv_mem = jnp.sum(state_decayed * k_t[:, None], axis=0)
    delta = (v_t - kv_mem) * beta_t
    state = state_decayed + k_t[:, None] * delta[None, :]
    out = jnp.sum(state * q_t[:, None], axis=0)
    out_ref[0, 0, 0] = out.astype(out_ref.dtype)
    final_state_ref[0, 0] = state.astype(final_state_ref.dtype)


def _run_single_step_forward(query, key, value, beta, decay, recurrent_state):
    bsz, num_heads, _, qk_dim = query.shape
    v_dim = value.shape[-1]
    beta = beta[..., None]
    decay = decay[..., None]
    call = pl.pallas_call(
        _gdr_single_step_fwd_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, qk_dim)),
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, 1, 1)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            out_specs=[
                _chunk_blockspec((1, 1, 1, v_dim)),
                _chunk_blockspec((1, 1, qk_dim, v_dim)),
            ],
            grid=(bsz, num_heads),
        ),
        out_shape=[
            jax.ShapeDtypeStruct((bsz, num_heads, 1, v_dim), query.dtype),
            jax.ShapeDtypeStruct((bsz, num_heads, qk_dim, v_dim), recurrent_state.dtype),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
    )
    return call(query, key, value, beta, decay, recurrent_state)


def _single_step_gdr_fwd_impl(query, key, value, beta, decay, recurrent_state, use_qk_l2norm):
    """Single-step forward with optional L2-norm, saving residuals for backward."""
    input_dtype = query.dtype
    decay_was_none = decay is None
    q_inv_norm = k_inv_norm = None
    query = query.astype(input_dtype)
    key = key.astype(input_dtype)
    value = value.astype(input_dtype)
    beta = beta.astype(input_dtype)
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)
    scale = 1.0 / math.sqrt(query.shape[-1])
    query = query * scale
    if decay is None:
        decay = jnp.zeros(beta.shape, dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)
    recurrent_state = recurrent_state.astype(input_dtype)
    output, final_state = _run_single_step_forward(query, key, value, beta, decay, recurrent_state)
    residual = (
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        q_inv_norm,
        k_inv_norm,
        decay_was_none,
    )
    return output, final_state, residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _single_step_gdr_fwd(
    query: Float[Array, "batch num_heads 1 head_dim"],
    key: Float[Array, "batch num_heads 1 head_dim"],
    value: Float[Array, "batch num_heads 1 d_state"],
    beta: Float[Array, "batch num_heads 1"],
    decay: Float[Array, "batch num_heads 1"] | None,
    recurrent_state: Float[Array, "batch num_heads head_dim d_state"],
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch num_heads 1 d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Single-step forward pass for Gated Delta Rule on TPU via Pallas."""
    output, final_state, _ = _single_step_gdr_fwd_impl(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        use_qk_l2norm,
    )
    return output, final_state


def _single_step_gdr_fwd_rule(query, key, value, beta, decay, recurrent_state, use_qk_l2norm):
    """Forward rule for single-step ``custom_vjp``: run forward with residuals."""
    output, final_state, residual = _single_step_gdr_fwd_impl(
        query,
        key,
        value,
        beta,
        decay,
        recurrent_state,
        use_qk_l2norm,
    )
    return (output, final_state), residual


def _single_step_gdr_bwd_rule(use_qk_l2norm, res, g):
    """Backward rule for single-step ``custom_vjp``: delegates to analytical backward."""
    from ._pallas_impl_bwd import _single_step_gdr_bwd

    return _single_step_gdr_bwd(use_qk_l2norm, res, g)


_single_step_gdr_fwd.defvjp(_single_step_gdr_fwd_rule, _single_step_gdr_bwd_rule)
