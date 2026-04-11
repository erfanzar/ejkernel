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

Two-phase architecture for high MXU utilization on TPU v4:
  Phase 1 (parallel): Neumann inverse + state-independent quantities for ALL
           chunks simultaneously via a single pallas_call.
  Phase 2 (sequential): Lightweight lax.scan with only 4 matmuls per chunk.

Supports inference=True mode for faster forward-only execution.
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

from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm_with_inv, _recurrent_gdr_fwd

_P = lax.Precision.DEFAULT
_N_FUSE = 1


def _dot(a, b):
    """Matrix multiply with the module-level precision setting."""
    return lax.dot(a, b, precision=_P)


def _chunk_blockspec(shape: tuple[int, ...]) -> pl.BlockSpec:
    """Create a Pallas BlockSpec indexed by ``(batch, head)`` with remaining axes at 0."""
    return pl.BlockSpec(shape, lambda b, h: (b, h, *([0] * (len(shape) - 2))))


def _neumann_inv(A, C, strict_lower=None, lower_mask=None):
    """Compute (I - A)^{-1} via repeated squaring. Input must be pre-sanitized."""
    _hp = lax.Precision.HIGHEST
    # ``A`` is strict-lower triangular, so the Neumann series terminates
    # exactly after at most ``C - 1`` powers. Repeated squaring needs
    # ``ceil(log2(C))`` stages to materialize all terms up to ``A^(C-1)``.
    #
    # Clipping this to 4 only reconstructs powers through ``A^15``. That
    # makes partially-filled chunks numerically wrong once the valid prefix
    # exceeds ~16 tokens, which shows up most clearly on padded-heavy SFT
    # batches where the last chunk is only partially active.
    num_iters = math.ceil(math.log2(C)) if C > 1 else 0
    if strict_lower is None:
        strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.float32), k=-1)
    if lower_mask is None:
        lower_mask = strict_lower + jnp.eye(C, dtype=jnp.float32)
    S = jnp.eye(C, dtype=jnp.float32)
    P = jnp.where(strict_lower, A, 0.0)
    for _ in range(num_iters):
        S = jnp.where(lower_mask, S + lax.dot(P, S, precision=_hp), 0.0)
        P = jnp.where(strict_lower, lax.dot(P, P, precision=_hp), 0.0)
    return jnp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)


def _process_one_chunk(q, k, v, beta, decay, state, C):
    """Process a single chunk: returns (output, new_state). Used by backward."""
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)
    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]
    g_cumsum = jnp.sum(lower_mask * decay[None, :], axis=1, keepdims=True)
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask
    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)
    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))
    value_local = jnp.nan_to_num(_dot(attn_inv, v_beta), nan=0.0, posinf=0.0, neginf=0.0)
    k_cumdecay = jnp.nan_to_num(_dot(attn_inv, k_beta * g_cumsum_exp), nan=0.0, posinf=0.0, neginf=0.0)
    attn_qk = jnp.nan_to_num(_dot(q, k.T) * decay_mask, nan=0.0, posinf=0.0, neginf=0.0)
    q_scaled = q * g_cumsum_exp
    v_prime = jnp.nan_to_num(_dot(k_cumdecay, state), nan=0.0, posinf=0.0, neginf=0.0)
    attn_inter = jnp.nan_to_num(_dot(q_scaled, state), nan=0.0, posinf=0.0, neginf=0.0)
    v_new = jnp.nan_to_num(value_local - v_prime, nan=0.0, posinf=0.0, neginf=0.0)
    core_out = attn_inter + _dot(attn_qk, v_new)
    core_out = jnp.nan_to_num(core_out, nan=0.0, posinf=0.0, neginf=0.0)
    k_scaled = k * g_diff_state_exp
    new_state = state * g_end_exp + _dot(k_scaled.T, v_new)
    new_state = jnp.nan_to_num(new_state, nan=0.0, posinf=0.0, neginf=0.0)
    return core_out, new_state


def _phase1_kernel_infer(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    decay_mask_ref,
    g_cumsum_ref,
    value_local_ref,
    k_cumdecay_ref,
    attn_qk_ref,
    q_scaled_ref,
    k_scaled_ref,
    g_end_exp_ref,
):
    """Fast inference Phase 1: uses precomputed decay_mask + g_cumsum. No attn_inv output."""
    C = q_ref.shape[3]
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, 0, 0]
    decay_mask = decay_mask_ref[0, 0, 0].astype(jnp.float32)
    g_cumsum = g_cumsum_ref[0, 0, 0, 0]

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum[:, None], -20.0, 20.0))
    g_end_val = g_cumsum[-1:]
    g_end_exp = jnp.exp(jnp.clip(g_end_val, -20.0, 20.0)).reshape(1, 1)
    g_diff_state_exp = jnp.exp(jnp.clip(g_end_val[:, None] - g_cumsum[:, None], -20.0, 20.0))

    k_beta_scaled = k_beta * g_cumsum_exp
    combined_rhs = jnp.concatenate([v_beta, k_beta_scaled], axis=-1)
    combined_out = _dot(attn_inv, combined_rhs)
    V = v_beta.shape[-1]
    value_local = combined_out[:, :V]
    k_cumdecay = combined_out[:, V:]

    attn_qk = _dot(q, k.T) * decay_mask
    q_scaled = q * g_cumsum_exp
    k_scaled = k * g_diff_state_exp

    value_local_ref[0, 0, 0] = value_local.astype(value_local_ref.dtype)
    k_cumdecay_ref[0, 0, 0] = k_cumdecay.astype(k_cumdecay_ref.dtype)
    attn_qk_ref[0, 0, 0] = attn_qk.astype(attn_qk_ref.dtype)
    q_scaled_ref[0, 0, 0] = q_scaled.astype(q_scaled_ref.dtype)
    k_scaled_ref[0, 0, 0] = k_scaled.astype(k_scaled_ref.dtype)
    g_end_exp_ref[0, 0, 0] = g_end_exp.astype(g_end_exp_ref.dtype)


def _phase1_kernel_train(
    q_ref,
    k_ref,
    v_ref,
    beta_ref,
    decay_ref,
    value_local_ref,
    k_cumdecay_ref,
    attn_qk_ref,
    q_scaled_ref,
    k_scaled_ref,
    g_end_exp_ref,
    attn_inv_ref,
):
    """Training Phase 1: full computation + saves attn_inv for backward."""
    C = q_ref.shape[3]
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    beta = beta_ref[0, 0, 0, 0]
    decay = decay_ref[0, 0, 0, 0]

    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
    strict_lower = lower_mask - jnp.eye(C, dtype=jnp.float32)

    v_beta = v * beta[:, None]
    k_beta = k * beta[:, None]

    g_cumsum = jnp.sum(lower_mask * decay[None, :], axis=1, keepdims=True)
    g_diff = g_cumsum - g_cumsum.T
    decay_mask = jnp.exp(jnp.clip(g_diff * lower_mask, -20.0, 20.0)) * lower_mask

    attn_neg = -(_dot(k_beta, k.T) * decay_mask) * strict_lower
    attn_neg = jnp.nan_to_num(attn_neg, nan=0.0, posinf=0.0, neginf=0.0)
    attn_inv = _neumann_inv(attn_neg, C, strict_lower=strict_lower, lower_mask=lower_mask)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))
    g_end = g_cumsum[C - 1 : C, :]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0))
    g_diff_state_exp = jnp.exp(jnp.clip(g_end - g_cumsum, -20.0, 20.0))

    value_local = _dot(attn_inv, v_beta)
    k_cumdecay = _dot(attn_inv, k_beta * g_cumsum_exp)
    attn_qk = _dot(q, k.T) * decay_mask
    q_scaled = q * g_cumsum_exp
    k_scaled = k * g_diff_state_exp

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    value_local_ref[0, 0, 0] = _s(value_local).astype(value_local_ref.dtype)
    k_cumdecay_ref[0, 0, 0] = _s(k_cumdecay).astype(k_cumdecay_ref.dtype)
    attn_qk_ref[0, 0, 0] = _s(attn_qk).astype(attn_qk_ref.dtype)
    q_scaled_ref[0, 0, 0] = q_scaled.astype(q_scaled_ref.dtype)
    k_scaled_ref[0, 0, 0] = k_scaled.astype(k_scaled_ref.dtype)
    g_end_exp_ref[0, 0, 0] = jnp.broadcast_to(g_end_exp, (1, 1)).astype(g_end_exp_ref.dtype)
    attn_inv_ref[0, 0, 0] = attn_inv.astype(attn_inv_ref.dtype)


def _run_phase1(query_c, key_c, value_c, beta_c, decay_c, *, inference=False):
    """Launch Phase 1 parallel kernel for ALL chunks."""
    B, H, NC, C, K = query_c.shape
    V = value_c.shape[-1]

    def bs3(shape):
        return pl.BlockSpec((1, 1, 1, *shape), lambda b, h, c: (b, h, c, *([0] * len(shape))))

    if inference:
        decay_flat = decay_c.squeeze(-2)
        g_cumsum = jnp.cumsum(decay_flat, axis=-1)
        g_cs = g_cumsum[..., None]
        lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.float32))
        decay_mask = jnp.exp(jnp.clip((g_cs - g_cs.transpose(0, 1, 2, 4, 3)) * lower_mask, -20.0, 20.0)) * lower_mask
        g_cumsum_input = g_cumsum.reshape(B, H, NC, 1, C).astype(jnp.float32)

        call = pl.pallas_call(
            _phase1_kernel_infer,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((C, V)),
                    bs3((1, C)),
                    bs3((1, C)),
                    bs3((C, C)),
                    bs3((1, C)),
                ],
                out_specs=[
                    bs3((C, V)),
                    bs3((C, K)),
                    bs3((C, C)),
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((1, 1)),
                ],
                grid=(B, H, NC),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((B, H, NC, C, V), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, 1, 1), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
            ),
        )
        results = call(query_c, key_c, value_c, beta_c, decay_c, decay_mask, g_cumsum_input)
        return (*results, None)
    else:
        call = pl.pallas_call(
            _phase1_kernel_train,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((C, V)),
                    bs3((1, C)),
                    bs3((1, C)),
                ],
                out_specs=[
                    bs3((C, V)),
                    bs3((C, K)),
                    bs3((C, C)),
                    bs3((C, K)),
                    bs3((C, K)),
                    bs3((1, 1)),
                    bs3((C, C)),
                ],
                grid=(B, H, NC),
            ),
            out_shape=[
                jax.ShapeDtypeStruct((B, H, NC, C, V), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, K), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, 1, 1), jnp.float32),
                jax.ShapeDtypeStruct((B, H, NC, C, C), jnp.float32),
            ],
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "parallel", "arbitrary"),
            ),
        )
        return call(query_c, key_c, value_c, beta_c, decay_c)


def _phase2_scan_body(state, inputs):
    """Lightweight scan body: 4 matmuls + element-wise ops. No Neumann."""
    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp = inputs

    def _s(x):
        return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    v_prime = _s(jnp.einsum("bhck,bhkv->bhcv", k_cumdecay, state))
    attn_inter = _s(jnp.einsum("bhck,bhkv->bhcv", q_scaled, state))
    v_new = _s(value_local - v_prime)
    core_out = _s(attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_qk, v_new))

    state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new)
    new_state = _s(state * g_end_exp + state_update)

    return new_state, (core_out, state)


def _phase2_scan_body_infer(state, inputs):
    """Inference scan body: no nan_to_num on intermediates (only state carry)."""
    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp = inputs

    v_prime = jnp.einsum("bhck,bhkv->bhcv", k_cumdecay, state)
    attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_scaled, state)
    v_new = value_local - v_prime
    core_out = attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_qk, v_new)

    state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new)
    new_state = jnp.nan_to_num(state * g_end_exp + state_update, nan=0.0, posinf=0.0, neginf=0.0)

    return new_state, (core_out, state)


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
    inference: bool = False,
):
    """Two-phase chunked GDR forward.

    Args:
        inference: If True, uses faster kernels (precomputed decay_mask,
                   no attn_inv output, no intermediate nan_to_num).
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
    beta_c = beta.reshape(B, H, num_chunks, 1, chunk_size).astype(jnp.float32)
    decay_c = decay.reshape(B, H, num_chunks, 1, chunk_size).astype(jnp.float32)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    value_local, k_cumdecay, attn_qk, q_scaled, k_scaled, g_end_exp, _attn_inv = _run_phase1(
        query_c,
        key_c,
        value_c,
        beta_c,
        decay_c,
        inference=inference,
    )

    scan_inputs = (
        value_local.transpose(2, 0, 1, 3, 4),
        k_cumdecay.transpose(2, 0, 1, 3, 4),
        attn_qk.transpose(2, 0, 1, 3, 4),
        q_scaled.transpose(2, 0, 1, 3, 4),
        k_scaled.transpose(2, 0, 1, 3, 4),
        g_end_exp.transpose(2, 0, 1, 3, 4),
    )

    scan_fn = _phase2_scan_body_infer if inference else _phase2_scan_body
    final_state, (core_out_tm, state_pre_tm) = lax.scan(
        scan_fn,
        initial_state,
        scan_inputs,
    )

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
        beta_c.squeeze(-2),
        decay_c.squeeze(-2),
        state_pre_all,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
        chunk_size,
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
        inference=True,
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
    """Chunked forward pass for GDR on TPU via 2-phase Pallas kernel."""
    return _chunk_gdr_fwd_impl(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm)


def _chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Forward rule: training mode (saves residuals for backward)."""
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
        inference=False,
    )
    return (output, final_state), residual


def _chunk_gdr_bwd_rule(chunk_size, use_qk_l2norm, res, g):
    """Backward rule for ``custom_vjp``."""
    from ._pallas_impl_bwd import _chunk_gdr_bwd

    return _chunk_gdr_bwd(chunk_size, use_qk_l2norm, res, g)


_chunk_gdr_fwd.defvjp(_chunk_gdr_fwd_rule, _chunk_gdr_bwd_rule)


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
    """Exact multi-token chunked GDR path for TPU.

    Keep the optimized Pallas single-token decode kernel, but route the
    unstable multi-token training/prefill path through the exact chunked
    triangular-solve implementation.
    """
    return _recurrent_gdr_fwd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        chunk_size=chunk_size,
    )


def _gdr_single_step_fwd_kernel(q_ref, k_ref, v_ref, beta_ref, decay_ref, state_ref, out_ref, final_state_ref):
    q_t = q_ref[0, 0, 0].astype(jnp.float32)
    k_t = k_ref[0, 0, 0].astype(jnp.float32)
    v_t = v_ref[0, 0, 0].astype(jnp.float32)
    beta_t = beta_ref[0, 0].reshape(())
    g_exp = jnp.exp(jnp.clip(decay_ref[0, 0].reshape(()), -20.0, 20.0))
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
    beta = beta[..., None].astype(jnp.float32)
    decay = decay[..., None].astype(jnp.float32)
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
            out_specs=[_chunk_blockspec((1, 1, 1, v_dim)), _chunk_blockspec((1, 1, qk_dim, v_dim))],
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
    residual = (query, key, value, beta, decay, recurrent_state, q_inv_norm, k_inv_norm, decay_was_none)
    return output, final_state, residual


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _single_step_gdr_fwd(query, key, value, beta, decay, recurrent_state, use_qk_l2norm=True):
    output, final_state, _ = _single_step_gdr_fwd_impl(query, key, value, beta, decay, recurrent_state, use_qk_l2norm)
    return output, final_state


def _single_step_gdr_fwd_rule(query, key, value, beta, decay, recurrent_state, use_qk_l2norm):
    output, final_state, residual = _single_step_gdr_fwd_impl(
        query, key, value, beta, decay, recurrent_state, use_qk_l2norm
    )
    return (output, final_state), residual


def _single_step_gdr_bwd_rule(use_qk_l2norm, res, g):
    from ._pallas_impl_bwd import _single_step_gdr_bwd

    return _single_step_gdr_bwd(use_qk_l2norm, res, g)


_single_step_gdr_fwd.defvjp(_single_step_gdr_fwd_rule, _single_step_gdr_bwd_rule)
