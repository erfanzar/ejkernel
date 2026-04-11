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

"""Forward pass implementations for Gated Delta Rule (GDR) linear attention.

This module provides three forward pass variants for GDR:

1. Recurrent (_recurrent_gdr_fwd):
   Pure sequential scan with O(L) time complexity. Best for very long
   sequences or memory-constrained inference.

2. Chunked (_chunk_gdr_fwd):
   Hybrid approach with parallel intra-chunk computation (Neumann series)
   and sequential inter-chunk state propagation. Includes a custom VJP
   with an analytical backward pass for numerical stability.

3. Single-step (_single_step_gdr_fwd):
   Optimized path for seq_len=1 during autoregressive inference.

The GDR update rule:
    h_t = exp(decay_t) * h_{t-1} + k_t (x) (beta_t * (v_t - h_{t-1} @ k_t))
    o_t = h_t @ q_t

Where h_t is the [head_dim, d_state] memory matrix that stores key-value
associations and supports efficient retrieval via query projection.

References:
    - Qwen3Next: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Float

_MATMUL_PRECISION = lax.Precision.HIGHEST


def _l2norm(x: Float[Array, "..."], axis: int = -1, eps: float = 1e-6) -> Float[Array, "..."]:
    """Apply L2 normalization along specified axis.

    Args:
        x: Input tensor to normalize.
        axis: Axis along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        L2-normalized tensor with same shape as input.
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _l2norm_with_inv(
    x: Float[Array, "..."], axis: int = -1, eps: float = 1e-6
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Returns both normalized tensor and inverse norm.

    Args:
        x: Input tensor to normalize.
        axis: Axis along which to normalize (default: -1).
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized tensor, inverse norm).
    """
    inv_norm = lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm, inv_norm


def _l2norm_bwd(grad_y, y, inv_norm):
    """Backward pass for y = l2norm(x).

    Args:
        grad_y: Upstream gradient.
        y: Normalized output from forward pass.
        inv_norm: Inverse norm saved from forward pass.

    Returns:
        Gradient with respect to x.
    """
    proj = jnp.sum(grad_y * y, axis=-1, keepdims=True)
    return inv_norm * (grad_y - y * proj)


def _strict_lower_inverse(matrix_strict_lower: jax.Array) -> jax.Array:
    """Compute inverse of (I + L) where L is strict-lower triangular."""
    n = int(matrix_strict_lower.shape[-1])
    eye = jnp.eye(n, dtype=matrix_strict_lower.dtype)
    flat = matrix_strict_lower.reshape((-1, n, n))

    def _solve_one(lhs):
        return lax.linalg.triangular_solve(
            lhs,
            eye,
            left_side=True,
            lower=True,
            unit_diagonal=True,
        )

    inv_flat = jax.vmap(_solve_one)(flat)
    return inv_flat.reshape(matrix_strict_lower.shape)


def _recurrent_gdr_fwd(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch num_heads seq_len head_dim"],
    value: Float[Array, "batch num_heads seq_len d_state"],
    beta: Float[Array, "batch num_heads seq_len"],
    decay: Float[Array, "batch num_heads seq_len"] | None,
    initial_state: Float[Array, "batch num_heads head_dim d_state"] | None = None,
    use_qk_l2norm: bool = True,
    chunk_size: int = 64,
) -> tuple[
    Float[Array, "batch num_heads seq_len d_state"],
    Float[Array, "batch num_heads head_dim d_state"],
]:
    """Chunked forward pass for Gated Delta Rule using triangular solve.

    Processes the sequence in chunks, computing intra-chunk attention in
    parallel via MXU matmuls and propagating state across chunks with a
    lightweight lax.scan (4 matmuls per chunk).

    Uses ``jax.scipy.linalg.solve_triangular`` for the exact (I+S)^{-1}
    computation instead of Neumann approximation.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, d_state].
        beta: Gating tensor [batch, num_heads, seq_len].
        decay: Per-token decay [batch, num_heads, seq_len].
        initial_state: Optional initial recurrent state
            [batch, num_heads, head_dim, d_state].
        use_qk_l2norm: Whether to L2-normalize queries and keys.
        chunk_size: Number of tokens per chunk for parallel processing.

    Returns:
        Tuple of (outputs, final_state).
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    C = chunk_size

    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    scale = 1.0 / (K_dim**0.5)
    query = (query * scale).astype(input_dtype)
    key = key.astype(input_dtype)
    value = value.astype(input_dtype)
    beta = beta.astype(input_dtype)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=jnp.float32)
    else:
        initial_state = initial_state.astype(jnp.float32)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=input_dtype)
    else:
        decay = decay.astype(input_dtype)

    pad_size = (C - L % C) % C
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    NC = (L + pad_size) // C

    q_c = query.reshape(B, H, NC, C, K_dim)
    k_c = key.reshape(B, H, NC, C, K_dim)
    v_c = value.reshape(B, H, NC, C, V_dim)
    beta_c = beta.reshape(B, H, NC, C)
    g_c = decay.reshape(B, H, NC, C)

    g_cumsum = jnp.cumsum(g_c, axis=-1)

    k_beta = k_c * beta_c[..., None]
    v_beta = v_c * beta_c[..., None]

    S = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, k_c, precision=_MATMUL_PRECISION).astype(jnp.float32)

    g_diff = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    strict_lower = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_), k=-1)
    lower_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    g_diff = jnp.where(strict_lower, g_diff, -1e30)
    S = jnp.where(strict_lower, S * jnp.exp(jnp.clip(g_diff, -20.0, 20.0)), 0.0)

    eye = jnp.eye(C, dtype=jnp.float32)
    lhs = jnp.broadcast_to(eye, S.shape) + S

    lhs_flat = lhs.reshape(-1, C, C)
    jnp.broadcast_to(eye, lhs_flat.shape)

    def _solve_one(m):
        return jax.scipy.linalg.solve_triangular(m, eye, lower=True, unit_diagonal=True)

    A_flat = jax.vmap(_solve_one)(lhs_flat)
    A = A_flat.reshape(B, H, NC, C, C)

    u_chunks = jnp.einsum("bhcij,bhcjv->bhciv", A, v_beta.astype(jnp.float32), precision=_MATMUL_PRECISION).astype(
        input_dtype
    )

    k_beta_g = k_beta.astype(jnp.float32) * jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))[..., None]
    w_chunks = jnp.einsum("bhcij,bhcjk->bhcik", A, k_beta_g, precision=_MATMUL_PRECISION).astype(input_dtype)

    attn_qk = jnp.einsum("bhcik,bhcjk->bhcij", q_c, k_c, precision=_MATMUL_PRECISION).astype(jnp.float32)
    g_diff_intra = g_cumsum[..., :, None] - g_cumsum[..., None, :]
    g_diff_intra = jnp.where(lower_mask, g_diff_intra, -1e30)
    attn_i = jnp.where(lower_mask, attn_qk * jnp.exp(jnp.clip(g_diff_intra, -20.0, 20.0)), 0.0).astype(input_dtype)

    q_g = (q_c.astype(jnp.float32) * jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0))[..., None]).astype(input_dtype)
    g_end_exp = jnp.exp(jnp.clip(g_cumsum[..., -1], -20.0, 20.0))[..., None, None]
    g_diff_state = jnp.exp(jnp.clip(g_cumsum[..., -1, None] - g_cumsum, -20.0, 20.0))[..., None]
    k_g_diff = (k_c.astype(jnp.float32) * g_diff_state).astype(input_dtype)

    xs = (
        u_chunks.transpose(2, 0, 1, 3, 4),
        w_chunks.transpose(2, 0, 1, 3, 4),
        q_g.transpose(2, 0, 1, 3, 4),
        attn_i.transpose(2, 0, 1, 3, 4),
        g_end_exp.transpose(2, 0, 1, 3, 4),
        k_g_diff.transpose(2, 0, 1, 3, 4),
    )

    def scan_body(state, inputs):
        u, w, q_scaled, attn_intra, g_last_exp, k_scaled = inputs

        v_prime = jnp.einsum("bhck,bhkv->bhcv", w, state, precision=_MATMUL_PRECISION)
        attn_inter = jnp.einsum("bhck,bhkv->bhcv", q_scaled, state, precision=_MATMUL_PRECISION)
        v_new = u.astype(jnp.float32) - v_prime
        core_out = attn_inter + jnp.einsum("bhcr,bhrv->bhcv", attn_intra, v_new, precision=_MATMUL_PRECISION)

        state_update = jnp.einsum("bhkc,bhcv->bhkv", k_scaled.transpose(0, 1, 3, 2), v_new, precision=_MATMUL_PRECISION)
        new_state = jnp.nan_to_num(state * g_last_exp + state_update, nan=0.0, posinf=0.0, neginf=0.0)

        return new_state, core_out

    final_state, core_out_tm = lax.scan(scan_body, initial_state, xs)

    core_out = core_out_tm.transpose(1, 2, 0, 3, 4)
    outputs = core_out.reshape(B, H, -1, V_dim)[:, :, :L, :].astype(input_dtype)

    return outputs, final_state.astype(input_dtype)


def _chunk_gdr_fwd_impl(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
):
    """Core implementation for chunked gated delta rule forward.

    This function is shared by the custom_vjp wrapper. Forward executes in
    ``input_dtype`` with Neumann-series accumulation in float32.
    """
    output, final_state, _ = _chunk_gdr_fwd_core(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        chunk_size=chunk_size,
        initial_state=initial_state,
        use_qk_l2norm=use_qk_l2norm,
        save_residual=False,
    )
    return output, final_state


def _chunk_gdr_fwd_core(
    query,
    key,
    value,
    beta,
    decay,
    chunk_size,
    initial_state,
    use_qk_l2norm,
    save_residual: bool,
):
    """Shared chunked forward path, optionally capturing backward residuals.

    This is the core Neumann-series-based chunked GDR computation. It processes
    the sequence in chunks of ``chunk_size``, computing intra-chunk attention in
    parallel and propagating state across chunks via lax.scan.

    Args:
        query: [batch, num_heads, seq_len, head_dim].
        key: [batch, num_heads, seq_len, head_dim].
        value: [batch, num_heads, seq_len, d_state].
        beta: [batch, num_heads, seq_len].
        decay: [batch, num_heads, seq_len] or None.
        chunk_size: Size of each chunk.
        initial_state: [batch, num_heads, head_dim, d_state] or None.
        use_qk_l2norm: Whether to L2-normalize queries and keys.
        save_residual: If True, save tensors needed for backward pass.

    Returns:
        Tuple of (output, final_state, residuals_or_none).
    """
    B, H, L, K_dim = query.shape
    V_dim = value.shape[-1]
    input_dtype = query.dtype
    decay_was_none = decay is None
    initial_state_was_none = initial_state is None

    q_inv_norm = None
    k_inv_norm = None
    if use_qk_l2norm:
        query, q_inv_norm = _l2norm_with_inv(query, axis=-1, eps=1e-6)
        key, k_inv_norm = _l2norm_with_inv(key, axis=-1, eps=1e-6)

    if decay is None:
        decay = jnp.zeros((B, H, L), dtype=jnp.float32)
    else:
        decay = decay.astype(jnp.float32)

    pad_size = (chunk_size - L % chunk_size) % chunk_size
    if pad_size > 0:
        query = jnp.pad(query, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        key = jnp.pad(key, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        value = jnp.pad(value, ((0, 0), (0, 0), (0, pad_size), (0, 0)))
        beta = jnp.pad(beta, ((0, 0), (0, 0), (0, pad_size)))
        decay = jnp.pad(decay, ((0, 0), (0, 0), (0, pad_size)))

    total_len = L + pad_size
    num_chunks = total_len // chunk_size

    scale = 1.0 / (K_dim**0.5)
    query = query * scale

    v_beta = value * beta[:, :, :, None]
    k_beta = key * beta[:, :, :, None]

    query = query.reshape(B, H, num_chunks, chunk_size, K_dim)
    key = key.reshape(B, H, num_chunks, chunk_size, K_dim)
    value = value.reshape(B, H, num_chunks, chunk_size, V_dim)
    beta = beta.reshape(B, H, num_chunks, chunk_size)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, K_dim)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, V_dim)
    g = decay.reshape(B, H, num_chunks, chunk_size)

    mask_triu = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=0)

    g_cumsum = jnp.cumsum(g, axis=-1)

    g_diff = g_cumsum[:, :, :, :, None] - g_cumsum[:, :, :, None, :]
    g_diff = jnp.tril(g_diff)
    decay_mask = jnp.exp(jnp.clip(g_diff, -20.0, 20.0))
    decay_mask = jnp.tril(decay_mask)

    attn = jnp.einsum("bhcik,bhcjk->bhcij", k_beta, key, precision=_MATMUL_PRECISION)
    attn = -(attn * decay_mask).astype(jnp.float32)
    attn = jnp.where(mask_triu, 0.0, attn)

    attn = jnp.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

    inv = _strict_lower_inverse(-attn)

    attn = jnp.nan_to_num(inv, nan=0.0, posinf=0.0, neginf=0.0).astype(input_dtype)

    g_cumsum_exp = jnp.exp(jnp.clip(g_cumsum, -20.0, 20.0)).astype(input_dtype)
    g_end = g_cumsum[:, :, :, -1]
    g_end_exp = jnp.exp(jnp.clip(g_end, -20.0, 20.0)).astype(input_dtype)
    g_diff_state_exp = jnp.exp(jnp.clip(g_end[:, :, :, None] - g_cumsum, -20.0, 20.0)).astype(input_dtype)

    value_local = jnp.einsum("bhcij,bhcjv->bhciv", attn, v_beta, precision=_MATMUL_PRECISION)
    k_beta_scaled = k_beta * g_cumsum_exp[:, :, :, :, None]
    k_cumdecay = jnp.einsum("bhcij,bhcjk->bhcik", attn, k_beta_scaled, precision=_MATMUL_PRECISION)

    if initial_state is None:
        initial_state = jnp.zeros((B, H, K_dim, V_dim), dtype=input_dtype)
    else:
        initial_state = initial_state.astype(input_dtype)

    mask_triu_inner = jnp.triu(jnp.ones((chunk_size, chunk_size), dtype=bool), k=1)

    xs = (
        query.transpose(2, 0, 1, 3, 4),
        key.transpose(2, 0, 1, 3, 4),
        value_local.transpose(2, 0, 1, 3, 4),
        k_cumdecay.transpose(2, 0, 1, 3, 4),
        g_cumsum_exp.transpose(2, 0, 1, 3),
        g_end_exp.transpose(2, 0, 1),
        g_diff_state_exp.transpose(2, 0, 1, 3),
        decay_mask.astype(input_dtype).transpose(2, 0, 1, 3, 4),
    )

    def chunk_step(state, inputs):
        q_i, k_i, v_i, k_cumdecay_i, g_exp_i, g_end_exp_i, g_diff_exp_i, decay_mask_i = inputs

        attn_qk = jnp.einsum("bhik,bhjk->bhij", q_i, k_i, precision=_MATMUL_PRECISION)
        attn_qk = attn_qk * decay_mask_i
        attn_qk = jnp.where(mask_triu_inner, 0.0, attn_qk)

        q_scaled = q_i * g_exp_i[:, :, :, None]
        qk_fused = jnp.stack([k_cumdecay_i, q_scaled], axis=0)
        both = jnp.einsum("nbhik,bhkv->nbhiv", qk_fused, state, precision=_MATMUL_PRECISION)
        v_prime = jnp.nan_to_num(both[0], nan=0.0, posinf=0.0, neginf=0.0)
        attn_inter = jnp.nan_to_num(both[1], nan=0.0, posinf=0.0, neginf=0.0)

        v_new = jnp.nan_to_num(v_i - v_prime, nan=0.0, posinf=0.0, neginf=0.0)
        core_out = jnp.nan_to_num(
            attn_inter + jnp.einsum("bhij,bhjv->bhiv", attn_qk, v_new, precision=_MATMUL_PRECISION),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        state_decayed = state * g_end_exp_i[:, :, None, None]
        k_scaled = k_i * g_diff_exp_i[:, :, :, None]
        state_update = jnp.einsum("bhik,bhiv->bhkv", k_scaled, v_new, precision=_MATMUL_PRECISION)
        new_state = jnp.nan_to_num(state_decayed + state_update, nan=0.0, posinf=0.0, neginf=0.0).astype(state.dtype)

        return new_state, core_out.astype(input_dtype)

    final_state, core_attn_out = lax.scan(chunk_step, initial_state, xs)

    core_attn_out = core_attn_out.transpose(1, 2, 0, 3, 4)
    core_attn_out = core_attn_out.reshape(B, H, -1, V_dim)
    core_attn_out = core_attn_out[:, :, :L, :]

    if not save_residual:
        return core_attn_out, final_state, None

    residual = (
        query,
        key,
        value,
        beta,
        attn,
        decay_mask.astype(input_dtype),
        g_cumsum_exp,
        g_end_exp,
        g_diff_state_exp,
        initial_state,
        q_inv_norm,
        k_inv_norm,
        L,
        pad_size,
        decay_was_none,
        initial_state_was_none,
    )
    return core_attn_out, final_state, residual


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
    """Chunked forward pass for gated delta rule with custom backward.

    Forward runs in ``input_dtype`` (bf16) for memory efficiency.
    Backward uses a hand-derived analytical reverse pass in float32,
    avoiding mixed-type issues inside ``shard_map`` backward.

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        value: Value tensor [batch, num_heads, seq_len, d_state].
        beta: Gating tensor [batch, num_heads, seq_len].
        decay: Per-token decay [batch, num_heads, seq_len].
        chunk_size: Size of chunks for parallel processing (non-diff).
        initial_state: Optional initial recurrent state.
        use_qk_l2norm: Whether to apply L2 normalization (non-diff).

    Returns:
        Tuple of (outputs, final_state).
    """
    return _chunk_gdr_fwd_impl(
        query,
        key,
        value,
        beta,
        decay,
        chunk_size,
        initial_state,
        use_qk_l2norm,
    )


def _chunk_gdr_fwd_rule(query, key, value, beta, decay, chunk_size, initial_state, use_qk_l2norm):
    """Forward pass for custom_vjp -- saves minimal backward context."""
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
    """Backward rule -- delegates to _xla_impl_bwd."""
    from ._xla_impl_bwd import _chunk_gdr_bwd

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
    """Exact multi-token chunked GDR path.

    The previous Neumann/custom-VJP implementation diverges catastrophically on
    real padded SFT batches even though isolated tensor probes looked small.
    Use the exact triangular-solve chunked formulation for the production
    multi-token path and rely on standard autodiff through it.
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
    """Single-step GDR update optimized for autoregressive inference.

    When seq_len=1 and we have an existing state, this function provides
    an optimized path that avoids the overhead of scan/chunk machinery.

    Args:
        query: Single query token [batch, num_heads, 1, head_dim].
        key: Single key token [batch, num_heads, 1, head_dim].
        value: Single value token [batch, num_heads, 1, d_state].
        beta: Gating for this token [batch, num_heads, 1].
        decay: Decay for this token [batch, num_heads, 1] or None.
        recurrent_state: Current memory state [batch, num_heads, head_dim, d_state].
        use_qk_l2norm: Whether to L2-normalize query and key.

    Returns:
        Tuple of (output, new_state).
    """
    if use_qk_l2norm:
        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    query = query.squeeze(2)
    key = key.squeeze(2)
    value = value.squeeze(2)
    beta = beta.squeeze(2)

    head_dim = query.shape[-1]
    scale = 1.0 / (head_dim**0.5)
    query = query * scale

    if decay is not None:
        decay = decay.squeeze(2)
        g_exp = jnp.exp(decay.astype(jnp.float32)).astype(recurrent_state.dtype)
        recurrent_state = recurrent_state * g_exp[:, :, None, None]

    kv_mem = jnp.sum(recurrent_state * key[:, :, :, None], axis=-2)

    beta_scaled = beta[:, :, None]
    delta = (value - kv_mem) * beta_scaled

    new_state = recurrent_state + key[:, :, :, None] * delta[:, :, None, :]

    output = jnp.sum(new_state * query[:, :, :, None], axis=-2)
    output = output[:, :, None, :]
    return output, new_state
