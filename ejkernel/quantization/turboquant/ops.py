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

"""TurboQuant compression, decompression, and asymmetric attention operations.

Core operations for the TurboQuant algorithm:
- Key compression: rotation + Lloyd-Max quantization + QJL sign encoding
- Value compression: rotation + Lloyd-Max quantization (MSE-only)
- Asymmetric attention scores: unbiased inner product estimation via QJL
- Value decompression: centroid lookup + inverse rotation
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .codebook import dequantize_from_indices, quantize_to_indices
from .packing import pack_4bit, pack_signs, unpack_4bit, unpack_signs


def turboquant_compress_keys(
    keys: jax.Array,
    rotation_matrix: jax.Array,
    key_codebook: jax.Array,
    projection_matrix: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compress key vectors using TurboQuant Stage 1 + Stage 2 (QJL).

    Pipeline:
        1. Normalize keys, store original norms
        2. Rotate by Pi: y = keys_norm @ Pi^T
        3. Lloyd-Max quantize each coordinate
        4. Compute MSE reconstruction residual
        5. Project residual through S, store sign bits and residual norms
        6. Pack indices (4-bit) and signs (1-bit)

    Args:
        keys: Key vectors, shape [..., head_dim].
        rotation_matrix: Orthogonal matrix Pi, shape [head_dim, head_dim].
        key_codebook: Lloyd-Max centroids for keys, shape [2^(bits-1)].
        projection_matrix: QJL matrix S, shape [qjl_dim, head_dim].

    Returns:
        Tuple of:
            - packed_indices: uint8, shape [..., head_dim // 2] (4-bit packed)
            - packed_signs: uint8, shape [..., qjl_dim // 8] (bit-packed)
            - norms: bf16, shape [..., 2] ([:, 0] = original norm, [:, 1] = residual norm)
    """
    keys.shape[:-1]
    keys.shape[-1]

    keys_f32 = keys.astype(jnp.float32)

    # 1. Normalize
    original_norms = jnp.linalg.norm(keys_f32, axis=-1, keepdims=True)
    safe_norms = jnp.maximum(original_norms, 1e-8)
    keys_normalized = keys_f32 / safe_norms

    # 2. Rotate: y = keys_norm @ Pi^T (each row is Pi^T @ key)
    rotated = keys_normalized @ rotation_matrix.T

    # 3. Lloyd-Max quantize
    indices = quantize_to_indices(rotated, key_codebook)

    # 4. Reconstruct and compute residual (in rotated space)
    reconstructed_rotated = dequantize_from_indices(indices, key_codebook)
    residual_rotated = rotated - reconstructed_rotated

    # 5. Project residual: projected = residual @ S^T, then take sign
    projected = residual_rotated @ projection_matrix.T
    signs_bool = projected >= 0

    # Compute residual norm (of the rotated-space residual, which equals
    # the norm of the original-space residual since Pi is orthogonal)
    residual_norms = jnp.linalg.norm(residual_rotated, axis=-1, keepdims=True)

    # 6. Pack
    packed_indices = pack_4bit(indices)
    packed_signs = pack_signs(signs_bool.astype(jnp.uint8))

    # Combine norms: [original_norm, residual_norm]
    norms = jnp.concatenate(
        [original_norms, residual_norms],
        axis=-1,
    ).astype(jnp.bfloat16)

    return packed_indices, packed_signs, norms


def turboquant_compress_values(
    values: jax.Array,
    rotation_matrix: jax.Array,
    value_codebook: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compress value vectors using TurboQuant Stage 1 only (MSE).

    Pipeline:
        1. Normalize values, store original norms
        2. Rotate by Pi: y = values_norm @ Pi^T
        3. Lloyd-Max quantize each coordinate
        4. Pack indices (4-bit)

    Args:
        values: Value vectors, shape [..., head_dim].
        rotation_matrix: Orthogonal matrix Pi, shape [head_dim, head_dim].
        value_codebook: Lloyd-Max centroids for values, shape [2^bits].

    Returns:
        Tuple of:
            - packed_indices: uint8, shape [..., head_dim // 2] (4-bit packed)
            - norms: bf16, shape [...] (original norms)
    """
    values_f32 = values.astype(jnp.float32)

    # 1. Normalize
    original_norms = jnp.linalg.norm(values_f32, axis=-1)
    safe_norms = jnp.maximum(original_norms, 1e-8)
    values_normalized = values_f32 / safe_norms[..., None]

    # 2. Rotate
    rotated = values_normalized @ rotation_matrix.T

    # 3. Quantize
    indices = quantize_to_indices(rotated, value_codebook)

    # 4. Pack
    packed_indices = pack_4bit(indices)

    return packed_indices, original_norms.astype(jnp.bfloat16)


def turboquant_dequantize_values(
    packed_indices: jax.Array,
    norms: jax.Array,
    value_codebook: jax.Array,
    rotation_matrix: jax.Array,
) -> jax.Array:
    """Full MSE decompression for value vectors.

    Reverses turboquant_compress_values:
        1. Unpack indices
        2. Look up centroids
        3. Apply inverse rotation: x_mse = centroids @ Pi
        4. Rescale by original norms

    Args:
        packed_indices: uint8, shape [..., head_dim // 2].
        norms: bf16, shape [...] (original vector norms).
        value_codebook: Lloyd-Max centroids, shape [2^bits].
        rotation_matrix: Orthogonal matrix Pi, shape [head_dim, head_dim].

    Returns:
        Dequantized values, shape [..., head_dim], same dtype as norms promoted to float32.
    """
    # 1. Unpack
    indices = unpack_4bit(packed_indices)

    # 2. Centroid lookup
    centroids = dequantize_from_indices(indices, value_codebook)

    # 3. Inverse rotation: x_approx = centroids @ Pi (since Pi is orthogonal, Pi^{-1} = Pi^T,
    #    and for row vectors: (Pi^T @ y)^T = y^T @ Pi, so row_vec @ Pi = inverse_rotate)
    derotated = centroids @ rotation_matrix

    # 4. Rescale
    return derotated * norms[..., None].astype(jnp.float32)


def turboquant_asymmetric_scores(
    q_rotated: jax.Array,
    q_projected: jax.Array,
    k_packed_indices: jax.Array,
    k_norms: jax.Array,
    k_packed_signs: jax.Array,
    key_codebook: jax.Array,
    qjl_dim: int,
) -> jax.Array:
    """Compute asymmetric attention scores from TurboQuant-compressed keys.

    Implements the unbiased inner product estimator:
        <q, k> ~ <q_rot, codebook[k_idx]> * k_orig_norm
                + sqrt(pi/2) / qjl_dim * <q_proj, signs> * k_residual_norm

    Args:
        q_rotated: Pre-rotated queries (Q @ Pi^T), shape [qblocks, kv_heads, q_per_kv, head_dim].
        q_projected: Pre-projected queries (Q @ S^T), shape [qblocks, kv_heads, q_per_kv, qjl_dim].
        k_packed_indices: Packed key codebook indices, shape [kv_tok, kv_heads, head_dim // 2].
        k_norms: Key norms, shape [kv_tok, kv_heads, 2] ([:,0]=original, [:,1]=residual).
        k_packed_signs: Packed QJL signs, shape [kv_tok, kv_heads, qjl_dim // 8].
        key_codebook: Lloyd-Max centroids, shape [2^(bits-1)].
        qjl_dim: QJL projection dimension.

    Returns:
        Attention logits, shape [qblocks, kv_heads, q_per_kv, kv_tok].
    """
    # Unpack key data
    k_indices = unpack_4bit(k_packed_indices)  # [kv_tok, kv_heads, head_dim]
    k_signs = unpack_signs(k_packed_signs)  # [kv_tok, kv_heads, qjl_dim]

    k_orig_norms = k_norms[..., 0].astype(jnp.float32)  # [kv_tok, kv_heads]
    k_res_norms = k_norms[..., 1].astype(jnp.float32)  # [kv_tok, kv_heads]

    # MSE term: q_rotated @ codebook[k_indices]^T * k_original_norms
    k_centroids = dequantize_from_indices(k_indices, key_codebook)  # [kv_tok, kv_heads, head_dim]

    # einsum output: [b(qblocks), i(kv_heads), h(q_per_kv), k(kv_tok)]
    logits_mse = jnp.einsum("bihd,kid->bihk", q_rotated, k_centroids)
    # k_orig_norms [kv_tok, kv_heads] -> [kv_heads, kv_tok] -> [1, kv_heads, 1, kv_tok]
    logits_mse = logits_mse * k_orig_norms.T[None, :, None, :]

    # QJL correction: sqrt(pi/2) / qjl_dim * (q_projected @ k_signs^T) * k_residual_norms
    # einsum: [b,i,h,m] x [k,i,m] -> [b,i,h,k]
    correction = jnp.einsum("bihm,kim->bihk", q_projected, k_signs)
    factor = jnp.sqrt(jnp.float32(jnp.pi / 2.0)) / jnp.float32(qjl_dim)
    correction = correction * k_res_norms.T[None, :, None, :] * factor

    return logits_mse + correction
