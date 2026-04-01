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

"""Lloyd-Max optimal scalar quantizer for TurboQuant.

After a random rotation by a Haar-distributed orthogonal matrix, each
coordinate of a d-dimensional unit vector follows a distribution well
approximated by N(0, 1/d). The Lloyd-Max quantizer finds the optimal
scalar quantization centroids for this distribution by iteratively
solving the continuous 1-D k-means conditions.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma as gamma_fn


@dataclasses.dataclass(frozen=True)
class LloydMaxCodebook:
    """Precomputed Lloyd-Max codebook for a given (bits, dim) pair.

    Attributes:
        bits: Number of quantization bits.
        dim: Dimensionality of the original vectors (used for variance = 1/dim).
        n_levels: Number of quantization levels (2^bits).
        centroids: Array of centroid values, shape [n_levels].
        boundaries: Array of decision boundaries, shape [n_levels - 1].
    """

    bits: int
    dim: int
    n_levels: int
    centroids: np.ndarray
    boundaries: np.ndarray


def solve_lloyd_max(
    bits: int,
    dim: int = 128,
    max_iters: int = 200,
    tol: float = 1e-10,
    use_gaussian_approx: bool = True,
) -> LloydMaxCodebook:
    """Solve for optimal Lloyd-Max centroids for N(0, 1/dim) distribution.

    Uses iterative optimization: alternate between computing decision
    boundaries (midpoints between adjacent centroids) and updating
    centroids as conditional expectations E[X | X in partition_i].

    Args:
        bits: Number of quantization bits (1-8).
        dim: Vector dimensionality (determines variance = 1/dim).
        max_iters: Maximum iterations for convergence.
        tol: Convergence tolerance on centroid change.
        use_gaussian_approx: If True, use Gaussian approximation (good for dim >= 64).
            Otherwise use exact Beta-type PDF.

    Returns:
        LloydMaxCodebook with precomputed centroids and boundaries.
    """
    from scipy import integrate
    from scipy.stats import norm

    n_levels = 1 << bits
    sigma = 1.0 / np.sqrt(dim)

    if use_gaussian_approx or dim >= 64:

        def pdf(x):
            return norm.pdf(x, loc=0, scale=sigma)

        def x_pdf(x):
            return x * norm.pdf(x, loc=0, scale=sigma)

    else:
        coeff = gamma_fn(dim / 2.0) / (np.sqrt(np.pi) * gamma_fn((dim - 1) / 2.0))

        def pdf(x):
            x2 = np.clip(x * x, 0, 1.0 - 1e-15)
            return coeff * (1.0 - x2) ** ((dim - 3) / 2.0)

        def x_pdf(x):
            return x * pdf(x)

    # Initialize centroids uniformly in [-3*sigma, 3*sigma]
    lo, hi = -3.0 * sigma, 3.0 * sigma
    centroids = np.linspace(lo, hi, n_levels)

    for _ in range(max_iters):
        # Decision boundaries: midpoints between adjacent centroids
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])

        # Compute new centroids as conditional expectations
        # E[X | b_{i-1} <= X < b_i] = integral(x*pdf(x), b_{i-1}, b_i) / integral(pdf(x), b_{i-1}, b_i)
        edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
        new_centroids = np.zeros(n_levels)

        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(x_pdf, a, b, limit=100)
            den, _ = integrate.quad(pdf, a, b, limit=100)
            if den > 1e-15:
                new_centroids[i] = num / den
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])

    return LloydMaxCodebook(
        bits=bits,
        dim=dim,
        n_levels=n_levels,
        centroids=centroids.astype(np.float32),
        boundaries=boundaries.astype(np.float32),
    )


@functools.lru_cache(maxsize=32)
def get_codebook(bits: int, dim: int = 128) -> LloydMaxCodebook:
    """Get or compute a cached Lloyd-Max codebook.

    Args:
        bits: Number of quantization bits.
        dim: Vector dimensionality.

    Returns:
        Cached LloydMaxCodebook instance.
    """
    return solve_lloyd_max(bits=bits, dim=dim)


def quantize_to_indices(
    x: jax.Array,
    centroids: jax.Array,
) -> jax.Array:
    """Quantize each element of x to the nearest centroid index.

    Args:
        x: Input array of arbitrary shape.
        centroids: 1D array of centroid values [n_levels].

    Returns:
        Integer array of same shape as x, with values in [0, n_levels).
    """
    diffs = jnp.abs(x[..., None] - centroids)
    return jnp.argmin(diffs, axis=-1).astype(jnp.uint8)


def dequantize_from_indices(
    indices: jax.Array,
    centroids: jax.Array,
) -> jax.Array:
    """Map indices back to centroid float values.

    Args:
        indices: Integer array of centroid indices.
        centroids: 1D array of centroid values [n_levels].

    Returns:
        Float array of same shape as indices.
    """
    # One-hot matmul avoids Mosaic's 2D-only gather on TPU Pallas.
    # Safe for small codebooks (8-16 entries at 3-4 bits).
    return jnp.sum(
        jax.nn.one_hot(indices, centroids.shape[0], dtype=jnp.float32) * centroids,
        axis=-1,
    )
