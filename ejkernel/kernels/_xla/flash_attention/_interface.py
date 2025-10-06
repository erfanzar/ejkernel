# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

import math

import chex
import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ejkernel.callib._ejit import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _flash_attention_bwd
from ._xla_impl_fwd import _flash_attention_fwd

# Precision/dtype code mappers (to avoid passing enums/dtypes through jit/custom_vjp)
_PREC_TO_CODE = {
    jax.lax.Precision.DEFAULT: 0,
    jax.lax.Precision.HIGHEST: 1,
    jax.lax.Precision.HIGH: 2,
}
_CODE_TO_PREC = {
    0: jax.lax.Precision.DEFAULT,
    1: jax.lax.Precision.HIGHEST,
    2: jax.lax.Precision.HIGH,
}
_DTYPE_TO_CODE = {
    jnp.dtype("float16"): 0,
    jnp.dtype("bfloat16"): 1,
    jnp.dtype("float32"): 2,
    jnp.dtype("float64"): 3,
}
_CODE_TO_DTYPE = {
    0: jnp.float16,
    1: jnp.bfloat16,
    2: jnp.float32,
    3: jnp.float64,
}


def _precision_to_code(precision) -> int:
    """Convert precision to code."""
    if isinstance(precision, int):
        return int(precision)
    try:
        return _PREC_TO_CODE[precision]
    except KeyError as e:
        raise ValueError("precision must be jax.lax.Precision.{DEFAULT|HIGHEST|HIGH} or an int code {0,1,2}.") from e


def _dtype_to_code(dtype) -> int:
    """Convert dtype to code."""
    d = jnp.dtype(dtype)
    try:
        return _DTYPE_TO_CODE[d]
    except KeyError as e:
        raise ValueError("logits_dtype must be one of float16, bfloat16, float32, float64.") from e


@jax.custom_vjp
def _flash_attention_core(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    attention_mask: chex.Array | None,
    softmax_aux: chex.Array | None,
    window_left: int,
    window_right: int,
    scale: float,
    softcap: float,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: chex.PRNGKey | None,
) -> chex.Array:
    """Core flash attention with custom_vjp and attention sinks."""
    sliding_window = None if (window_left < 0 or window_right < 0) else (window_left, window_right)
    precision = _CODE_TO_PREC[precision_code]
    logits_dtype = _CODE_TO_DTYPE[logits_dtype_code]
    logits_soft_cap = None if (softcap < 0) else softcap

    # Handle scale: if -1.0, compute default
    if scale < 0:
        D = query.shape[-1]
        scale = 1.0 / jnp.sqrt(float(D))

    return _flash_attention_fwd(
        query,
        key,
        value,
        scale=scale,
        logits_soft_cap=logits_soft_cap,
        bias=bias,
        mask=attention_mask,
        window=sliding_window,
        chunk_size_q=chunk_size_q,
        chunk_size_k=chunk_size_k,
        normalize_output=normalize_output,
        precision=precision,
        logits_dtype=logits_dtype,
        softmax_aux=softmax_aux,
        causal=causal,
        dropout_prob=dropout_prob,
        dropout_key=dropout_key,
    )


def _flash_attention_core_fwd(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    attention_mask: chex.Array | None,
    softmax_aux: chex.Array | None,
    window_left: int,
    window_right: int,
    scale: float,
    softcap: float,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: chex.PRNGKey | None,
):
    """Forward pass for custom_vjp."""
    y = _flash_attention_core(
        query,
        key,
        value,
        bias,
        attention_mask,
        softmax_aux,
        window_left,
        window_right,
        scale,
        softcap,
        chunk_size_q,
        chunk_size_k,
        normalize_output,
        precision_code,
        logits_dtype_code,
        causal,
        dropout_prob,
        dropout_key,
    )

    sliding_window = None if (window_left < 0 or window_right < 0) else (window_left, window_right)
    logits_soft_cap = None if (softcap < 0) else softcap

    # Handle scale: if -1.0, compute default
    if scale < 0:
        D = query.shape[-1]
        scale = 1.0 / jnp.sqrt(float(D))

    ctx = (
        bias,
        attention_mask,
        softmax_aux,
        sliding_window,
        scale,
        logits_soft_cap,
        chunk_size_q,
        chunk_size_k,
        normalize_output,
        precision_code,
        logits_dtype_code,
        query,
        key,
        value,
        causal,
        dropout_prob,
        dropout_key,
    )
    return y, ctx


def _flash_attention_core_bwd_wrapper(ctx, g):
    """Backward pass wrapper for custom_vjp."""
    (
        bias,
        attention_mask,
        softmax_aux,
        sliding_window,
        scale,
        logits_soft_cap,
        chunk_size_q,
        chunk_size_k,
        normalize_output,
        precision_code,
        logits_dtype_code,
        query,
        key,
        value,
        causal,
        dropout_prob,
        dropout_key,
    ) = ctx

    return _flash_attention_bwd(
        bias,
        attention_mask,
        softmax_aux,
        sliding_window,
        scale,
        logits_soft_cap,
        chunk_size_q,
        chunk_size_k,
        normalize_output,
        precision_code,
        logits_dtype_code,
        causal,
        dropout_prob,
        dropout_key,
        (query, key, value),
        g,
    )


_flash_attention_core.defvjp(_flash_attention_core_fwd, _flash_attention_core_bwd_wrapper)


@kernel_registry.register("flash_attention", Platform.XLA, Backend.ANY)
@ejit(
    static_argnames=[
        "softmax_scale",
        "dropout_prob",
        "causal",
        "dropout_seed",
        "sliding_window",
        "chunk_size_q",
        "chunk_size_k",
        "logits_soft_cap",
        "normalize_output",
        "debug",
        "logits_dtype",
        "precision",
    ]
)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size_q: int = 128,
    chunk_size_k: int = 128,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: jnp.dtype = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    debug: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """
    Flash attention with memory-efficient chunked computation and attention sinks.

    This implementation uses online softmax to compute attention in chunks,
    reducing memory usage from O(N²) to O(N). Supports sliding window attention,
    logit soft capping, grouped query attention (GQA/MQA), and attention sinks.

    Args:
        query: Query array of shape [B, Tq, H, D]
        key: Key array of shape [B, Tk, Hk, D] where Hk can be 1 (MQA) or H (MHA)
        value: Value array of shape [B, Tk, Hk, d]
        bias: Optional bias array broadcastable to [B, H, Tq, Tk]
        attention_mask: Optional boolean mask array broadcastable to [B, H, Tq, Tk]
        sliding_window: Optional sliding window size. Can be:
            - int: symmetric window (n tokens on each side)
            - tuple (left, right): asymmetric window
            - None: no window (full attention)
        scale: Optional softmax scale. If None, uses 1/sqrt(D)
        logits_soft_cap: Optional soft cap value for logits to prevent overflow
        chunk_size_q: Query chunk size for memory efficiency
        chunk_size_k: Key chunk size for memory efficiency
        normalize_output: Whether to normalize output by softmax denominator
        precision: Matrix multiplication precision
        logits_dtype: Dtype for logits computation (promotes to at least float32)
        softmax_aux: Optional attention sink logits of shape [H, num_sinks] or [num_sinks].
            These are auxiliary logits that participate in softmax normalization but
            don't contribute to output, allowing the model to absorb probability mass.

    Returns:
        Output array of shape [B, Tq, H, d]

    Examples:
        >>> # Standard attention
        >>> y = flash_attention(q, key, v)

        >>> # Causal attention with sliding window
        >>> y = flash_attention(q, key, value, sliding_window=(T-1, 0))

        >>> # MQA with soft cap
        >>> y = flash_attention(q, k[:, :, :1], v[:, :, :1], logits_soft_cap=20.0)

        >>> # Attention with sinks (4 learnable sink logits per head)
        >>> sinks = jax.random.normal(key, (H, 4))
        >>> y = flash_attention(q, key, value, softmax_aux=sinks)

    Note:
        Attention sinks are learnable parameters that participate in the softmax
        normalization but don't produce output. They allow the model to "dump"
        attention probability mass, which can improve numerical stability and help
        the model learn better attention distributions.
    """
    if kv_segment_ids is not None:
        raise NotImplementedError("`kv_segment_ids` is not implemented in xla!")
    if q_segment_ids is not None:
        raise NotImplementedError("`q_segment_ids` is not implemented in xla!")
    if cum_seqlens_k is not None:
        raise NotImplementedError("`cum_seqlens_k` is not implemented in xla!")
    if cum_seqlens_q is not None:
        raise NotImplementedError("`cum_seqlens_q` is not implemented in xla!")

    # Handle dropout seed and create RNG key
    dropout_key = None
    if dropout_prob > 0.0:
        if dropout_seed is None:
            dropout_seed = 0
        dropout_key = jax.random.PRNGKey(dropout_seed)

    # Convert window to tuple or encode as negative values
    if isinstance(sliding_window, int):
        window_left = window_right = int(sliding_window)
    elif sliding_window is None:
        window_left = window_right = -1
    else:
        window_left, window_right = sliding_window
        if window_left < 0 or window_right < 0:
            raise ValueError("Window bounds must be non-negative.")
        window_left = int(window_left)
        window_right = int(window_right)

    if softmax_scale is None:
        D = query.shape[-1]
        scale_val = float(1.0 / math.sqrt(D))
    else:
        scale_val = float(softmax_scale)

    # Convert soft cap - encode None as negative
    softcap = float(logits_soft_cap) if logits_soft_cap is not None else -1.0

    # Convert precision and dtype to codes
    precision_code = _precision_to_code(precision)
    logits_dtype_code = _dtype_to_code(logits_dtype)

    return _flash_attention_core(
        query,
        key,
        value,
        bias,
        attention_mask,
        softmax_aux,
        window_left,
        window_right,
        scale_val,
        softcap,
        int(chunk_size_q),
        int(chunk_size_k),
        bool(normalize_output),
        precision_code,
        logits_dtype_code,
        bool(causal),
        float(dropout_prob),
        dropout_key,
    )
