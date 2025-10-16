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


import math

import chex
import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int, PRNGKeyArray

from ejkernel.callib._ejit import ejit
from ejkernel.ops import BwdParams, FwdParams

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _flash_attention_bwd
from ._xla_impl_fwd import _flash_attention_fwd

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


def _make_core_func(
    precision_code_val: int,
    logits_dtype_code_val: int,
    chunk_size_q_val: int,
    chunk_size_k_val: int,
    normalize_output_val: bool,
    causal_val: bool,
    dropout_prob_val: float,
):
    """Create a specialized core function for given static parameters."""
    precision = _CODE_TO_PREC[precision_code_val]
    logits_dtype = _CODE_TO_DTYPE[logits_dtype_code_val]

    @jax.custom_vjp
    def _flash_attention_core_specialized(
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        bias: chex.Array | None,
        attention_mask: chex.Array | None,
        softmax_aux: chex.Array | None,
        sliding_window: tuple[int, int] | None,
        softmax_scale: float,
        logits_soft_cap: float | None,
        dropout_key: PRNGKeyArray | None,
    ) -> chex.Array:
        """Core flash attention with custom_vjp and attention sinks."""
        return _flash_attention_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=attention_mask,
            window=sliding_window,
            chunk_size_q=chunk_size_q_val,
            chunk_size_k=chunk_size_k_val,
            normalize_output=normalize_output_val,
            precision=precision,
            logits_dtype=logits_dtype,
            softmax_aux=softmax_aux,
            causal=causal_val,
            dropout_prob=dropout_prob_val,
            dropout_key=dropout_key,
        )

    def _fwd(
        query: chex.Array,
        key: chex.Array,
        value: chex.Array,
        bias: chex.Array | None,
        attention_mask: chex.Array | None,
        softmax_aux: chex.Array | None,
        sliding_window: tuple[int, int] | None,
        softmax_scale: float,
        logits_soft_cap: float | None,
        dropout_key: PRNGKeyArray | None,
    ):
        """Forward pass for custom_vjp: compute y and stash residuals."""
        y = _flash_attention_fwd(
            query,
            key,
            value,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=attention_mask,
            window=sliding_window,
            chunk_size_q=chunk_size_q_val,
            chunk_size_k=chunk_size_k_val,
            normalize_output=normalize_output_val,
            precision=precision,
            logits_dtype=logits_dtype,
            softmax_aux=softmax_aux,
            causal=causal_val,
            dropout_prob=dropout_prob_val,
            dropout_key=dropout_key,
        )

        ctx = (
            bias,
            attention_mask,
            softmax_aux,
            sliding_window,
            softmax_scale,
            logits_soft_cap,
            chunk_size_q_val,
            chunk_size_k_val,
            normalize_output_val,
            query,
            key,
            value,
            causal_val,
            dropout_prob_val,
            dropout_key,
        )
        return y, ctx

    def _bwd(ctx, g):
        """Backward pass wrapper for custom_vjp."""
        (
            bias,
            attention_mask,
            softmax_aux,
            sliding_window,
            softmax_scale,
            logits_soft_cap,
            chunk_size_q_val_,
            chunk_size_k_val_,
            normalize_output_val_,
            query,
            key,
            value,
            causal_val_,
            dropout_prob_val_,
            dropout_key,
        ) = ctx

        dq, dk, dv = _flash_attention_bwd(
            bias=bias,
            mask=attention_mask,
            softmax_aux=softmax_aux,
            window=sliding_window,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            chunk_size_q=chunk_size_q_val_,
            chunk_size_k=chunk_size_k_val_,
            normalize_output=normalize_output_val_,
            precision_code=precision_code_val,
            logits_dtype_code=logits_dtype_code_val,
            causal=causal_val_,
            dropout_prob=dropout_prob_val_,
            dropout_key=dropout_key,
            res=(query, key, value),
            g=g,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    _flash_attention_core_specialized.defvjp(_fwd, _bwd)
    return _flash_attention_core_specialized


_CORE_FUNC_CACHE = {}


def _flash_attention_core(
    query: chex.Array,
    key: chex.Array,
    value: chex.Array,
    bias: chex.Array | None,
    attention_mask: chex.Array | None,
    softmax_aux: chex.Array | None,
    sliding_window: tuple[int, int] | None,
    softmax_scale: float,
    logits_soft_cap: float | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: PRNGKeyArray | None,
) -> chex.Array:
    """Core flash attention dispatcher."""
    cache_key = (precision_code, logits_dtype_code, chunk_size_q, chunk_size_k, normalize_output, causal, dropout_prob)
    if cache_key not in _CORE_FUNC_CACHE:
        _CORE_FUNC_CACHE[cache_key] = _make_core_func(
            precision_code, logits_dtype_code, chunk_size_q, chunk_size_k, normalize_output, causal, dropout_prob
        )

    return _CORE_FUNC_CACHE[cache_key](
        query,
        key,
        value,
        bias,
        attention_mask,
        softmax_aux,
        sliding_window,
        softmax_scale,
        logits_soft_cap,
        dropout_key,
    )


@kernel_registry.register("flash_attention", Platform.XLA, Backend.ANY)
@ejit(
    static_argnames=[
        "softmax_scale",
        "dropout_prob",
        "causal",
        "dropout_seed",
        "sliding_window",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
        "normalize_output",
        "logits_dtype",
        "precision",
    ]
)
@jaxtyping.jaxtyped(typechecker=beartype)
def flash_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | Int[Array, "batch num_heads_or_1 seq_len_q seq_len_k"]
    | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    normalize_output: bool = True,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """
    Flash attention with memory-efficient chunked computation and attention sinks.

    This implementation uses online softmax to compute attention in chunks,
    reducing memory usage from O(N²) to O(N). Supports sliding window attention,
    logit soft capping, grouped query attention (GQA/MQA), and attention sinks.
    """
    if kv_segment_ids is not None and attention_mask is None:
        raise NotImplementedError("`kv_segment_ids` is not implemented in xla!")
    if q_segment_ids is not None and attention_mask is None:
        raise NotImplementedError("`q_segment_ids` is not implemented in xla!")
    if cum_seqlens_k is not None and attention_mask is None:
        raise NotImplementedError("`cum_seqlens_k` is not implemented in xla!")
    if cum_seqlens_q is not None and attention_mask is None:
        raise NotImplementedError("`cum_seqlens_q` is not implemented in xla!")

    dropout_key = None
    if dropout_prob > 0.0:
        if dropout_seed is None:
            dropout_seed = 0
        dropout_key = jax.random.PRNGKey(dropout_seed)

    if isinstance(sliding_window, int):
        window_tuple = (int(sliding_window), int(sliding_window))
    elif sliding_window is None:
        window_tuple = None
    else:
        window_left, window_right = sliding_window
        if window_left < 0 or window_right < 0:
            raise ValueError("Window bounds must be non-negative.")
        window_tuple = (int(window_left), int(window_right))

    if softmax_scale is None:
        D = query.shape[-1]
        scale_val = float(1.0 / math.sqrt(D))
    else:
        scale_val = float(softmax_scale)

    precision_code = _precision_to_code(precision)
    logits_dtype_code = _dtype_to_code(logits_dtype)

    return _flash_attention_core(
        query,
        key,
        value,
        bias,
        attention_mask,
        softmax_aux,
        window_tuple,
        scale_val,
        logits_soft_cap,
        int(fwd_params.q_blocksize),
        int(fwd_params.kv_blocksize),
        bool(normalize_output),
        precision_code,
        logits_dtype_code,
        bool(causal),
        float(dropout_prob),
        dropout_key,
    )
