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

import chex
import jax


def _flash_attention_bwd(
    bias: chex.Array | None,
    mask: chex.Array | None,
    softmax_aux: chex.Array | None,
    window: tuple[int, int] | None,
    scale: float,
    logits_soft_cap: float | None,
    chunk_size_q: int,
    chunk_size_k: int,
    normalize_output: bool,
    precision_code: int,
    logits_dtype_code: int,
    causal: bool,
    dropout_prob: float,
    dropout_key: chex.Array | None,
    res: tuple,
    g: chex.Array,
) -> tuple:
    """
    Backward pass for flash attention using JAX autodiff.

    Args:
            bias: Optional bias array
            mask: Optional mask array
            softmax_aux: Optional attention sink logits
            window: Optional sliding window
            scale: Softmax scale factor
            logits_soft_cap: Optional soft cap value
            chunk_size_q: Query chunk size
            chunk_size_k: Key chunk size
            normalize_output: Whether to normalize output
            precision_code: Precision code
            logits_dtype_code: Logits dtype code
            causal: Whether to apply causal masking
            dropout_prob: Dropout probability
            dropout_key: PRNG key for dropout
            res: Residuals from forward pass (q, k, v)
            g: Gradient of output

    Returns:
            Tuple of gradients (dq, dk, dv, None, None, ...)
    """
    from ._xla_impl_fwd import _flash_attention_fwd

    # Precision/dtype mappers
    _CODE_TO_PREC = {
        0: jax.lax.Precision.DEFAULT,
        1: jax.lax.Precision.HIGHEST,
        2: jax.lax.Precision.HIGH,
    }
    _CODE_TO_DTYPE = {
        0: jax.numpy.float16,
        1: jax.numpy.bfloat16,
        2: jax.numpy.float32,
        3: jax.numpy.float64,
    }

    q, k, v = res
    precision = _CODE_TO_PREC[precision_code]
    logits_dtype = _CODE_TO_DTYPE[logits_dtype_code]

    def f(q_, k_, v_):
        return _flash_attention_fwd(
            q_,
            k_,
            v_,
            scale=scale,
            logits_soft_cap=logits_soft_cap,
            bias=bias,
            mask=mask,
            window=window,
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

    _, pullback = jax.vjp(f, q, k, v)
    dq, dk, dv = pullback(g)

    return (
        dq,
        dk,
        dv,
        None,  # bias
        None,  # mask
        None,  # softmax_aux
        None,  # window_left
        None,  # window_right
        None,  # scale
        None,  # softcap
        None,  # chunk_size_q
        None,  # chunk_size_k
        None,  # normalize_output
        None,  # precision_code
        None,  # logits_dtype_code
        None,  # causal
        None,  # dropout_prob
        None,  # dropout_key
    )
