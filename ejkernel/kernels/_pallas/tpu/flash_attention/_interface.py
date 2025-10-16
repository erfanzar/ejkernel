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


import functools

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib import ejit
from ejkernel.ops import BwdParams, FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import _flash_attention_bwd
from ._pallas_impl_fwd import _flash_attention_fwd, _flash_attention_impl
from ._utils import BlockSizes, SegmentIds


@kernel_registry.register("flash_attention", Platform.PALLAS, Backend.TPU)
@ejit(
    static_argnames=[
        "causal",
        "softmax_scale",
        "dropout_prob",
        "sliding_window",
        "logits_soft_cap",
        "logits_dtype",
        "precision",
        "normalize_output",
        "fwd_params",
        "bwd_params",
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
):
    del normalize_output, precision, logits_dtype, dropout_prob, dropout_seed

    if cum_seqlens_q is not None:
        raise NotImplementedError("Variable-length sequences (cum_seqlens_q) are not supported on TPU")
    if cum_seqlens_k is not None:
        raise NotImplementedError("Variable-length sequences (cum_seqlens_k) are not supported on TPU")
    if sliding_window is not None:
        raise NotImplementedError("Sliding window attention is not supported on TPU")
    if logits_soft_cap is not None:
        raise NotImplementedError("Logits soft cap is not supported on TPU")
    if softmax_aux is not None:
        raise NotImplementedError("Attention sinks (softmax_aux) are not supported on TPU")

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    batch_size, q_seq_len, num_heads, d_model = query.shape
    batch_size_k, kv_seq_len, num_heads_k, d_model_k = key.shape
    batch_size_v, kv_seq_len_v, num_heads_v, d_model_v = value.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and {batch_size_v} (for query, key, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        key = jnp.repeat(key, num_heads // num_heads_k, 2)
        value = jnp.repeat(value, num_heads // num_heads_v, 2)
    if d_model != d_model_k:
        raise ValueError(f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k respectively)")
    if d_model != d_model_v:
        raise NotImplementedError("V model dimension unequal to KV model dimension unsupported")
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}")
    if bias is not None:
        if bias.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
            raise ValueError(
                f"Attention bias shape mismatch: expected ({batch_size=},"
                f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {bias.shape}"
            )
    segment_ids = None
    if q_segment_ids is not None and kv_segment_ids is not None:
        if q_segment_ids.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=}, {q_seq_len=},), got {q_segment_ids.shape}"
            )

        if kv_segment_ids.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=}, {kv_seq_len=},), got {kv_segment_ids.shape}"
            )
        segment_ids = SegmentIds(q=q_segment_ids, kv=kv_segment_ids)
    block_sizes = BlockSizes(
        block_q=fwd_params.q_blocksize,
        block_k_major=fwd_params.kv_blocksize,
        block_k=fwd_params.kv_blocksize,
        block_b=1,
        block_q_major_dkv=bwd_params.q_blocksize,
        block_k_major_dkv=bwd_params.kv_blocksize,
        block_k_dkv=bwd_params.kv_blocksize,
        block_q_dkv=bwd_params.q_blocksize,
        block_k_major_dq=bwd_params.kv_blocksize,
        block_k_dq=bwd_params.kv_blocksize,
        block_q_dq=bwd_params.q_blocksize,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return _flash_attention(
        query.transpose(0, 2, 1, 3),
        key.transpose(0, 2, 1, 3),
        value.transpose(0, 2, 1, 3),
        bias,
        segment_ids,
        False,
        causal,
        softmax_scale,
        block_sizes,
    ).transpose(0, 2, 1, 3)


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 9))
def _flash_attention(
    query,
    key,
    value,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_sizes,
):
    return _flash_attention_impl(
        q=query,
        k=key,
        v=value,
        ab=ab,
        segment_ids=segment_ids,
        save_residuals=save_residuals,
        causal=causal,
        softmax_scale=softmax_scale,
        block_b=block_sizes.block_b,
        block_q=block_sizes.block_q,
        block_k_major=block_sizes.block_k_major,
        block_k=block_sizes.block_k,
    )


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)
