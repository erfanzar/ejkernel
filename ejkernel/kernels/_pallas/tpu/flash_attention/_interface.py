import functools

import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ejkernel.callib import ejit

from ...._registry import Backend, Platform, kernel_registry
from ._pallas_impl_bwd import _flash_attention_bwd
from ._pallas_impl_fwd import _flash_attention_fwd, _flash_attention_impl
from ._utils import BlockSizes, SegmentIds


@kernel_registry.register("flash_attention", Platform.PALLAS, Backend.TPU)
@ejit(static_argnames=["causal", "softmax_scale", "debug", "dropout_prob", "sliding_window", "logits_soft_cap"])
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
):
    del normalize_output, precision, logits_dtype
    # Validate unsupported parameters for TPU
    if attention_mask is not None:
        raise NotImplementedError("attention_mask parameter is not supported on TPU. Use bias instead.")
    if dropout_prob != 0.0:
        raise NotImplementedError("Dropout is not supported on TPU flash attention")
    if dropout_seed is not None:
        raise NotImplementedError("dropout_seed parameter is not supported on TPU")
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

    batch_size, num_heads, q_seq_len, d_model = query.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = key.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = value.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and {batch_size_v} (for query, key, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k}, {num_heads_v} (for query, key, v respectively)"
        )
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
        block_q=chunk_size_q,
        block_k_major=chunk_size_k,
        block_k=chunk_size_k,
        block_b=1,
        block_q_major_dkv=chunk_size_q,
        block_k_major_dkv=chunk_size_k,
        block_k_dkv=chunk_size_k,
        block_q_dkv=chunk_size_q,
        block_k_major_dq=chunk_size_k,
        block_k_dq=chunk_size_k,
        block_q_dq=chunk_size_q,
    )
    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    return _flash_attention(query, key, value, bias, segment_ids, False, causal, softmax_scale, block_sizes, debug)


@functools.partial(jax.custom_vjp, nondiff_argnums=range(5, 10))
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
    debug,
):
    return _flash_attention_impl(
        query,
        key,
        value,
        ab,
        segment_ids,
        save_residuals,
        causal,
        softmax_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        debug,
    )


_flash_attention.defvjp(fwd=_flash_attention_fwd, bwd=_flash_attention_bwd)
