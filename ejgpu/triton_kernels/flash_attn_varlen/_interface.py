# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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

from functools import partial

import jax
from eformer.callib import cdiv, next_power_of_2, triton_call
from jax import numpy as jnp

from ...utils import (
    Layouts,
    get_padded_headsize,
    get_shape_from_layout,
    get_strides,
    get_strides_from_layout,
    narrow,
)
from ._triton_impl_fwd_decode import _fwd_kernel_stage1, _fwd_kernel_stage2, get_split_k
from ._triton_impl_fwd_prefill import attn_fwd


@partial(jax.jit, static_argnums=(3, 4, 5, 7, 11, 12))
def _flash_attn_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sm_scale: float | None = None,
    causal: bool = True,
    layout: Layouts = "bshd",
    b: jax.Array | None = None,
    dropout_p: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    cu_seqlens_q: jax.Array | None = None,
    cu_seqlens_k: jax.Array | None = None,
    max_seqlens_q: int | None = None,
    max_seqlens_k: int | None = None,
):
    is_varlen = layout == "thd"

    if b is not None:
        assert b.size < 2**31
    batch, nheads_q, nheads_k, head_size, seqlen_q, seqlen_k = get_shape_from_layout(
        q,
        k,
        layout,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
    )
    if sm_scale is None:
        sm_scale = head_size**-0.5
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, q, layout)

    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vk, stride_vn = v_strides
    stride_oz, stride_oh, stride_om, stride_on = o_strides

    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)

    if is_varlen:
        softmax_lse_shape = (q.shape[0], nheads_q)
        stride_lm, stride_lh = get_strides(softmax_lse_shape)
        stride_lz = 0
    else:
        softmax_lse_shape = (batch, nheads_q, max_seqlens_q)
        stride_lz, stride_lh, stride_lm = get_strides(softmax_lse_shape)

    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    alibi_strides = (0, 0)
    bias_strides = (0, 0, 0, 0)

    if b is not None:
        bias_strides = get_strides(b)

    if alibi_slopes is not None:
        alibi_strides = get_strides(alibi_slopes)

    _, stride_bh, stride_bm, stride_bn = bias_strides

    stride_az, stride_ah = alibi_strides

    out_shape = []
    out_shape.append(jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype))
    out_shape.append(jax.ShapeDtypeStruct(shape=softmax_lse_shape, dtype=jnp.float32))

    metaparams = dict(
        ACTUAL_BLOCK_DIM=head_size,
        ENABLE_DROPOUT=dropout_p > 0.0,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        BLOCK_DIM=padded_d_model,
        USE_ALIBI=False if alibi_slopes is None else True,
        USE_BIAS=False if b is None else True,
        kv_heads=nheads_k,
        VARLEN=is_varlen,
        q_heads=nheads_q,
    )

    o, softmax_lse = triton_call(
        q,
        k,
        v,
        b if b is not None else 0,
        sm_scale,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vk,
        stride_vn,
        stride_oz,
        stride_oh,
        stride_om,
        stride_on,
        stride_bh,
        stride_bm,
        stride_bn,
        stride_az,
        stride_ah,
        stride_lz,
        stride_lh,
        stride_lm,
        cu_seqlens_q if cu_seqlens_q is not None else 0,
        cu_seqlens_k if cu_seqlens_k is not None else 0,
        dropout_p,
        philox_seed,
        philox_offset,
        alibi_slopes if alibi_slopes is not None else 0,
        grid=lambda META: (cdiv(max_seqlens_q, META["BLOCK_M"]), nheads_q, batch),
        out_shape=out_shape,
        kernel=attn_fwd,
        name="ejgpu:attn_forward:prefill",
        **metaparams,
    )

    return o, softmax_lse


@partial(jax.jit, static_argnums=(3, 4, 6))
def _flash_attn_varlen_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sm_scale: float | None,
    causal: bool,
    alibi_slopes: jax.Array | None,
    layout: Layouts | None,
    cache_seqlens: jax.Array | int | list[int],
    cache_batch_idx: int | list[int],
    new_kv: jax.Array | None,
    k_new: jax.Array | None,
    v_new: jax.Array | None,
):
    original_layout = layout
    if layout == "bshd":
        q = jnp.expand_dims(q, 2)
        k = jnp.expand_dims(k, 2)
        v = jnp.expand_dims(v, 2)
        if new_kv:
            k_new = jnp.expand_dims(k_new, 2)
            v_new = jnp.expand_dims(v_new, 2)
        layout = "bsghd"
    elif layout == "bhsd":
        q = jnp.expand_dims(q.transpose(0, 2, 1, 3), 2)
        k = jnp.expand_dims(k.transpose(0, 2, 1, 3), 2)
        v = jnp.expand_dims(v.transpose(0, 2, 1, 3), 2)
        if new_kv:
            k_new = jnp.expand_dims(k_new.transpose(0, 2, 1, 3), 2)
            v_new = jnp.expand_dims(v_new.transpose(0, 2, 1, 3), 2)
        layout = "bsghd"
    elif layout == "bsghd":
        pass
    elif layout is None:
        raise ValueError("Layout not given")
    assert layout == "bsghd"

    batch_size, seqlen_q, n_group_q, heads_per_group_q, dim_q = q.shape
    z, seqlen_k, n_group_k, heads_per_group_k, dim_k = k.shape
    z, seqlen_v, n_group_v, heads_per_group_v, dim_v = v.shape

    assert dim_q == dim_k == dim_v, f"Dimensions must match: {dim_q}, {dim_k}, {dim_v}"

    dim_padded = get_padded_headsize(dim_k)
    if heads_per_group_q > heads_per_group_k:
        is_gqa = True
    elif heads_per_group_q < heads_per_group_k:
        raise ValueError("heads_per_group_q < heads_per_group_k")
    else:
        is_gqa = False

    assert dim_k == dim_q, f"Keys have head dim {dim_k} but queries have head dim {dim_q}"

    BLOCK_M = 16
    BLOCK_N = 64

    split_k = get_split_k(batch_size, n_group_q, heads_per_group_q, seqlen_k)
    seqlen_q_ceil = (seqlen_q + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    out_splitk_shape = (batch_size * n_group_q * heads_per_group_q, split_k, seqlen_q_ceil, dim_padded)
    metadata_shape = (batch_size * n_group_q * heads_per_group_q, 2, split_k, seqlen_q_ceil)

    lse_shape = (batch_size * n_group_q * heads_per_group_q, seqlen_q)
    out_shape = (batch_size, seqlen_q, n_group_q, heads_per_group_q, dim_padded)
    grid = (cdiv(seqlen_q, BLOCK_M), batch_size * n_group_q * heads_per_group_q, split_k)

    num_warps = 1
    split_size = (seqlen_k + split_k - 1) // split_k
    use_cache_seqlens = cache_seqlens is not None

    stride_qz, stride_qm, stride_qg, stride_qh, stride_qd = get_strides(q)
    stride_kz, stride_kn, stride_kg, stride_kh, stride_kd = get_strides(k)
    stride_vz, stride_vn, stride_vg, stride_vh, stride_vd = get_strides(v)
    stride_osk_zhg, stride_osk_s, stride_osk_m, stride_osk_k = get_strides(out_splitk_shape)
    stride_mzhg, stride_m2, stride_ms, stride_mm = get_strides(metadata_shape)
    stride_kn_z, stride_kn_n, stride_kn_g, stride_kn_h, stride_kn_d = get_strides(k_new)
    stride_vn_z, stride_vn_n, stride_vn_g, stride_vn_h, stride_vn_d = get_strides(v_new)
    stride_az, stride_ah = get_strides(alibi_slopes)

    stride_oz, stride_om, _, stride_og, stride_oh, stride_ok = get_strides(out_shape)
    stride_lse_zhg, stride_lse_m = get_strides(lse_shape)

    metaparams = dict(
        q_heads=heads_per_group_q,
        kv_heads=heads_per_group_k,
        GroupQuery=n_group_q,
        BLOCK_DIM=dim_padded,
        ACTUAL_BLOCK_DIM=dim_k,
        BOUNDS_CHECKS_N=(split_size % BLOCK_N) > 0 or use_cache_seqlens,
        USE_CACHE_SEQLENs=use_cache_seqlens,
        USE_CACHE_BATCH_IDX=cache_batch_idx is not None,
        NEW_KV=new_kv,
        IS_GQA=is_gqa,
        IS_CAUSAL=causal,
        USE_ALIBI=False if alibi_slopes is None else True,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    out_splitk, metadata = triton_call(
        q,
        k,
        v,
        sm_scale,
        k_new if k_new is not None else 1,
        v_new if v_new is not None else 1,
        cache_seqlens if cache_seqlens is not None else 1,
        cache_batch_idx if cache_batch_idx is not None else 1,
        alibi_slopes if alibi_slopes is not None else 1,
        stride_qz,
        stride_qm,
        stride_qg,
        stride_qh,
        stride_qd,
        stride_kz,
        stride_kn,
        stride_kg,
        stride_kh,
        stride_kd,
        stride_vz,
        stride_vn,
        stride_vg,
        stride_vh,
        stride_vd,
        stride_osk_zhg,
        stride_osk_s,
        stride_osk_m,
        stride_mzhg,
        stride_m2,
        stride_ms,
        stride_kn_z,
        stride_kn_n,
        stride_kn_g,
        stride_kn_h,
        stride_kn_d,
        stride_vn_z,
        stride_vn_n,
        stride_vn_g,
        stride_vn_h,
        stride_vn_d,
        stride_az,
        stride_ah,
        seqlen_q,
        seqlen_k,
        k_new.shape[1] if new_kv else 1,
        split_size,
        kernel=_fwd_kernel_stage1,
        name="ejgpu:attn_forward:fwd_kernel_splitK",
        grid=lambda META: grid,
        out_shape=[
            jax.ShapeDtypeStruct(shape=out_splitk_shape, dtype=jnp.float32),
            jax.ShapeDtypeStruct(shape=metadata_shape, dtype=jnp.float32),
        ],
        num_warps=num_warps,
        num_stages=1,
        **metaparams,
    )

    splitK_pow2 = next_power_of_2(split_k)
    use_mask = splitK_pow2 > split_k
    if batch_size * n_group_q * heads_per_group_q * seqlen_q >= 512:
        k_block_num = 1
    else:
        k_block_num = 2
    assert dim_padded % k_block_num == 0
    k_block_size = dim_padded // k_block_num
    grid = (batch_size * n_group_q * heads_per_group_q, seqlen_q, k_block_num)

    metaparams = dict(
        H=heads_per_group_q,
        G=n_group_q,
        split_k=split_k,
        splitK_pow2=splitK_pow2,
        use_mask=use_mask,
        IS_CAUSAL=causal,
        BLOCK_SIZE=k_block_size,
    )
    out, lse = triton_call(
        out_splitk,
        metadata,
        stride_osk_zhg,
        stride_osk_s,
        stride_osk_m,
        stride_osk_k,
        stride_mzhg,
        stride_m2,
        stride_ms,
        stride_mm,
        stride_oz,
        stride_oh,
        stride_og,
        stride_om,
        stride_lse_zhg,
        kernel=_fwd_kernel_stage2,
        name="ejgpu:attn_forward:_fwd_kernel_stage2",
        grid=lambda META: grid,
        out_shape=[
            jax.ShapeDtypeStruct(shape=out_shape, dtype=q.shape),
            jax.ShapeDtypeStruct(shape=lse_shape, dtype=jnp.float32),
        ],
        num_warps=4,
        **metaparams,
    )

    lse = lse.reshape(batch_size, n_group_q, heads_per_group_q, seqlen_q)
    if q.ndim == 4:
        assert n_group_q == 1
        out = out[:, :, 0]
        lse = lse[:, 0]
    if seqlen_k == 0:
        out = jnp.zeros_like(out)
    out = out.reshape(batch_size, heads_per_group_q * n_group_q, -1, dim_padded)

    if original_layout == "bshd":
        out = out.reshape(batch_size, seqlen_q, -1, dim_padded)

    return narrow(out, -1, 0, dim_k), lse


def flash_attn_varlen(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sm_scale: float | None = None,
    causal: bool = True,
    layout: Layouts = "bshd",
    b: jax.Array | None = None,
    dropout_p: float = 0.0,
    alibi_slopes: jax.Array | None = None,
    cu_seqlens_q: jax.Array | list[int] | None = None,
    cu_seqlens_k: jax.Array | list[int] | None = None,
    max_seqlens_q: int | None = None,
    max_seqlens_k: int | None = None,
) -> jax.Array:
    """Performs Flash Attention forward pass for variable-length sequences (prefill stage).

    This function is optimized for the prefill step in autoregressive generation,
    where an entire prompt is processed at once. It can handle both standard
    padded inputs and packed variable-length inputs for maximum efficiency.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        sm_scale: Scaling factor for softmax. If None, defaults to `1/sqrt(head_dim)`.
        causal: If True, applies causal attention masking.
        layout: The memory layout of the input tensors.
            - "bshd": (batch, seqlen, num_heads, head_dim)
            - "bhsd": (batch, num_heads, seqlen, head_dim)
            - "thd": (total_tokens, num_heads, head_dim) for packed varlen.
        b: Optional bias tensor to be added to attention scores.
        dropout_p: Dropout probability.
        alibi_slopes: Optional tensor of ALiBi slopes for positional bias.
        cu_seqlens_q: Cumulative sequence lengths for Q. Required for "thd" layout
            to identify sequence boundaries in the packed tensor.
            Example: `[0, 5, 12]` for two sequences of length 5 and 7.
        cu_seqlens_k: Cumulative sequence lengths for K. Required for "thd" layout.
        max_seqlens_q: Maximum sequence length for Q. Used for non-varlen layouts.
        max_seqlens_k: Maximum sequence length for K. Used for non-varlen layouts.

    Returns:
        - o (jax.Array): The attention output tensor, with the same shape as `q`.
    """
    return _flash_attn_varlen(
        q=q,
        k=k,
        v=v,
        sm_scale=sm_scale,
        causal=causal,
        layout=layout,
        b=b,
        dropout_p=dropout_p,
        alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlens_q=max_seqlens_q,
        max_seqlens_k=max_seqlens_k,
    )[0]


def flash_attn_varlen_decode(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sm_scale: float | None,
    causal: bool,
    alibi_slopes: jax.Array | None,
    layout: Layouts | None,
    cache_seqlens: jax.Array | int | list[int],
    cache_batch_idx: int | list[int],
    new_kv: jax.Array | None,
    k_new: jax.Array | None,
    v_new: jax.Array | None,
):
    """Performs Flash Attention forward pass for the decoding stage.

    This function is optimized for the single-token (or few-token) generation
    step in an autoregressive model. It efficiently computes attention by
    attending to a large KV cache. It uses a two-step split-K algorithm:
    1. A first kernel computes partial attention results in parallel over the K/V
       sequence length.
    2. A second kernel reduces these partial results to get the final output.
    This approach supports GQA/MQA and flexible KV cache management.

    Args:
        q: Query tensor, typically with a short sequence length (e.g., 1).
        k: Key tensor, representing the historical KV cache.
        v: Value tensor, representing the historical KV cache.
        sm_scale: Scaling factor for softmax.
        causal: If True, applies causal attention masking.
        alibi_slopes: Optional tensor of ALiBi slopes for positional bias.
        layout: The original memory layout of inputs ('bshd', 'bhsd'). The
            function internally converts inputs to 'bsghd' for processing.
        cache_seqlens: Tensor or list of actual sequence lengths for each item
            in the KV cache batch. Essential for correct attention computation
            on padded caches.
        cache_batch_idx: Maps request indices to batch indices in the KV cache,
            enabling advanced batching strategies like paged attention.
        new_kv: A flag indicating that new K/V tensors are provided.
        k_new: New key tensors to be used in the computation.
        v_new: New value tensors to be used in the computation.

    Returns:
        A tuple containing:
            - out (jax.Array): The attention output tensor. The shape is derived
              from the query shape, with the head dimension matching the input.
            - lse (jax.Array): The log-sum-exp of the attention scores.
    """
    return _flash_attn_varlen_decode(
        q,
        k,
        v,
        sm_scale,
        causal,
        alibi_slopes,
        layout,
        cache_seqlens,
        cache_batch_idx,
        new_kv,
        k_new,
        v_new,
    )
