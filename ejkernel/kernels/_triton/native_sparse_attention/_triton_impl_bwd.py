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


import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import cdiv, next_power_of_2, triton_call

from ....xla_utils.utils import prepare_chunk_indices
from ._utilities import nsa_block_mask


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BLOCKSIZE", "BLOCKSIZE_K", "BLOCKSIZE_V"],
)
@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] != 1,
        "USE_BLOCK_COUNTS": lambda args: isinstance(args["block_counts"], jax.Array),
    }
)
@triton.jit
def nsa_bwd_kernel_dq(
    q,
    k,
    v,
    lse,
    delta,
    do,
    softmax_scale,
    block_indices,
    block_counts,
    cu_seqlens,
    token_indices,
    dq,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    IndicesSize: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    q += (bos + i_t) * Q_HEADS * BLOCK_DIMK
    do += (bos + i_t) * Q_HEADS * BLOCK_DIMV
    lse += (bos + i_t) * Q_HEADS
    delta += (bos + i_t) * Q_HEADS
    dq += (i_v * scope + bos + i_t) * Q_HEADS * BLOCK_DIMK
    block_indices += (bos + i_t) * KV_HEADS * IndicesSize + i_h * IndicesSize

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * KV_HEADS + i_h)
    else:
        NS = IndicesSize

    k += (bos * KV_HEADS + i_h) * BLOCK_DIMK
    v += (bos * KV_HEADS + i_h) * BLOCK_DIMV

    p_q = tl.make_block_ptr(
        q, (Q_HEADS, BLOCK_DIMK), (BLOCK_DIMK, 1), (i_h * QK_GROUPS, 0), (QK_GROUPS, BLOCKSIZE_K), (1, 0)
    )
    p_dq = tl.make_block_ptr(
        dq, (Q_HEADS, BLOCK_DIMK), (BLOCK_DIMK, 1), (i_h * QK_GROUPS, 0), (QK_GROUPS, BLOCKSIZE_K), (1, 0)
    )

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * softmax_scale).to(b_q.dtype)

    p_do = tl.make_block_ptr(
        do,
        (Q_HEADS, BLOCK_DIMV),
        (BLOCK_DIMV, 1),
        (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
        (QK_GROUPS, BLOCKSIZE_V),
        (1, 0),
    )
    p_lse = lse + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
    p_delta = delta + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)

    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_lse = tl.load(p_lse)
    b_delta = tl.load(p_delta)
    b_dq = tl.zeros([QK_GROUPS, BLOCKSIZE_K], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BLOCKSIZE
        if i_s <= i_t and i_s >= 0:
            p_k = tl.make_block_ptr(
                k, (BLOCK_DIMK, SEQUENCE), (1, KV_HEADS * BLOCK_DIMK), (0, i_s), (BLOCKSIZE_K, BLOCKSIZE), (0, 1)
            )
            p_v = tl.make_block_ptr(
                v,
                (BLOCK_DIMV, SEQUENCE),
                (1, KV_HEADS * BLOCK_DIMV),
                (i_v * BLOCKSIZE_V, i_s),
                (BLOCKSIZE_V, BLOCKSIZE),
                (0, 1),
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_v = tl.load(p_v, boundary_check=(0, 1))

            b_s = tl.dot(b_q, b_k)
            b_p = tl.exp(b_s - b_lse[:, None])
            b_p = tl.where((i_t >= (i_s + tl.arange(0, BLOCKSIZE)))[None, :], b_p, 0)

            b_dp = tl.dot(b_do, b_v)
            b_ds = b_p * (b_dp.to(tl.float32) - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))
    b_dq *= softmax_scale

    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4]],
    key=["BLOCKSIZE", "BLOCKSIZE_K", "BLOCKSIZE_V"],
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def nsa_bwd_kernel_dkv(
    q,
    k,
    v,
    lse,
    delta,
    do,
    block_mask,
    cu_seqlens,
    chunk_indices,
    softmax_scale,
    dk,
    dv,
    SEQUENCE: tl.constexpr,
    Z: tl.constexpr,
    KV_HEADS: tl.constexpr,
    Q_HEADS: tl.constexpr,
    QK_GROUPS: tl.constexpr,
    BLOCK_DIMK: tl.constexpr,
    BLOCK_DIMV: tl.constexpr,
    MaskSize: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    BLOCKSIZE_K: tl.constexpr,
    BLOCKSIZE_V: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // KV_HEADS, i_bh % KV_HEADS

    scope = Z * SEQUENCE
    if IS_VARLEN:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
    else:
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    p_k = tl.make_block_ptr(
        k + (bos * KV_HEADS + i_h) * BLOCK_DIMK,
        (SEQUENCE, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_s * BLOCKSIZE, 0),
        (BLOCKSIZE, BLOCKSIZE_K),
        (1, 0),
    )
    p_v = tl.make_block_ptr(
        v + (bos * KV_HEADS + i_h) * BLOCK_DIMV,
        (SEQUENCE, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_s * BLOCKSIZE, i_v * BLOCKSIZE_V),
        (BLOCKSIZE, BLOCKSIZE_V),
        (1, 0),
    )
    p_dk = tl.make_block_ptr(
        dk + (i_v * scope * KV_HEADS + bos * KV_HEADS + i_h) * BLOCK_DIMK,
        (SEQUENCE, BLOCK_DIMK),
        (KV_HEADS * BLOCK_DIMK, 1),
        (i_s * BLOCKSIZE, 0),
        (BLOCKSIZE, BLOCKSIZE_K),
        (1, 0),
    )
    p_dv = tl.make_block_ptr(
        dv + (bos * KV_HEADS + i_h) * BLOCK_DIMV,
        (SEQUENCE, BLOCK_DIMV),
        (KV_HEADS * BLOCK_DIMV, 1),
        (i_s * BLOCKSIZE, i_v * BLOCKSIZE_V),
        (BLOCKSIZE, BLOCKSIZE_V),
        (1, 0),
    )

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BLOCKSIZE, BLOCKSIZE_K], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BLOCKSIZE, BLOCKSIZE_V], dtype=tl.float32)

    for i in range(i_s * BLOCKSIZE, SEQUENCE):
        b_m = tl.load(block_mask + (bos + i) * KV_HEADS * MaskSize + i_h * MaskSize + i_s)
        if b_m:
            p_q = tl.make_block_ptr(
                q + (bos + i) * Q_HEADS * BLOCK_DIMK,
                (Q_HEADS, BLOCK_DIMK),
                (BLOCK_DIMK, 1),
                (i_h * QK_GROUPS, 0),
                (QK_GROUPS, BLOCKSIZE_K),
                (1, 0),
            )
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * softmax_scale).to(b_q.dtype)

            p_do = tl.make_block_ptr(
                do + (bos + i) * Q_HEADS * BLOCK_DIMV,
                (Q_HEADS, BLOCK_DIMV),
                (BLOCK_DIMV, 1),
                (i_h * QK_GROUPS, i_v * BLOCKSIZE_V),
                (QK_GROUPS, BLOCKSIZE_V),
                (1, 0),
            )
            p_lse = lse + (bos + i) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
            p_delta = delta + (bos + i) * Q_HEADS + i_h * QK_GROUPS + tl.arange(0, QK_GROUPS)
            b_do = tl.load(p_do, boundary_check=(0, 1))
            b_lse = tl.load(p_lse)
            b_delta = tl.load(p_delta)
            b_s = tl.dot(b_k, tl.trans(b_q))
            b_p = tl.exp(b_s - b_lse[None, :])
            b_p = tl.where((i >= (i_s * BLOCKSIZE + tl.arange(0, BLOCKSIZE)))[:, None], b_p, 0)
            b_dv += tl.dot(b_p.to(b_do.dtype), b_do)
            b_dp = tl.dot(b_v, tl.trans(b_do))
            b_ds = b_p * (b_dp - b_delta[None, :])
            b_dk += tl.dot(b_ds.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def bwd_kernel_preprocess(o, do, delta, Z: tl.constexpr, BLOCK_DIMV: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, Z)
    m_d = o_d < BLOCK_DIMV

    b_o = tl.load(o + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * BLOCK_DIMV + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def bwd_triton_impl(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    o: jax.Array,
    lse: jax.Array,
    do: jax.Array,
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    block_size: int = 64,
    softmax_scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    token_indices: jax.Array | None = None,
):
    Z, SEQUENCE, KV_HEADS, BLOCK_DIMK, BLOCK_DIMV, IndicesSize = (
        *k.shape,
        v.shape[-1],
        block_indices.shape[-1],
    )
    Q_HEADS = q.shape[2]
    QK_GROUPS = Q_HEADS // KV_HEADS
    BLOCKSIZE = block_size
    BLOCKSIZE_K = next_power_of_2(BLOCK_DIMK)
    BLOCKSIZE_V = min(128, next_power_of_2(v.shape[-1]))
    NumVBlocks = cdiv(BLOCK_DIMV, BLOCKSIZE_V)

    delta_shape = o.shape[:-1]
    delta_numel = jnp.prod(jnp.array(delta_shape))
    (delta,) = triton_call(
        o,
        do,
        kernel=bwd_kernel_preprocess,
        grid=lambda META: (delta_numel,),
        out_shape=[jax.ShapeDtypeStruct(delta_shape, dtype="f4")],
        name="ejkernel::triton::sparse_attn_compression_bwd_preprocess",
        Z=next_power_of_2(o.shape[-1]),
        BLOCK_DIMV=o.shape[-1],
    )

    outputs = [jax.ShapeDtypeStruct((NumVBlocks, *q.shape), dtype=q.dtype if NumVBlocks == 1 else "f4")]
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        Z=Z,
        KV_HEADS=KV_HEADS,
        Q_HEADS=Q_HEADS,
        QK_GROUPS=QK_GROUPS,
        BLOCK_DIMK=BLOCK_DIMK,
        BLOCK_DIMV=BLOCK_DIMV,
        IndicesSize=IndicesSize,
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
    )
    (dq,) = triton_call(
        q,
        k,
        v,
        lse,
        delta,
        do,
        softmax_scale,
        block_indices if block_indices is not None else 1,
        block_counts if block_counts is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        token_indices if token_indices is not None else 1,
        kernel=nsa_bwd_kernel_dq,
        grid=lambda META: (SEQUENCE, NumVBlocks, Z * KV_HEADS),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_bwd_dq",
        **metaparams,
    )
    dq = jnp.sum(dq, 0)

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BLOCKSIZE)
        NS = len(chunk_indices)
    else:
        chunk_indices = None
        NS = cdiv(SEQUENCE, BLOCKSIZE)

    block_mask = nsa_block_mask(block_indices, block_counts, cu_seqlens, block_size)
    outputs = [
        jax.ShapeDtypeStruct((NumVBlocks, *k.shape), dtype=k.dtype if NumVBlocks == 1 else "f4"),
        jax.ShapeDtypeStruct(v.shape, dtype=v.dtype),
    ]
    metaparams = dict(
        SEQUENCE=SEQUENCE,
        Z=Z,
        KV_HEADS=KV_HEADS,
        Q_HEADS=Q_HEADS,
        QK_GROUPS=QK_GROUPS,
        BLOCK_DIMK=BLOCK_DIMK,
        BLOCK_DIMV=BLOCK_DIMV,
        MaskSize=block_mask.shape[-1],
        BLOCKSIZE=BLOCKSIZE,
        BLOCKSIZE_K=BLOCKSIZE_K,
        BLOCKSIZE_V=BLOCKSIZE_V,
    )
    dk, dv = triton_call(
        q,
        k,
        v,
        lse,
        delta,
        do,
        block_mask if block_mask is not None else 1,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        softmax_scale,
        kernel=nsa_bwd_kernel_dkv,
        grid=lambda META: (NumVBlocks, NS, Z * KV_HEADS),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_bwd_dkdv",
        **metaparams,
    )
    dk = jnp.sum(dk, 0)
    return dq, dk, dv
