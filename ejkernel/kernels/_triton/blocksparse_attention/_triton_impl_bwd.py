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

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Float, Int

from ejkernel.callib import triton_call
from ejkernel.utils import dtype_index, get_strides


@triton.heuristics(
    {
        "EVEN_M_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_M"] == 0,
    }
)
@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    L,  # assume contiguous for Out, DO, L, NewDO, Delta layout
    NewDO,
    Delta,
    N_CTX,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
):
    """Preprocess gradients for backward pass.

    Normalizes output gradients by log-sum-exp and computes delta values
    for stable gradient computation.

    Args:
        Out: Forward pass output
        DO: Output gradients
        L: Log-sum-exp from forward pass
        NewDO: Normalized output gradients (output)
        Delta: Delta values for gradient computation (output)
        N_CTX: Context length
        BLOCK_M: Block size for M dimension
        D_HEAD: Head dimension
        EVEN_M_BLOCK: Whether blocks are aligned
    """
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)

    # Load
    if EVEN_M_BLOCK:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
    else:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_d[None, :], mask=off_m[:, None] < N_CTX).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_d[None, :], mask=off_m[:, None] < N_CTX).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)

    # Compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)

    # Write back
    if EVEN_M_BLOCK:
        tl.store(NewDO + off_m[:, None] * D_HEAD + off_d[None, :], do)
    else:
        tl.store(NewDO + off_m[:, None] * D_HEAD + off_d[None, :], do, mask=off_m[:, None] < N_CTX)
    tl.store(Delta + off_m, delta)


@triton.heuristics(
    {
        "EVEN_M_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_M"] == 0,
        "EVEN_N_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_N"] == 0,
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    layout_ccol_ptr,
    layout_row_ptr,
    layout_ccol_stride_h,
    layout_ccol_stride_m,
    layout_row_stride_h,
    layout_row_stride_m,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    M,
    D,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_oz,
    stride_oh,
    stride_om,
    stride_od,
    Z,
    H,
    N_CTX,
    num_block,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
):
    """Blocksparse attention backward kernel.

    Computes gradients for Q, K, V using the transpose of the sparse
    pattern from the forward pass.

    Args:
        Q, K, V: Input tensors from forward pass
        sm_scale: Softmax scaling factor
        layout_ccol_ptr, layout_row_ptr: Compressed column storage pointers
        layout_ccol_stride_*, layout_row_stride_*: Strides for layout tensors
        Out: Forward pass output
        DO: Output gradients
        DQ, DK, DV: Gradient outputs
        L, M, D: Saved tensors from forward pass
        stride_*: Memory strides
        Z, H, N_CTX: Batch, heads, context dimensions
        num_block: Number of blocks
        BLOCK_M, BLOCK_N, BLOCK_DMODEL: Block dimensions
        EVEN_M_BLOCK, EVEN_N_BLOCK: Whether blocks are aligned
        NUM_DBLOCKS: Number of head dimension blocks
    """
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_oz + off_h * stride_oh
    DK += off_z * stride_oz + off_h * stride_oh
    DV += off_z * stride_oz + off_h * stride_oh

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Initialize pointers to value-like data
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # Pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX

    # Initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # k and v stay in SRAM throughout
    if EVEN_N_BLOCK:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dv2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        if EVEN_N_BLOCK:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd)
        else:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd, mask=offs_n[:, None] < N_CTX)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd, mask=offs_n[:, None] < N_CTX)

    # Load layout pointers for this block column
    layout_ptr = layout_ccol_ptr + off_h * layout_ccol_stride_h + start_n * layout_ccol_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_ccol_stride_m).to(tl.int32)

    # Loop over sparse blocks in this column
    for row_idx_idx in range(start_l, end_l):
        row_idx = tl.load(layout_row_ptr + off_h * layout_row_stride_h + row_idx_idx * layout_row_stride_m).to(tl.int32)
        start_m = row_idx * BLOCK_M

        offs_m_curr = start_m + offs_m
        q_ptrs = Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
        dq_ptrs = DQ + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)

        # Load q, do on-chip
        if EVEN_M_BLOCK:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)

        # Re-compute p = softmax(qk, dim=-1).T
        qk = tl.dot(q, tl.trans(k))

        if NUM_DBLOCKS >= 2:
            if EVEN_M_BLOCK:
                q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
            else:
                q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m_curr[:, None] < N_CTX)
            qk += tl.dot(q2, tl.trans(k2))

        # Apply causal mask
        qk += tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), 0, float("-inf"))

        if EVEN_M_BLOCK:
            m = tl.load(m_ptrs + offs_m_curr)
        else:
            m = tl.load(m_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
        p = tl.exp(qk * sm_scale - m[:, None])

        # Compute dv
        if EVEN_M_BLOCK:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)

        if NUM_DBLOCKS >= 2:
            if EVEN_M_BLOCK:
                do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od)
            else:
                do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od, mask=offs_m_curr[:, None] < N_CTX)

        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

        if NUM_DBLOCKS >= 2:
            dv2 += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do2)

        # Compute dp = dot(v, do)
        if EVEN_M_BLOCK:
            Di = tl.load(D_ptrs + offs_m_curr)
        else:
            Di = tl.load(D_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, tl.trans(v))

        if NUM_DBLOCKS >= 2:
            dp += tl.dot(do2, tl.trans(v2))

        # Compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # Compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        if NUM_DBLOCKS >= 2:
            dk2 += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q2)

        # Compute dq
        dq = tl.dot(ds.to(Q.dtype.element_ty), k)
        if EVEN_M_BLOCK:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < N_CTX)

        if NUM_DBLOCKS >= 2:
            dq2 = tl.dot(ds.to(Q.dtype.element_ty), k2)
            dq_ptrs2 = dq_ptrs + BLOCK_DMODEL * stride_od
            if EVEN_M_BLOCK:
                tl.atomic_add(dq_ptrs2, dq2)
            else:
                tl.atomic_add(dq_ptrs2, dq2, mask=offs_m_curr[:, None] < N_CTX)

    # Write back dv and dk
    dv_ptrs = DV + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    dk_ptrs = DK + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    if EVEN_N_BLOCK:
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    else:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dv_ptrs2 = dv_ptrs + BLOCK_DMODEL * stride_od
        dk_ptrs2 = dk_ptrs + BLOCK_DMODEL * stride_od
        if EVEN_N_BLOCK:
            tl.store(dv_ptrs2, dv2)
            tl.store(dk_ptrs2, dk2)
        else:
            tl.store(dv_ptrs2, dv2, mask=offs_n[:, None] < N_CTX)
            tl.store(dk_ptrs2, dk2, mask=offs_n[:, None] < N_CTX)


def _bwd_attention_kernel_call(
    dO: Float[Array, "batch seq_len num_heads head_dim"],
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_heads head_dim"],
    v: Float[Array, "batch seq_len num_heads head_dim"],
    o: Float[Array, "batch seq_len num_heads head_dim"],
    M: Float[Array, "batch num_heads seq_len"],
    layout_ccol_ptr: Int[Array, "num_heads num_blocks_k + 1"] | Int[Array, "num_blocks_k + 1"],
    layout_row_indices: Int[Array, "num_heads nnz"] | Int[Array, "nnz"],
    softmax_scale: float | None = None,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[
    Float[Array, "batch seq_len num_heads head_dim"],
    Float[Array, "batch seq_len num_heads head_dim"],
    Float[Array, "batch seq_len num_heads head_dim"],
]:
    """Call the blocksparse attention backward kernel.

    Args:
        dO: Gradient of output [batch, seq_len, num_heads, head_dim]
        q: Query tensor from forward pass
        k: Key tensor from forward pass
        v: Value tensor from forward pass
        o: Output from forward pass
        M: Log-sum-exp from forward pass [batch, num_heads, seq_len]
        layout_ccol_ptr: Compressed column pointers for sparse layout
        layout_row_indices: Row indices for non-zero blocks
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        block_m: Block size for M dimension
        block_n: Block size for N dimension

    Returns:
        tuple: (dq, dk, dv) gradients for query, key, and value
    """
    batch, seq_len, num_heads, head_dim = q.shape

    assert q.shape == k.shape == v.shape == o.shape == dO.shape
    assert q.dtype == k.dtype == v.dtype == o.dtype == dO.dtype
    assert q.dtype in [jnp.float16, jnp.bfloat16]

    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

    # Determine layout dimensions
    homo_head = layout_ccol_ptr.ndim == 1
    if homo_head:
        layout_ccol_stride_h = 0
        layout_row_stride_h = 0
    else:
        layout_ccol_stride_h = layout_ccol_ptr.shape[1]
        layout_row_stride_h = layout_row_indices.shape[1]

    layout_ccol_stride_m = 1
    layout_row_stride_m = 1

    # Reshape M to match memory layout
    M_flat = M.transpose(0, 2, 1).reshape(-1)  # [batch * num_heads * seq_len]

    # Calculate strides
    qz, qm, qh, qd = get_strides(q.shape)
    kz, kn, kh, kd = get_strides(k.shape)
    vz, vn, vh, vd = get_strides(v.shape)
    oz, om, oh, od = get_strides(o.shape)

    block_headdim = max(triton.next_power_of_2(head_dim), 16)
    num_dblocks = (head_dim + block_headdim - 1) // block_headdim

    # Preprocessing step
    delta = jnp.zeros((batch * num_heads * seq_len,), dtype=jnp.float32)
    new_do = jnp.zeros_like(dO)

    # Flatten tensors for preprocessing
    o_flat = o.reshape(-1, head_dim)
    do_flat = dO.reshape(-1, head_dim)
    new_do_flat = new_do.reshape(-1, head_dim)

    # Run preprocessing kernel
    num_blocks_m = math.ceil(seq_len / block_m)
    triton_call(
        o_flat,
        do_flat,
        M_flat,
        new_do_flat,
        delta,
        seq_len,
        dtype_index(q),
        kernel=_bwd_preprocess,
        out_shape=[
            jax.ShapeDtypeStruct(new_do_flat.shape, new_do_flat.dtype),
            jax.ShapeDtypeStruct(delta.shape, delta.dtype),
        ],
        grid=(num_blocks_m * batch * num_heads,),
        BLOCK_M=block_m,
        D_HEAD=head_dim,
        name="triton::ops::blocksparse_attn_bwd_preprocess",
    )

    # Initialize gradients
    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    metaparams = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_headdim,
        NUM_DBLOCKS=num_dblocks,
    )

    num_blocks_n = math.ceil(seq_len / block_n)
    num_warps = 4 if head_dim <= 64 else 8

    # Run main backward kernel
    dq, dk, dv = triton_call(
        q,
        k,
        v,
        softmax_scale,
        layout_ccol_ptr,
        layout_row_indices,
        layout_ccol_stride_h,
        layout_ccol_stride_m,
        layout_row_stride_h,
        layout_row_stride_m,
        o,
        new_do,
        dq,
        dk,
        dv,
        M_flat,
        M_flat,
        delta,
        qz,
        qh,
        qm,
        qd,
        kz,
        kh,
        kn,
        kd,
        vz,
        vh,
        vn,
        vd,
        oz,
        oh,
        om,
        od,
        batch,
        num_heads,
        seq_len,
        num_blocks_n,
        dtype_index(q),
        kernel=_bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(dq.shape, dq.dtype),
            jax.ShapeDtypeStruct(dk.shape, dk.dtype),
            jax.ShapeDtypeStruct(dv.shape, dv.dtype),
        ],
        grid=lambda META: (num_blocks_n, batch * num_heads),
        num_warps=num_warps,
        name="triton::ops::blocksparse_attn_bwd",
        **metaparams,
    )

    return dq, dk, dv