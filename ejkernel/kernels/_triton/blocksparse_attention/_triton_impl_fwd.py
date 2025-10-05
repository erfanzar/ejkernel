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
from typing import Any

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jaxtyping import Array, Bool, Float, Int
from triton import Config

from ejkernel.callib import triton_call
from ejkernel.utils import dtype_index, get_strides

from ._utilities import create_blocksparse_metadata, padded_load


def config_prune_kernel(
    configs: list[Config],
    named_args: dict[str, Any],
    **kwargs: Any,
) -> list[Config]:
    """Prune Triton auto-tuning configurations based on problem size."""
    kept_configs = []
    for config in configs:
        largerst_m = config.kwargs["BLOCK_M"] > named_args["N_CTX"]
        largerst_n = config.kwargs["BLOCK_N"] > named_args["N_CTX"]
        if largerst_m or largerst_n:
            pass
        else:
            kept_configs.append(config)
    if kept_configs:
        return kept_configs
    return [
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    ]


@triton.autotune(
    configs=[
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
        Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
    ],
    key=["Z", "H", "N_CTX", "BLOCK_DMODEL"],
    prune_configs_by={"early_config_prune": config_prune_kernel},
)
@triton.heuristics(
    {
        "EVEN_M_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_M"] == 0,
        "EVEN_N_BLOCK": lambda kwargs: kwargs["N_CTX"] % kwargs["BLOCK_N"] == 0,
    }
)
@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h,
    layout_crow_stride_m,
    layout_col_stride_h,
    layout_col_stride_m,
    TMP,
    L,
    M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug
    Out,
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
    PAST_LEN,
    Q_ROUNDED_LEN,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    INFERENCE: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
):
    """Blocksparse attention forward kernel.

    Computes attention with block-level sparsity pattern using compressed
    storage format for efficient memory access.

    Args:
        Q, K, V: Query, Key, Value tensors
        sm_scale: Softmax scaling factor
        layout_crow_ptr, layout_col_ptr: Compressed row storage pointers
        layout_crow_stride_*, layout_col_stride_*: Strides for layout tensors
        TMP, L, M: Temporary buffers for computation
        Out: Output tensor
        stride_*: Memory strides for tensors
        Z, H, N_CTX: Batch, heads, context dimensions
        PAST_LEN: Past context length for caching
        Q_ROUNDED_LEN: Rounded query length for alignment
        BLOCK_M, BLOCK_N, BLOCK_DMODEL: Block dimensions
        EVEN_M_BLOCK, EVEN_N_BLOCK: Whether blocks are aligned
        INFERENCE: Whether in inference mode
        NUM_DBLOCKS: Number of head dimension blocks
    """
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointer to m and l
    t_ptrs = TMP + off_hz * Q_ROUNDED_LEN + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # Load q: it will stay in SRAM throughout
    if EVEN_M_BLOCK:
        q = tl.load(q_ptrs)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m[:, None] < Q_LEN)

    # Load layout pointers for this block row
    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    # Loop over sparse blocks in this row
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_col_ptr + off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
        start_n = col_idx * BLOCK_N

        # Compute qk
        if EVEN_N_BLOCK:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < N_CTX)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd)
            else:
                k = tl.load(
                    k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd,
                    mask=offs_n[None, :] + start_n < N_CTX,
                )
            qk += tl.dot(q2, k)

        qk *= sm_scale

        # Apply causal mask if needed
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, float("-inf"))

        # Compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        # Update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # Update output accumulator
        # Scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # Scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        if NUM_DBLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)

        # Update acc
        if EVEN_N_BLOCK:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < N_CTX)
        acc += tl.dot(p, v)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd)
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd,
                    mask=offs_n[:, None] + start_n < N_CTX,
                )
            acc2 += tl.dot(p, v)

        # Update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # Write back l and m
    if not INFERENCE:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        if EVEN_M_BLOCK:
            tl.store(l_ptrs, l_i)
            tl.store(m_ptrs, m_i)
        else:
            tl.store(l_ptrs, l_i, mask=offs_m < Q_LEN)
            tl.store(m_ptrs, m_i, mask=offs_m < Q_LEN)

    # Initialize pointers to output
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < Q_LEN)
    if NUM_DBLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2, mask=offs_m[:, None] < Q_LEN)


def _fwd_attention_kernel_call(
    q: Float[Array, "batch seq_len_q num_heads head_dim"],
    k: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    v: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    layout_crow_ptr: Int[Array, "num_heads num_blocks_q + 1"] | Int[Array, "num_blocks_q + 1"],
    layout_col_indices: Int[Array, "num_heads nnz"] | Int[Array, "nnz"],
    softmax_scale: float | None = None,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
) -> tuple[Float[Array, "batch seq_len_q num_heads head_dim"], Float[Array, "batch num_heads seq_len_q"]]:
    """Call the blocksparse attention forward kernel.

    Args:
        q: Query tensor [batch, seq_len_q, num_heads, head_dim]
        k: Key tensor [batch, seq_len_k, num_kv_heads, head_dim]
        v: Value tensor [batch, seq_len_k, num_kv_heads, head_dim]
        layout_crow_ptr: Compressed row pointers for sparse layout
        layout_col_indices: Column indices for non-zero blocks
        softmax_scale: Scaling factor for attention scores
        causal: Whether to apply causal masking
        block_m: Block size for queries
        block_n: Block size for keys

    Returns:
        tuple: (attention_output, log_sum_exp) for backward pass
    """
    batch, seq_len_q, num_heads_q, head_dim = q.shape
    _, seq_len_k, num_heads_kv, _ = k.shape

    assert num_heads_q % num_heads_kv == 0
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [jnp.float16, jnp.bfloat16]

    softmax_scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale

    # Determine layout dimensions
    homo_head = layout_crow_ptr.ndim == 1
    if homo_head:
        # Broadcast layout to all heads
        layout_crow_stride_h = 0
        layout_col_stride_h = 0
    else:
        layout_crow_stride_h = layout_crow_ptr.shape[1]
        layout_col_stride_h = layout_col_indices.shape[1]

    layout_crow_stride_m = 1
    layout_col_stride_m = 1

    # Calculate strides
    qz, qm, qh, qd = get_strides(q.shape)
    kz, kn, kh, kd = get_strides(k.shape)
    vz, vn, vh, vd = get_strides(v.shape)
    oz, om, oh, od = get_strides(q.shape)

    # Prepare metadata
    q_rounded_len = math.ceil(seq_len_q / 128) * 128
    block_headdim = max(triton.next_power_of_2(head_dim), 16)
    padded_heads = block_headdim > head_dim
    num_dblocks = (head_dim + block_headdim - 1) // block_headdim

    metaparams = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DMODEL=block_headdim,
        INFERENCE=False,
        NUM_DBLOCKS=num_dblocks,
    )

    # Temporary buffers
    tmp_shape = (batch * num_heads_q * q_rounded_len,)
    tmp = jnp.zeros(tmp_shape, dtype=jnp.float32)
    lse = jnp.zeros((batch * num_heads_q * seq_len_q,), dtype=jnp.float32)
    m = jnp.zeros((batch * num_heads_q * seq_len_q,), dtype=jnp.float32)

    out_shape = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct((batch, num_heads_q, seq_len_q), jnp.float32),
    ]

    num_warps = 4 if head_dim <= 64 else 8
    num_blocks_q = math.ceil(seq_len_q / block_m)

    out, lse_out = triton_call(
        q,
        k,
        v,
        softmax_scale,
        layout_crow_ptr,
        layout_col_indices,
        layout_crow_stride_h,
        layout_crow_stride_m,
        layout_col_stride_h,
        layout_col_stride_m,
        tmp,
        lse,
        m,
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
        num_heads_q,
        seq_len_k,
        0,  # PAST_LEN
        q_rounded_len,
        dtype_index(q),
        kernel=_fwd_kernel,
        out_shape=out_shape,
        grid=lambda META: (num_blocks_q, batch * num_heads_q),
        num_warps=num_warps,
        name="triton::ops::blocksparse_attn_fwd",
        **metaparams,
    )

    # Reshape lse to match expected output
    lse_out = lse_out.reshape(batch, num_heads_q, seq_len_q)

    return out, lse_out