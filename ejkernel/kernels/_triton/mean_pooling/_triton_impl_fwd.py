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
from jaxtyping import Array, Float, Int

from ejkernel.callib import cdiv, triton_call
from ejkernel.xla_utils.utils import prepare_chunk_indices


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_DIM": BLOCK_DIM}, num_warps=num_warps)
        for BLOCK_DIM in [16, 32, 64, 128]
        for num_warps in [1, 2, 4, 8]
    ],
    key=["BLOCK_SEQ"],
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] != 1})
@triton.jit
def fwd_kernel(
    x,
    cu_seqlens,
    chunk_indices,
    o,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Triton kernel for forward pass of mean pooling operation.

    Computes mean pooling over sequence dimension in chunks for efficiency.
    Supports both fixed-length and variable-length sequences.

    Args:
        x: Input tensor pointer
        cu_seqlens: Cumulative sequence lengths for variable-length mode
        chunk_indices: Pre-computed chunk indices for variable-length mode
        o: Output tensor pointer
        SEQUENCE: Sequence length (compile-time constant)
        HEAD: Number of attention heads
        DIM: Hidden dimension size
        BLOCK_SEQ: Block size for sequence dimension
        BLOCK_DIM: Block size for hidden dimension
        IS_VARLEN: Whether using variable-length sequences
    """
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // HEAD, i_bh % HEAD
    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        SEQUENCE = eos - bos
        NUM_SEQ_BLOCK = tl.cdiv(SEQUENCE, BLOCK_SEQ)
    else:
        NUM_SEQ_BLOCK = tl.cdiv(SEQUENCE, BLOCK_SEQ)
        i_tg = i_b * NUM_SEQ_BLOCK + i_t
        bos, eos = i_b * SEQUENCE, i_b * SEQUENCE + SEQUENCE

    p_x = tl.make_block_ptr(
        x + (bos * HEAD + i_h) * DIM,
        (SEQUENCE, DIM),
        (HEAD * DIM, 1),
        (i_t * BLOCK_SEQ, i_d * BLOCK_DIM),
        (BLOCK_SEQ, BLOCK_DIM),
        (1, 0),
    )
    p_o = tl.make_block_ptr(o + (i_tg * HEAD + i_h) * DIM, (DIM,), (1,), (i_d * BLOCK_DIM,), (BLOCK_DIM,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)

    b_o = tl.sum(b_x, axis=0) / min(BLOCK_SEQ, SEQUENCE - i_t * BLOCK_SEQ)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0,))


def fwd_triton_impl(
    x: Float[Array, "batch seq_len hidden_dim"],
    chunk_size: int,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> Float[Array, "batch hidden_dim"]:
    """Execute mean pooling forward pass using Triton kernel.

    Launches the Triton kernel for efficient mean pooling computation,
    handling both standard and variable-length sequence formats.

    Args:
        x: Input tensor of shape [batch, seq_len, heads, dim] or packed format
        chunk_size: Size of chunks for processing sequences
        cu_seqlens: Optional cumulative sequence lengths for variable-length mode

    Returns:
        jax.Array: Mean-pooled output tensor with reduced sequence dimension
    """
    Z, SEQUENCE, HEAD, DIM = x.shape
    BLOCK_SEQ = chunk_size
    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    NUM_SEQ_BLOCK = cdiv(SEQUENCE, BLOCK_SEQ) if cu_seqlens is None else len(chunk_indices)
    metaparams = dict(SEQUENCE=SEQUENCE, HEAD=HEAD, DIM=DIM, BLOCK_SEQ=BLOCK_SEQ)
    (o,) = triton_call(
        x,
        cu_seqlens if cu_seqlens is not None else 1,
        chunk_indices if chunk_indices is not None else 1,
        kernel=fwd_kernel,
        out_shape=[jax.ShapeDtypeStruct((Z, NUM_SEQ_BLOCK, HEAD, DIM), x.dtype)],
        name="ejkernel::triton::mean_pooling_fwd",
        grid=lambda META: (cdiv(DIM, META["BLOCK_DIM"]), NUM_SEQ_BLOCK, Z * HEAD),
        **metaparams,
    )
    return o
