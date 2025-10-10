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

from ejkernel.callib import cdiv, triton_call

from ....xla_utils.utils import prepare_lens


@triton.jit
def nsa_kernel_mask(
    block_indices,
    block_counts,
    block_mask,
    SEQUENCE: tl.constexpr,
    HEAD: tl.constexpr,
    SIZE: tl.constexpr,
    BLOCKSIZE: tl.constexpr,
    NUM_SEQS: tl.constexpr,
    USE_BLOCK_COUNTS: tl.constexpr,
):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // SIZE, i_hs % SIZE

    b_i = tl.load(block_indices + i_b * SEQUENCE * HEAD * SIZE + i_t * HEAD * SIZE + i_h * SIZE + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BLOCKSIZE <= i_t and i_s < tl.load(block_counts + i_b * SEQUENCE * HEAD + i_t * HEAD + i_h)
    else:
        b_m = b_i * BLOCKSIZE <= i_t

    if b_i < NUM_SEQS and b_i >= 0:
        tl.store(
            block_mask + i_b * SEQUENCE * HEAD * NUM_SEQS + i_t * HEAD * NUM_SEQS + i_h * NUM_SEQS + b_i,
            b_m.to(block_mask.dtype.element_ty),
        )


def nsa_block_mask(
    block_indices: jax.Array,
    block_counts: jax.Array | int,
    cu_seqlens: jax.Array,
    block_size: int,
):
    B, SEQUENCE, HEAD, SIZE = block_indices.shape
    BLOCKSIZE = block_size
    if cu_seqlens is not None:
        NUM_SEQS = cdiv(prepare_lens(cu_seqlens).max(), BLOCKSIZE)
    else:
        NUM_SEQS = cdiv(SEQUENCE, BLOCKSIZE)

    outputs = [jax.ShapeDtypeStruct((B, SEQUENCE, HEAD, NUM_SEQS), dtype="b1")]

    metaparams = dict(
        SEQUENCE=SEQUENCE,
        HEAD=HEAD,
        SIZE=SIZE,
        BLOCKSIZE=BLOCKSIZE,
        NUM_SEQS=NUM_SEQS,
        USE_BLOCK_COUNTS=isinstance(block_counts, jax.Array),
    )

    (block_mask,) = triton_call(
        block_indices,
        block_counts,
        kernel=nsa_kernel_mask,
        grid=lambda META: (SEQUENCE, B, HEAD * SIZE),
        out_shape=outputs,
        name="ejkernel::triton::sparse_attn_mask",
        **metaparams,
    )

    return block_mask
