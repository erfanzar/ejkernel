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


"""Flash Attention TPU kernel."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8
MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array
    kv: jax.Array


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int

    block_q_major_dkv: int | None = None
    block_k_major_dkv: int | None = None
    block_k_dkv: int | None = None
    block_q_dkv: int | None = None

    block_k_major_dq: int | None = None
    block_k_dq: int | None = None
    block_q_dq: int | None = None

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(f"{prefix}{suffix}={minor} should be smaller than {prefix}_major{suffix}={major}")
            if major % minor != 0:
                raise ValueError(f"{prefix}{suffix}={minor} should divide {prefix}_major{suffix}={major}")

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)
        if self.block_q_major_dkv is not None and self.block_q_dkv is not None:
            verify_major_minor("block_q", "_dkv", self.block_q_major_dkv, self.block_q_dkv)
        if self.block_k_major_dkv is not None and self.block_k_dkv is not None:
            verify_major_minor("block_k", "_dkv", self.block_k_major_dkv, self.block_k_dkv)
        if self.block_k_major_dq is not None and self.block_k_dq is not None:
            verify_major_minor("block_k", "_dq", self.block_k_major_dq, self.block_k_dq)

    @property
    def has_backward_blocks(self) -> bool:
        backward_blocks = (
            self.block_q_major_dkv,
            self.block_k_major_dkv,
            self.block_q_dkv,
            self.block_k_dkv,
            self.block_k_major_dq,
            self.block_k_dq,
            self.block_q_dq,
        )
        return all(b is not None for b in backward_blocks)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        del batch_size, num_heads, q_seq_len, kv_len, d_model
        return BlockSizes(
            block_q=128,
            block_k_major=128,
            block_k=128,
            block_b=1,
            block_q_major_dkv=128,
            block_k_major_dkv=128,
            block_k_dkv=128,
            block_q_dkv=128,
            block_k_major_dq=128,
            block_k_dq=128,
            block_q_dq=128,
        )


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(f"{block_name}={block} should be smaller or equal to {dim_name}={dim}")
    if should_divide and dim % block != 0:
        raise ValueError(f"{dim_name}={dim} should be divisible by {block_name}={block}")


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    softmax_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(mha_reference, q, k, v, ab, segment_ids, causal=causal, softmax_scale=softmax_scale)
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )


def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale: float = 1.0,
    save_residuals: bool = False,
):
    logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    if ab is not None:
        logits += ab
    if softmax_scale != 1.0:
        logits *= softmax_scale

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
    if save_residuals:
        return out, l, m
    return out


@functools.partial(jax.jit, static_argnames=["causal", "mask_value", "softmax_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale=1.0,
):
    return _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=False,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8))
def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
):
    return mha_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=save_residuals,
    )


def _mha_reference_fwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
):
    if save_residuals:
        raise NotImplementedError
    res = _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
        save_residuals=True,
    )
    assert isinstance(res, tuple)
    out, l, m = res
    return out, (q, k, v, ab, segment_ids, out, l, m)


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "mask_value",
        "softmax_scale",
    ],
)
def mha_reference_bwd(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    o,
    l,
    m,
    do,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    softmax_scale: float = 1.0,
):
    if softmax_scale != 1.0:
        raise NotImplementedError

    logits = jnp.einsum(
        "bhqc,bhkc->bhqk",
        q.astype(jnp.float32),
        k.astype(jnp.float32),
    )
    if ab is not None:
        logits += ab

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    unnormalized = jnp.exp(logits - m[..., None])
    p = unnormalized / l[..., None]
    dv = jnp.einsum("bhpt,bhpd->bhtd", p, do.astype(jnp.float32)).astype(v.dtype)

    dp = jnp.einsum("bhpd,bhtd->bhpt", do.astype(jnp.float32), v.astype(jnp.float32))

    di = jnp.sum(o.astype(jnp.float32) * do.astype(jnp.float32), axis=-1)[..., None]

    ds = (dp - di) * p
    dk = jnp.einsum("bhsd,bhst->bhtd", q.astype(jnp.float32), ds).astype(k.dtype)
    dq = jnp.einsum("bhst,bhtd->bhsd", ds, k.astype(jnp.float32)).astype(q.dtype)

    dab = ds if ab is not None else None
    return dq, dk, dv, dab


def _mha_reference_bwd(
    causal: bool,
    mask_value: float,
    softmax_scale: float,
    save_residuals: bool,
    residuals,
    do,
):
    del save_residuals
    q, k, v, ab, segment_ids, o, l, m = residuals
    dq, dk, dv, dab = mha_reference_bwd(
        q,
        k,
        v,
        ab,
        segment_ids,
        o,
        l,
        m,
        do,
        causal=causal,
        mask_value=mask_value,
        softmax_scale=softmax_scale,
    )
    return dq, dk, dv, dab, None


_mha_reference.defvjp(fwd=_mha_reference_fwd, bwd=_mha_reference_bwd)


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)
