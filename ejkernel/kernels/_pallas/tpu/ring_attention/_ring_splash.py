# Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2025 DeepMind Technologies Limited (modified from original tokamax implementation).
# (we dont use their splash impl as is, but modified our splash for ring attention)
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

"""Ring Attention implementation using Splash Attention kernels.

This module provides ring attention by wrapping JAX's splash attention kernels
with a ring communication topology for distributed attention computation.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util

from ..blocksparse_attention import _info as mask_info_lib
from ..blocksparse_attention import _kernel as splash_kernel
from ..blocksparse_attention import _masks as mask_lib

partial = functools.partial

# Default axis name for ring communication
RING_AXIS = "sp"

# Type aliases from splash attention
MaskInfo = mask_info_lib.MaskInfo
BlockSizes = splash_kernel.BlockSizes
MaskFunctionType = splash_kernel.MaskFunctionType
DEFAULT_MASK_VALUE = splash_kernel.DEFAULT_MASK_VALUE


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences."""

    q: jax.Array  # [q_seq_len]
    kv: jax.Array  # [kv_seq_len]


def _update_out_and_lse(
    out: jax.Array,
    lse: jax.Array,
    block_out: jax.Array,
    block_lse: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    is_first = lse == -jnp.inf
    block_lse_expanded = block_lse[..., None]
    lse_expanded = lse[..., None]
    sigmoid_weight = jax.nn.sigmoid(block_lse_expanded - lse_expanded)
    new_out = out - sigmoid_weight * (out - block_out)
    new_lse = lse + jax.nn.softplus(block_lse - lse)
    new_out = jnp.where(is_first[..., None], block_out, new_out)
    new_lse = jnp.where(is_first, block_lse, new_lse)
    return new_out, new_lse


def _ring_attention_forward(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    ring_axis_size = lax.psum(1, ring_axis)
    device_idx = lax.axis_index(ring_axis)

    num_heads = q.shape[0]
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa else k.shape[1]

    shift = partial(
        lax.ppermute,
        axis_name=ring_axis,
        perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
    )

    o_shape = q.shape
    o_init = jnp.zeros(o_shape, dtype=jnp.float32)
    lse_init = jnp.full((num_heads, q_seq_len), -jnp.inf, dtype=jnp.float32)

    splash_segment_ids = None
    if segment_ids is not None:
        splash_segment_ids = splash_kernel.SegmentIds(q=segment_ids.q, kv=segment_ids.kv)

    if causal:
        base_q_sequence = jnp.arange(q_seq_len, dtype=jnp.int32)

        def causal_mask_fn(q_ids, kv_ids):
            return q_ids >= kv_ids

    def body(carry, iteration):
        o_prev, lse_prev, k_current, v_current, kv_source_device = carry
        k_next = shift(k_current)
        v_next = shift(v_current)
        is_first_iteration = iteration == 0

        if causal:
            offset = device_idx * q_seq_len - kv_source_device * kv_seq_len
            modified_q_sequence = base_q_sequence + offset
            fwd_mask_info_iter = MaskInfo(
                data_next=None,
                mask_next=None,
                block_mask=None,
                partial_mask_blocks=None,
                q_sequence=modified_q_sequence,
            )
            mask_function_iter = causal_mask_fn
        else:
            fwd_mask_info_iter = fwd_mask_info
            mask_function_iter = mask_function

        sinks_iter = None
        if sinks is not None:
            sinks_iter = jnp.where(is_first_iteration, sinks, jnp.full_like(sinks, -1e9))

        out_curr, residuals = splash_kernel._splash_attention_forward(
            fwd_mask_info=fwd_mask_info_iter,
            q=q,
            k=k_current,
            v=v_current,
            segment_ids=splash_segment_ids,
            sinks=sinks_iter,
            mask_value=mask_value,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            residual_checkpoint_name=None,
            mask_function=mask_function_iter,
            save_residuals=True,
            logits_soft_cap=logits_soft_cap,
        )

        lse_curr = residuals[0].astype(jnp.float32)
        out_curr = out_curr.astype(jnp.float32)
        o_next, lse_next = _update_out_and_lse(o_prev, lse_prev, out_curr, lse_curr)
        kv_source_next = (kv_source_device - 1) % ring_axis_size

        return (o_next, lse_next, k_next, v_next, kv_source_next), None

    initial_kv_source = device_idx.astype(jnp.int32)
    initial_carry = (o_init, lse_init, k, v, initial_kv_source)
    (o_final, lse_final, _, _, _), _ = lax.scan(
        body,
        initial_carry,
        xs=jnp.arange(0, ring_axis_size),
        length=ring_axis_size,
        unroll=True,
    )

    out = o_final.astype(q.dtype)
    return out, (lse_final, lse_final)


def _ring_attention_backward(
    res: tuple,
    do: jax.Array,
    *,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    (q, k, v, segment_ids, sinks, out, logsumexp, _fwd_mask_info, dq_mask_info, dkv_mask_info) = res
    do_main = do.astype(jnp.float32)

    ring_axis_size = lax.psum(1, ring_axis)

    shift = partial(
        lax.ppermute,
        axis_name=ring_axis,
        perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
    )

    dq_accum = jnp.zeros_like(q, dtype=jnp.float32)
    dk_accum = jnp.zeros_like(k, dtype=jnp.float32)
    dv_accum = jnp.zeros_like(v, dtype=jnp.float32)
    dsinks_accum = None
    if sinks is not None:
        dsinks_accum = jnp.zeros_like(sinks, dtype=jnp.float32)

    splash_segment_ids = None
    if segment_ids is not None:
        splash_segment_ids = splash_kernel.SegmentIds(q=segment_ids.q, kv=segment_ids.kv)

    def body(carry, _: int):
        dq_accum, dk_accum, dv_accum, k_cur, v_cur, dsinks = carry
        k_next = shift(k_cur)
        v_next = shift(v_cur)

        residuals_for_chunk = (
            q,
            k_cur,
            v_cur,
            splash_segment_ids,
            sinks,
            out,
            logsumexp,
            dq_mask_info,
            dkv_mask_info,
        )

        grads = splash_kernel._splash_attention_bwd(
            save_residuals=False,
            mask_value=mask_value,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            residual_checkpoint_name=None,
            mask_function=mask_function,
            logits_soft_cap=logits_soft_cap,
            interpret=False,
            res=residuals_for_chunk,
            do=do_main,
        )

        dq_i = grads[3].astype(jnp.float32)
        dk_i = grads[4].astype(jnp.float32)
        dv_i = grads[5].astype(jnp.float32)
        dsinks_i = grads[7]

        dv_accum = dv_accum + dv_i
        dv_next = shift(dv_accum)
        dk_accum = dk_accum + dk_i
        dk_next = shift(dk_accum)
        dq_accum = dq_accum + dq_i

        if dsinks is not None and dsinks_i is not None:
            dsinks = dsinks + dsinks_i.astype(jnp.float32)

        return (dq_accum, dk_next, dv_next, k_next, v_next, dsinks), None

    initial_carry = (dq_accum, dk_accum, dv_accum, k, v, dsinks_accum)
    (dq_final, dk_final, dv_final, _, _, dsinks_final), _ = lax.scan(
        body,
        initial_carry,
        xs=jnp.arange(0, ring_axis_size),
        length=ring_axis_size,
        unroll=True,
    )

    if sinks is not None and dsinks_final is not None:
        dsinks_final = jax.lax.psum(dsinks_final, axis_name=ring_axis)

    dq_final = dq_final.astype(q.dtype)
    dk_final = dk_final.astype(k.dtype)
    dv_final = dv_final.astype(v.dtype)

    return dq_final, dk_final, dv_final, dsinks_final


def _ring_attention_fwd_rule(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    *,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
) -> tuple[jax.Array, tuple]:
    out, (logsumexp, _) = _ring_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
    )
    residuals_for_bwd = (
        q,
        k,
        v,
        segment_ids,
        sinks,
        out,
        logsumexp,
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
    )
    return out, residuals_for_bwd


def _ring_attention_bwd_rule(
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
    res: tuple,
    do: jax.Array,
):
    dq, dk, dv, dsinks = _ring_attention_backward(
        res,
        do,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
    )
    return (None, None, None, dq, dk, dv, None, dsinks)


@partial(
    jax.custom_vjp,
    nondiff_argnums=(8, 9, 10, 11, 12, 13, 14),
)
def _ring_attention_custom(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
) -> jax.Array:
    out, _ = _ring_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
    )
    return out


def _ring_attention_custom_fwd(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
):
    return _ring_attention_fwd_rule(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
    )


def _ring_attention_custom_bwd(
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
    causal: bool,
    res: tuple,
    do: jax.Array,
):
    return _ring_attention_bwd_rule(
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        res=res,
        do=do,
    )


_ring_attention_custom.defvjp(_ring_attention_custom_fwd, _ring_attention_custom_bwd)


def _has_axis(axis_name: str) -> bool:
    try:
        lax.psum(1, axis_name)
        return True
    except (NameError, ValueError):
        return False


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "mask_value",
        "mask_function",
        "logits_soft_cap",
        "ring_axis",
        "causal",
    ],
)
def ring_splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_value: float = DEFAULT_MASK_VALUE,
    mask_function: MaskFunctionType | None = None,
    logits_soft_cap: float | None = None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
) -> jax.Array:
    if not _has_axis(ring_axis):
        raise ValueError(f"Ring axis '{ring_axis}' does not exist in the current context.")

    dq_mask_info = fwd_mask_info if block_sizes.has_backward_blocks else None

    return _ring_attention_custom(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value,
        is_mqa,
        block_sizes,
        mask_function,
        logits_soft_cap,
        ring_axis,
        causal,
    )


@jax.tree_util.register_pytree_node_class
class RingSplashAttentionKernel:
    def __init__(
        self,
        fwd_mask_info: MaskInfo,
        dkv_mask_info: MaskInfo | None,
        ring_axis: str = RING_AXIS,
        **kwargs,
    ):
        self.fwd_mask_info = fwd_mask_info
        self.dkv_mask_info = dkv_mask_info
        self.ring_axis = ring_axis
        self.kwargs = kwargs

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        segment_ids: SegmentIds | None = None,
        sinks: jax.Array | None = None,
    ) -> jax.Array:
        return ring_splash_attention(
            self.fwd_mask_info,
            self.dkv_mask_info,
            q,
            k,
            v,
            segment_ids=segment_ids,
            sinks=sinks,
            ring_axis=self.ring_axis,
            **self.kwargs,
        )

    def tree_flatten(self):
        children = (self.fwd_mask_info, self.dkv_mask_info)
        aux_data = {"ring_axis": self.ring_axis, **self.kwargs}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        fwd_mask_info, dkv_mask_info = children
        if isinstance(fwd_mask_info, tuple):
            fwd_mask_info = MaskInfo(*fwd_mask_info)
        if dkv_mask_info is not None and isinstance(dkv_mask_info, tuple):
            dkv_mask_info = MaskInfo(*dkv_mask_info)
        return cls(fwd_mask_info, dkv_mask_info, **aux_data)


def make_ring_attention(
    mask: np.ndarray | jax.Array | mask_lib.Mask,
    *,
    block_sizes: BlockSizes | None = None,
    is_mqa: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    logits_soft_cap: float | None = None,
    ring_axis: str = RING_AXIS,
    q_seq_shards: int = 1,
) -> RingSplashAttentionKernel:
    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")

    if isinstance(mask, np.ndarray):
        mask = mask_lib.NumpyMask(mask)

    if block_sizes is None:
        block_sizes = BlockSizes.get_default()

    multi_head_mask = mask_lib.MultiHeadMask(masks=(mask,))

    fwd_mask_info, mask_function = mask_info_lib._process_mask(
        multi_head_mask,
        (block_sizes.block_q, block_sizes.block_kv),
        is_dkv=False,
        q_seq_shards=q_seq_shards,
    )
    fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

    dkv_mask_info = None
    if block_sizes.has_backward_blocks:
        dkv_mask_info, _ = mask_info_lib._process_mask(
            multi_head_mask,
            (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
            is_dkv=True,
        )
        dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

    return RingSplashAttentionKernel(
        fwd_mask_info,
        dkv_mask_info,
        ring_axis=ring_axis,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_value=mask_value,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
    )
