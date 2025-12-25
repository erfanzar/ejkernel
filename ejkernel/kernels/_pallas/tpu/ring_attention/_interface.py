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

"""Ring Attention interface using Splash Attention kernels.

This module provides the public API for ring attention on TPU using the
splash attention implementation with ring communication topology.
"""

from __future__ import annotations

import typing

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
from beartype import beartype
from einops import rearrange
from jax import Array
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as mask_lib
from jax.experimental.pallas.ops.tpu.splash_attention import (
    splash_attention_mask_info as mask_info_lib,
)
from jax.scipy.special import logsumexp
from jaxtyping import Float, Int

from ejkernel.ops import BwdParams, FwdParams

from ...._registry import Backend, Platform, kernel_registry
from ._ring_splash import (
    DEFAULT_MASK_VALUE,
    RING_AXIS,
    BlockSizes,
    SegmentIds,
    ring_splash_attention,
)

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


class _AttentionSinkMask(mask_lib._ComputableMask):
    """Allows attending to the first `attention_sink_size` KV positions."""

    attention_sink_size: int

    def __init__(self, *, shape: tuple[int, int], attention_sink_size: int, shard_count: int = 1):
        self.attention_sink_size = int(attention_sink_size)

        def sink_mask_function(q_ids: np.ndarray, kv_ids: np.ndarray) -> np.ndarray:
            return np.broadcast_to(kv_ids < self.attention_sink_size, (q_ids.shape[0], kv_ids.shape[1]))

        super().__init__(shape=shape, mask_function=sink_mask_function, shard_count=shard_count)

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.attention_sink_size == other.attention_sink_size
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.attention_sink_size,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


def _build_mask(
    q_seq_len: int,
    kv_seq_len: int,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
    attention_sink_size: int = 0,
    chunk_size: int | None = None,
) -> mask_lib.Mask:
    """Build a mask from attention parameters.

    Args:
        q_seq_len: Query sequence length.
        kv_seq_len: Key/value sequence length.
        causal: Whether to use causal masking.
        sliding_window: Sliding window size (int or (left, right) tuple).
        chunk_size: Chunk size for chunked causal attention.

    Returns:
        Constructed Mask object.
    """
    shape = (q_seq_len, kv_seq_len)

    # Start with either causal or full mask
    if chunk_size is not None:
        # Chunked causal attention (Llama4 style)
        mask = mask_lib.ChunkedCausalMask(shape=shape, chunk_size=chunk_size)
    elif causal:
        mask = mask_lib.CausalMask(shape=shape)
    else:
        mask = mask_lib.FullMask(shape)

    # Apply sliding window if specified
    if sliding_window is not None:
        if isinstance(sliding_window, int):
            window = (sliding_window, sliding_window)
        else:
            window = sliding_window
        local_mask = mask_lib.LocalMask(shape=shape, window_size=window, offset=0)
        if attention_sink_size > 0:
            sink_size = min(int(attention_sink_size), kv_seq_len)
            local_mask = local_mask | _AttentionSinkMask(shape=shape, attention_sink_size=sink_size)
        # Combine with AND operation
        mask = mask & local_mask

    return mask


def _make_block_sizes(
    fwd_params: FwdParams | None,
    bwd_params: BwdParams | None,
) -> BlockSizes:
    """Create BlockSizes from FwdParams and BwdParams.

    Args:
        fwd_params: Forward pass parameters.
        bwd_params: Backward pass parameters.

    Returns:
        BlockSizes configuration.
    """
    if fwd_params is None:
        return BlockSizes.get_default()

    block_q = fwd_params.q_blocksize
    block_kv = fwd_params.kv_blocksize

    # Set backward block sizes if provided
    if bwd_params is not None:
        block_q_dkv = bwd_params.q_blocksize
        block_kv_dkv = bwd_params.kv_blocksize
        block_q_dq = bwd_params.q_blocksize
        block_kv_dq = bwd_params.kv_blocksize
    else:
        block_q_dkv = block_q
        block_kv_dkv = block_kv
        block_q_dq = block_q
        block_kv_dq = block_kv

    return BlockSizes(
        block_q=block_q,
        block_kv=block_kv,
        block_kv_compute=block_kv,
        block_q_dkv=block_q_dkv,
        block_kv_dkv=block_kv_dkv,
        block_kv_dkv_compute=block_kv_dkv,
        block_q_dq=block_q_dq,
        block_kv_dq=block_kv_dq,
    )


@kernel_registry.register("ring_attention", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ring_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    q_position_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_position_ids: Int[Array, "batch seq_len_k"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    mask_builder: Callable[[int, int, int, int, int], Mask] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = False,
    logits_soft_cap: float | None = None,
    softmax_scale: float | None = None,
    axis_name: str | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    fused_backward: bool = False,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    """Computes ring attention using Splash Attention kernels on TPU.

    This implementation uses JAX's splash attention with ring communication
    topology for distributed attention computation across devices.

    Args:
        query: Query tensor [batch, q_len, num_heads, head_dim].
        key: Key tensor [batch, kv_len, num_kv_heads, head_dim].
        value: Value tensor [batch, kv_len, num_kv_heads, head_dim].
        q_segment_ids: Optional query segment IDs [batch, q_len].
        kv_segment_ids: Optional KV segment IDs [batch, kv_len].
        softmax_aux: Optional attention sink logits (maps to sinks parameter).
        bias: Optional attention bias (not supported in splash attention).
        mask_builder: Optional custom mask builder function.
        sliding_window: Sliding window size for local attention.
        chunk_size: Chunk size for chunked causal attention.
        causal: Whether to use causal masking.
        logits_soft_cap: Soft cap for attention logits.
        softmax_scale: Scaling factor for attention scores.
        axis_name: Name of the ring communication axis.
        fwd_params: Forward pass block size parameters.
        bwd_params: Backward pass block size parameters.
        fused_backward: Whether to use fused backward kernel.

    Returns:
        Attention output [batch, q_len, num_heads, head_dim].

    Raises:
        NotImplementedError: If bias is provided (not supported).
    """
    if bias is not None:
        raise NotImplementedError(
            "Attention bias is not supported in splash ring attention. "
            "Please remove the bias parameter or use a different kernel."
        )
    del mask_builder, fused_backward

    # Get dimensions
    _, q_len, num_heads, head_dim = query.shape
    _, kv_len, num_kv_heads, _ = key.shape

    # Determine if this is MQA (multi-query attention)
    is_mqa = num_kv_heads == 1 and num_heads > 1

    # Apply softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    query = query * softmax_scale

    # Create block sizes configuration
    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=min(512, q_len), kv_blocksize=min(512, kv_len))

    block_sizes = _make_block_sizes(fwd_params, bwd_params)

    # Build mask from parameters
    mask = _build_mask(
        q_seq_len=q_len,
        kv_seq_len=kv_len,
        causal=causal,
        sliding_window=sliding_window,
        attention_sink_size=0,
        chunk_size=chunk_size,
    )

    # Set ring axis
    ring_axis = axis_name if axis_name is not None else RING_AXIS

    sinks = None
    if softmax_aux is not None:
        aux = jnp.asarray(softmax_aux, dtype=jnp.float32)
        if aux.ndim == 1:
            sinks = jnp.broadcast_to(logsumexp(aux), (num_heads,))
        else:
            raise ValueError(f"softmax_aux must be 1D, got shape {aux.shape}.")

    # Create segment IDs if provided
    segment_ids = None
    if q_segment_ids is not None or kv_segment_ids is not None:
        q_seg = q_segment_ids if q_segment_ids is not None else kv_segment_ids
        kv_seg = kv_segment_ids if kv_segment_ids is not None else q_segment_ids
        segment_ids = SegmentIds(q=q_seg[0], kv=kv_seg[0])

    def single_batch_attention(q, k, v, seg_ids, sinks_batch):
        """Process single batch element."""
        # Rearrange from [seq_len, num_heads, head_dim] to [num_heads, seq_len, head_dim]
        q = rearrange(q, "s h d -> h s d")
        k = rearrange(k, "s h d -> h s d")
        v = rearrange(v, "s h d -> h s d")

        if is_mqa:
            k = k.squeeze(0)  # [seq_len, head_dim]
            v = v.squeeze(0)  # [seq_len, head_dim]

        multi_head_mask = mask_lib.MultiHeadMask(masks=tuple([mask] * num_heads))

        fwd_mask_info, mask_function = mask_info_lib._process_mask(
            multi_head_mask,
            (block_sizes.block_q, block_sizes.block_kv),
            is_dkv=False,
        )
        fwd_mask_info = jax.tree_util.tree_map(jnp.array, fwd_mask_info)

        dkv_mask_info = None
        if block_sizes.has_backward_blocks:
            dkv_mask_info, _ = mask_info_lib._process_mask(
                multi_head_mask,
                (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
                is_dkv=True,
            )
            dkv_mask_info = jax.tree_util.tree_map(jnp.array, dkv_mask_info)

        out = ring_splash_attention(
            fwd_mask_info=fwd_mask_info,
            dkv_mask_info=dkv_mask_info,
            q=q,
            k=k,
            v=v,
            segment_ids=seg_ids,
            sinks=sinks_batch,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            mask_value=DEFAULT_MASK_VALUE,
            mask_function=mask_function,
            logits_soft_cap=logits_soft_cap,
            ring_axis=ring_axis,
            causal=causal,
        )

        out = rearrange(out, "h s d -> s h d")
        return out

    outputs = jax.vmap(lambda q, k, v: single_batch_attention(q, k, v, segment_ids, sinks))(query, key, value)

    return outputs
