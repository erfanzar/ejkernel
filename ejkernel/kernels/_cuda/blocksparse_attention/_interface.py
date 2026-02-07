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

"""Public interface for the CUDA block-sparse attention kernel.

This module exposes :func:`blocksparse_attention`, which is registered in the
ejkernel kernel registry under the ``CUDA`` platform and ``GPU`` backend.  It
also implements the JAX custom-VJP rule so that:

* The **forward pass** is dispatched to the compiled CUDA kernel via
  :func:`._cuda_impl.blocksparse_attention_cuda`.
* The **backward pass** is not implemented by the CUDA path and raises
  ``NotImplementedError`` when gradients are requested.

Helper utilities for converting sparsity layouts to dense boolean masks and
for creating empty placeholder layouts are also defined here.
"""

from __future__ import annotations

import functools
import typing

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from beartype.typing import Callable
from jaxtyping import Array, ArrayLike, Bool, Float, Int

from ejkernel.callib import ejit
from ejkernel.kernels._registry import Backend, Platform, kernel_registry
from ejkernel.kernels._triton.blocksparse_attention._mask import SparseMask, create_sparsity_mask
from ejkernel.ops import BwdParams, FwdParams

from ._cuda_impl import blocksparse_attention_cuda

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import Mask


@functools.partial(jax.custom_vjp, nondiff_argnums=[8, 11, 12, 13, 14, 15, 16, 17, 18])
@functools.partial(
    jax.jit,
    static_argnames=[
        "softmax_scale",
        "apply_load_balance",
        "sequence_parallelism_mesh_axis_name",
        "window_left",
        "window_right",
        "causal",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
    ],
)
def _blocksparse_attention_bhtd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None = None,
    softmax_scale: float = 1.0,
    softmax_aux: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    apply_load_balance: bool = True,
    sequence_parallelism_mesh_axis_name: str | None = None,
    window_left: int = -1,
    window_right: int = -1,
    causal: bool = True,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    logits_soft_cap: float | None = None,
) -> ArrayLike:
    """Compute block-sparse attention in BHTD layout with a custom VJP rule.

    This is the core differentiable primitive. The forward evaluation
    delegates to the CUDA kernel, while JAX's custom VJP machinery routes
    gradient computation through :func:`_blocksparse_attention_bhtd_fwd`
    and :func:`_blocksparse_attention_bhtd_bwd`.

    Args:
        query: Query tensor, shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor, shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor, shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Query position indices, shape ``(batch, q_len)``.
        q_segment_ids: Query segment identifiers, shape ``(batch, q_len)``.
        kv_positions: Key/value position indices, shape ``(batch, kv_len)``.
        kv_segment_ids: Key/value segment identifiers, shape
            ``(batch, kv_len)``.
        qkv_layouts: Block-level sparsity layouts.
        softmax_scale: Logit scaling factor.
        softmax_aux: Optional attention-sink auxiliary weights.
        bias: Explicit additive bias (not supported; raises if provided).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size (``-1`` to disable).
        window_right: Right sliding-window size (``-1`` to disable).
        causal: Whether to apply causal masking.
        fwd_params: Forward-pass kernel configuration.
        bwd_params: Backward-pass kernel configuration (unused in the
            forward path).
        logits_soft_cap: Optional soft-cap for attention logits.

    Returns:
        Attention output of shape
        ``(batch, num_heads, q_len, v_head_dim)``.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
        AssertionError: If *fwd_params* is ``None``.
    """
    del apply_load_balance, sequence_parallelism_mesh_axis_name, bwd_params

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")

    assert fwd_params is not None, "fwd_params must be provided"

    return blocksparse_attention_cuda(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )


def _blocksparse_attention_bhtd_fwd(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    q_positions: ArrayLike,
    q_segment_ids: ArrayLike,
    kv_positions: ArrayLike,
    kv_segment_ids: ArrayLike,
    qkv_layouts: tuple[SparseMask] | None,
    softmax_scale: float,
    softmax_aux: ArrayLike | None,
    bias: ArrayLike | None,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    bwd_params: BwdParams,
    logits_soft_cap: float | None,
):
    """Forward pass for the custom VJP of block-sparse attention.

    Runs the CUDA forward kernel and returns both the output and a residuals
    tuple that the backward pass (:func:`_blocksparse_attention_bhtd_bwd`)
    will consume.

    Args:
        query: Query tensor, shape ``(batch, num_heads, q_len, head_dim)``.
        key: Key tensor, shape ``(batch, num_kv_heads, kv_len, head_dim)``.
        value: Value tensor, shape
            ``(batch, num_kv_heads, kv_len, v_head_dim)``.
        q_positions: Query position indices.
        q_segment_ids: Query segment identifiers.
        kv_positions: Key/value position indices.
        kv_segment_ids: Key/value segment identifiers.
        qkv_layouts: Block-level sparsity layouts.
        softmax_scale: Logit scaling factor.
        softmax_aux: Optional attention-sink auxiliary weights.
        bias: Explicit additive bias (not supported).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size.
        window_right: Right sliding-window size.
        causal: Whether to apply causal masking.
        fwd_params: Forward-pass kernel configuration.
        bwd_params: Backward-pass kernel configuration (unused here but
            accepted for signature consistency).
        logits_soft_cap: Optional soft-cap for attention logits.

    Returns:
        A tuple ``(out, res)`` where *out* is the attention output tensor
        and *res* is a tuple of values saved for the backward pass.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
    """
    del apply_load_balance, sequence_parallelism_mesh_axis_name, bwd_params

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")

    out = blocksparse_attention_cuda(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        logits_soft_cap=logits_soft_cap,
    )

    res = (
        query,
        key,
        value,
        q_positions,
        q_segment_ids,
        kv_positions,
        kv_segment_ids,
        qkv_layouts,
        softmax_scale,
        softmax_aux,
        window_left,
        window_right,
        causal,
        logits_soft_cap,
        fwd_params,
    )
    return out, res


def _blocksparse_attention_bhtd_bwd(
    softmax_scale: float,
    apply_load_balance: bool,
    sequence_parallelism_mesh_axis_name: str | None,
    window_left: int,
    window_right: int,
    causal: bool,
    fwd_params: FwdParams,
    bwd_params: BwdParams,
    logits_soft_cap: float | None,
    res,
    dout: ArrayLike,
):
    """Backward pass for the custom VJP of block-sparse attention.

    Args:
        softmax_scale: Logit scaling factor (non-differentiable).
        apply_load_balance: Unused; reserved for API compatibility.
        sequence_parallelism_mesh_axis_name: Unused; reserved for API
            compatibility.
        window_left: Left sliding-window size (non-differentiable).
        window_right: Right sliding-window size (non-differentiable).
        causal: Whether causal masking is applied (non-differentiable).
        fwd_params: Forward-pass kernel configuration (non-differentiable).
        bwd_params: Backward-pass kernel configuration (unused).
        logits_soft_cap: Optional soft-cap for attention logits
            (non-differentiable).
        res: Residuals tuple saved by
            :func:`_blocksparse_attention_bhtd_fwd`.
        dout: Upstream gradient tensor of the same shape as the forward
            output.
    """
    del (
        softmax_scale,
        apply_load_balance,
        sequence_parallelism_mesh_axis_name,
        window_left,
        window_right,
        causal,
        fwd_params,
        bwd_params,
        logits_soft_cap,
        res,
        dout,
    )
    raise NotImplementedError(
        "CUDA blocksparse_attention does not implement backward. "
        "Fallback gradients are disabled."
    )


_blocksparse_attention_bhtd.defvjp(_blocksparse_attention_bhtd_fwd, _blocksparse_attention_bhtd_bwd)


@kernel_registry.register("blocksparse_attention", Platform.CUDA, Backend.GPU)
@ejit(
    static_argnames=(
        "softmax_scale",
        "mask_builder",
        "sliding_window",
        "chunk_size",
        "causal",
        "fused_backward",
        "fwd_params",
        "bwd_params",
        "logits_soft_cap",
    )
)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch num_heads seq_len head_dim"],
    key: Float[Array, "batch kv_num_heads kv_len head_dim"],
    value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
    q_segment_ids: Int[Array, "batch seq_len"] | None = None,
    kv_segment_ids: Int[Array, "batch kv_len"] | None = None,
    q_positions: Int[Array, "batch seq_len"] | None = None,
    kv_positions: Int[Array, "batch kv_len"] | None = None,
    softmax_aux: Float[Array, "num_sinks"] | None = None,
    bias: Float[Array, "batch num_heads seq_len kv_len"] | None = None,
    attention_mask: Bool[Array, "batch num_heads_or_1 seq_len kv_len"]
    | Int[Array, "batch num_heads_or_1 seq_len kv_len"]
    | None = None,
    sequence_parallelism_mesh_axis_name: str | None = None,
    logits_soft_cap: float | None = None,
    qkv_layouts: tuple["SparseMask"] | None = None,
    softmax_scale: float | None = None,
    fwd_params: FwdParams | None = None,
    bwd_params: BwdParams | None = None,
    mask_builder: Callable[[int, int, int, int, int], "Mask"] | Callable[[], "SparseMask"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    chunk_size: int | None = None,
    causal: bool = True,
    fused_backward: bool = False,
) -> Float[Array, "batch num_heads seq_len vhead_dim"]:
    """Compute block-sparse multi-head attention on a CUDA GPU.

    This is the high-level entry point registered in the ejkernel kernel
    registry for the ``CUDA`` platform / ``GPU`` backend.  It validates and
    defaults all parameters, constructs sparsity layouts if they are not
    supplied, and delegates to the CUDA forward kernel.

    Grouped-query attention (GQA) is supported: *key* and *value* may have
    fewer heads than *query*, and the kernel will broadcast internally.

    Args:
        query: Query tensor of shape
            ``(batch, num_heads, seq_len, head_dim)``.
        key: Key tensor of shape
            ``(batch, kv_num_heads, kv_len, head_dim)``.
        value: Value tensor of shape
            ``(batch, kv_num_heads, kv_len, vhead_dim)``.
        q_segment_ids: Segment IDs for the query sequence, shape
            ``(batch, seq_len)``.  Tokens with different segment IDs do not
            attend to each other.  Defaults to all-ones.
        kv_segment_ids: Segment IDs for the key/value sequence, shape
            ``(batch, kv_len)``.  Defaults to all-ones.
        q_positions: Integer position indices for queries, shape
            ``(batch, seq_len)``.  Defaults to ``arange(seq_len)``
            broadcast across the batch.
        kv_positions: Integer position indices for keys/values, shape
            ``(batch, kv_len)``.  Defaults to ``arange(kv_len)``.
        softmax_aux: Optional attention-sink auxiliary weights for the
            softmax computation.
        bias: Explicit additive attention bias.  **Not supported** by the
            CUDA kernel; passing a non-``None`` value raises
            :class:`NotImplementedError`.
        attention_mask: Optional dense boolean or integer attention mask of
            shape ``(batch, num_heads_or_1, seq_len, kv_len)``.  When
            segment IDs are not provided, the mask is converted to segment
            IDs internally.
        sequence_parallelism_mesh_axis_name: Mesh axis name for sequence
            parallelism (reserved; currently unused).
        logits_soft_cap: Optional soft-cap applied to attention logits
            before the softmax.  ``None`` disables capping.
        qkv_layouts: Pre-computed block-level sparsity layouts as a tuple
            of :class:`SparseMask` instances.  When ``None``, layouts are
            built automatically via *mask_builder* or
            :func:`create_sparsity_mask`.
        softmax_scale: Scaling factor for dot-product logits.  Defaults to
            ``head_dim ** -0.5``.
        fwd_params: Forward-pass kernel parameters (:class:`FwdParams`).
            Must specify ``q_blocksize`` and ``kv_blocksize``.  Defaults
            to ``FwdParams(q_blocksize=64, kv_blocksize=64, num_stages=2,
            num_warps=4)``.
        bwd_params: Backward-pass kernel parameters (:class:`BwdParams`).
            Defaults to ``BwdParams(q_blocksize=32, kv_blocksize=32,
            num_stages=2, num_warps=4)``.
        mask_builder: A callable that returns a :class:`SparseMask` (or
            compatible ``Mask`` object) when *qkv_layouts* is ``None``.
        sliding_window: Sliding-window size.  Can be a single ``int``
            (symmetric left/right window) or a ``(left, right)`` tuple.
            ``None`` disables windowing.
        chunk_size: Unused; reserved for API compatibility.
        causal: Whether to apply lower-triangular causal masking.
            Defaults to ``True``.
        fused_backward: Unused; reserved for API compatibility.

    Returns:
        The attention output tensor of shape
        ``(batch, num_heads, seq_len, vhead_dim)``.

    Raises:
        NotImplementedError: If *bias* is not ``None``.
        ValueError: If *fwd_params* does not specify ``q_blocksize`` and
            ``kv_blocksize``.
    """
    del fused_backward, chunk_size

    if bias is not None:
        raise NotImplementedError("Bias is not supported in CUDA block-sparse attention.")

    qlen = query.shape[2]
    kvlen = key.shape[2]

    if mask_builder is not None and qkv_layouts is None:
        qkv_layouts = mask_builder()
    if isinstance(qkv_layouts, SparseMask):
        qkv_layouts = (qkv_layouts,)

    if fwd_params is None:
        fwd_params = FwdParams(q_blocksize=64, kv_blocksize=64, num_stages=2, num_warps=4)
    if bwd_params is None:
        bwd_params = BwdParams(q_blocksize=32, kv_blocksize=32, num_stages=2, num_warps=4)
    if fwd_params.q_blocksize is None or fwd_params.kv_blocksize is None:
        raise ValueError("CUDA blocksparse_attention requires q_blocksize and kv_blocksize in fwd_params.")

    if sliding_window is None:
        window_left = window_right = -1
    elif isinstance(sliding_window, int):
        window_left = window_right = sliding_window
    else:
        window_left, window_right = sliding_window

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5
    if q_positions is None:
        q_positions = jnp.arange(0, qlen).reshape(1, -1).repeat(query.shape[0], 0)
    if kv_positions is None:
        kv_positions = jnp.arange(0, kvlen).reshape(1, -1).repeat(key.shape[0], 0)

    if attention_mask is not None and (q_segment_ids is None or kv_segment_ids is None):
        from ejkernel.types.mask import mask_to_segment_ids

        inferred_q_seg, inferred_kv_seg = mask_to_segment_ids(attention_mask)
        if q_segment_ids is None:
            q_segment_ids = inferred_q_seg
        if kv_segment_ids is None:
            kv_segment_ids = inferred_kv_seg

    if q_segment_ids is None:
        q_segment_ids = jnp.ones_like(q_positions)
    if kv_segment_ids is None:
        kv_segment_ids = jnp.ones_like(kv_positions)

    if qkv_layouts is None:
        qkv_layouts = create_sparsity_mask(
            q_blocksize=fwd_params.q_blocksize,
            kv_blocksize=fwd_params.kv_blocksize,
            kv_positions=kv_positions,
            kv_segment_ids=kv_segment_ids,
            q_positions=q_positions,
            q_segment_ids=q_segment_ids,
            causal=causal,
            window_left=window_left,
            window_right=window_right,
        )

    return _blocksparse_attention_bhtd(
        query=query,
        key=key,
        value=value,
        q_positions=q_positions,
        q_segment_ids=q_segment_ids,
        kv_positions=kv_positions,
        kv_segment_ids=kv_segment_ids,
        qkv_layouts=qkv_layouts,
        softmax_scale=softmax_scale,
        softmax_aux=softmax_aux,
        bias=bias,
        apply_load_balance=True,
        sequence_parallelism_mesh_axis_name=sequence_parallelism_mesh_axis_name,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        fwd_params=fwd_params,
        bwd_params=bwd_params,
        logits_soft_cap=logits_soft_cap,
    )
