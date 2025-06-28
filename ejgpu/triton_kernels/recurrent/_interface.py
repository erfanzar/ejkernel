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

from ._triton_impl_bwd import bwd_triton_impl
from ._triton_impl_fwd import fwd_triton_impl


def _fwd_call(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
):
    """
    Forward pass for recurrent linear attention in a custom VJP.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        g: Optional gate tensor for GLA-style gating.
        g_gamma: Optional decay factor for Lightning-style attention.
        gk: Optional gate applied directly to K.
        gv: Optional gate applied directly to V.
        scale: Scaling factor for attention.
        initial_state: Initial hidden state for the recurrence.
        reverse: If True, process sequence in reverse.
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.

    Returns:
        A tuple containing the output and final state, and another tuple of
        residuals for the backward pass.
    """
    o, ht = fwd_triton_impl(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    residual = q, k, v, g, gk, gv, o, initial_state
    return (o, ht), residual


def _bwd_call(
    g_gamma: jax.Array | None,
    scale: float | None,
    reverse: bool,
    cu_seqlens: jax.Array | None,
    residual: tuple[jax.Array],
    dout: jax.Array,
):
    """
    Backward pass for recurrent linear attention in a custom VJP.

    Args:
        g_gamma: Non-differentiable decay factor.
        scale: Non-differentiable scaling factor.
        reverse: Non-differentiable reverse flag.
        cu_seqlens: Non-differentiable cumulative sequence lengths.
        residual: Tensors saved from the forward pass.
        dout: A tuple containing the gradients of the output (`do`) and the
            final hidden state (`dht`).

    Returns:
        A tuple of gradients corresponding to the differentiable inputs
        (q, k, v, g, gk, gv, initial_state).
    """
    do, dht = dout
    q, k, v, g, gk, gv, o, initial_state = residual
    dq, dk, dv, dg, dgk, dgv, dh0 = bwd_triton_impl(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        o=o,
        do=do,
        dht=dht,
        scale=scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
    return dq, dk, dv, dg, dgk, dgv, dh0


@partial(jax.custom_vjp, nondiff_argnums=(4, 7, 9, 10))
@partial(jax.jit, static_argnums=(7, 9))
def _recurrent(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Core JIT-compiled recurrent function with a custom VJP.

    This is an internal function that directly calls the Triton implementation
    and is registered with JAX's custom differentiation system.

    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        g: Optional gate tensor for GLA-style gating.
        g_gamma: Optional decay factor for Lightning-style attention.
        gk: Optional gate applied directly to K.
        gv: Optional gate applied directly to V.
        scale: Scaling factor for attention (static argument).
        initial_state: Initial hidden state for the recurrence.
        reverse: If True, process sequence in reverse (static argument).
        cu_seqlens: Cumulative sequence lengths for variable-length sequences.

    Returns:
        A tuple containing:
            - The output tensor `o`.
            - The final hidden state `ht`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return fwd_triton_impl(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )


_recurrent.defvjp(_fwd_call, _bwd_call)


def recurrent(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    gk: jax.Array | None = None,
    gv: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    reverse: bool = False,
    cu_seqlens: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """
    Computes a general recurrent linear attention using a custom Triton kernel.

    This function provides a highly optimized and flexible implementation of
    recurrent linear attention. It processes sequences step-by-step, resulting
    in O(N) complexity, which is ideal for long sequences. The implementation
    is general enough to support various linear attention mechanisms by
    configuring the gate inputs.

    It supports both standard batch processing and variable-length sequence
    processing using cumulative sequence lengths (`cu_seqlens`).

    Args:
        q: The query tensor.
        k: The key tensor.
        v: The value tensor.
        g: Optional gate tensor for Gated Linear Attention (GLA) style gating.
        g_gamma: Optional decay factor, used for mechanisms like Lightning
            Attention where the decay is fixed per-head or per-layer.
        gk: Optional gate tensor applied element-wise to keys.
        gv: Optional gate tensor applied element-wise to values.
        scale: A scaling factor applied to the query. If `None`, it defaults
            to `1 / sqrt(head_dim)`.
        initial_state: The initial hidden state for the recurrence. This is
            useful for chunked processing of very long sequences or for stateful
            autoregressive decoding.
        reverse: If `True`, the sequence is processed in reverse order (from
            last token to first).
        cu_seqlens: Cumulative sequence lengths for variable-length inputs.
            This is a 1D tensor like `[0, len_seq1, len_seq1+len_seq2, ...]`.
            If provided, the input tensors are expected to be "packed" with a
            batch size of 1.

    Returns:
        A tuple containing:
            - o (jax.Array): The output tensor, with the same shape as `q`.
            - final_state (jax.Array): The final hidden state of the recurrence,
              which can be used as `initial_state` for a subsequent segment.
    """
    return _recurrent(
        q=q,
        k=k,
        v=v,
        g=g,
        g_gamma=g_gamma,
        gk=gk,
        gv=gv,
        scale=scale,
        initial_state=initial_state,
        reverse=reverse,
        cu_seqlens=cu_seqlens,
    )
