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


from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_bwd import _recurrent_attention_bwd
from ._xla_impl_fwd import _recurrent_attention_fwd, _recurrent_attention_varlen_fwd


@partial(jax.custom_vjp, nondiff_argnums=(4, 7, 9, 10))
def _recurrent_core(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "... num_heads head_dim head_dim"]]:
    """Core recurrent attention with custom VJP."""
    if cu_seqlens is not None:
        return _recurrent_attention_varlen_fwd(
            query, key, value, cu_seqlens, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )
    else:
        output, _, final_state = _recurrent_attention_fwd(
            query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )
        return output, final_state


def _recurrent_fwd(
    query: Float[Array, "batch seq_len num_heads head_dim"],
    key: Float[Array, "batch seq_len num_heads head_dim"],
    value: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[
    tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim head_dim"]],
    tuple,
]:
    """Forward with residuals for backward."""

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)
    if g is None:
        g = jnp.zeros_like(query)
    if g_gamma is None:
        g_gamma = jnp.zeros((query.shape[2],))
    if gk is None:
        gk = jnp.zeros_like(query)
    if gv is None:
        gv = jnp.zeros_like(query)
    if initial_state is None:
        initial_state = jnp.zeros((query.shape[0], query.shape[2], query.shape[3], query.shape[3]))

    if cu_seqlens is not None:
        raise NotImplementedError("Custom backward for varlen not yet implemented")
    else:
        o, hidden_states, final_state = _recurrent_attention_fwd(
            query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse
        )

    residuals = (query, key, value, g, g_gamma, gk, gv, hidden_states, initial_state)
    return (o, final_state), residuals


def _recurrent_bwd(
    g_gamma_nondiff: Float[Array, "... num_heads"],
    scale_nondiff: float | None,
    reverse: bool,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residuals: tuple,
    grads: tuple,
) -> tuple:
    """Backward pass with custom implementation."""
    query, key, value, g, g_gamma, gk, gv, hidden_states, initial_state = residuals
    do, dfinal_state = grads

    if scale_nondiff is None:
        softmax_scale = 1.0 / jnp.sqrt(query.shape[-1]).astype(jnp.float32)
    else:
        softmax_scale = scale_nondiff

    if cu_seqlens is not None:
        raise NotImplementedError("Variable-length backward not yet implemented")

    dq, dk, dv, dg, dgk, dgv, dinitial_state = _recurrent_attention_bwd(
        query, key, value, g, g_gamma, gk, gv, hidden_states, do, dfinal_state, softmax_scale, initial_state, reverse
    )

    return (dq, dk, dv, dg, dgk, dgv, dinitial_state)


_recurrent_core.defvjp(_recurrent_fwd, _recurrent_bwd)


@kernel_registry.register("recurrent", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def recurrent(
    query: Float[Array, "batch seq_len num_heads qk_head_dim"],
    key: Float[Array, "batch seq_len num_kv_heads qk_head_dim"],
    value: Float[Array, "batch seq_len num_kv_heads v_head_dim"],
    g: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    g_gamma: Float[Array, "... num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads qk_head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads v_head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "... num_heads qk_head_dim v_head_dim"] | None = None,
    reverse: bool = False,
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch seq_len num_heads v_head_dim"], Float[Array, "... num_heads qk_head_dim v_head_dim"]]:
    """
    Recurrent linear attention with O(N) complexity using JAX/XLA.

    This implements linear attention as a recurrent process, maintaining a hidden
    state that accumulates key-value information sequentially. Unlike standard
    O(N²) attention, this achieves O(N) complexity by processing tokens one at a time.

    The core update is:
        h_t = decay_t * h_{t-1} + k_t^T ⊗ v_t
        o_t = h_t @ q_t

    Supports various gating mechanisms for different attention variants.

    Args:
        query: Query tensor [batch, seq_len, num_heads, head_dim]
        key: Key tensor [batch, seq_len, num_heads, head_dim]
        value: Value tensor [batch, seq_len, num_heads, head_dim]
        g: Optional gate tensor for GLA-style gating [batch, seq_len, num_heads, head_dim]
        g_gamma: Optional per-head decay factor [num_heads] for Lightning attention
        gk: Optional gate applied to keys [batch, seq_len, num_heads, head_dim]
        gv: Optional gate applied to values [batch, seq_len, num_heads, head_dim]
        softmax_scale: Query scaling factor. If None, defaults to 1/sqrt(head_dim)
        initial_state: Initial hidden state [batch, num_heads, head_dim, head_dim]
        reverse: If True, process sequence in reverse order
        cu_seqlens: Cumulative sequence lengths for variable-length inputs [num_seqs+1]

    Returns:
        Tuple of:
            - output: Attention output [batch, seq_len, num_heads, head_dim]
            - final_state: Final hidden state [batch, num_heads, head_dim, head_dim]

    Examples:
        >>>
        >>> query = jnp.ones((2, 100, 8, 64))
        >>> key = jnp.ones((2, 100, 8, 64))
        >>> value = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = recurrent(query, key, value)
        >>> output.shape
        (2, 100, 8, 64)

        >>>
        >>> g = jnp.ones((2, 100, 8, 64))
        >>> output, final_state = recurrent(query, key, value, g=g)

        >>>
        >>> g_gamma = -jnp.arange(8, dtype=jnp.float32) * 0.1
        >>> output, final_state = recurrent(query, key, value, g_gamma=g_gamma)

        >>>
        >>> query = jnp.ones((150, 8, 64))
        >>> key = jnp.ones((150, 8, 64))
        >>> value = jnp.ones((150, 8, 64))
        >>> cu_seqlens = jnp.array([0, 50, 100, 150])
        >>> output, final_state = recurrent(query, key, value, cu_seqlens=cu_seqlens)
        >>> output.shape
        (150, 8, 64)
        >>> final_state.shape
        (3, 8, 64, 64)
    """
    return _recurrent_core(query, key, value, g, g_gamma, gk, gv, softmax_scale, initial_state, reverse, cu_seqlens)
