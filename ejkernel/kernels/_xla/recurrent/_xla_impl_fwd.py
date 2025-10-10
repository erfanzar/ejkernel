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
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _recurrent_attention_step(
    carry: tuple[Float[Array, "num_heads key_dim value_dim"]],
    inputs: tuple,
    softmax_scale: float,
    use_g: bool,
    use_g_gamma: bool,
    use_gk: bool,
    use_gv: bool,
) -> tuple[tuple[Float[Array, "num_heads key_dim value_dim"]], Float[Array, "num_heads value_dim"]]:
    """
    Single step of recurrent linear attention.

    Updates hidden state: h_t = decay * h_{t-1} + k_t^T @ v_t
    Computes output: o_t = h_t @ q_t

    Args:
        carry: Hidden state (h,) where h is [num_heads, key_dim, value_dim]
        inputs: Tuple of (q, k, v, g, g_gamma, gk, gv) for current timestep
        softmax_scale: Query scaling factor
        use_g, use_g_gamma, use_gk, use_gv: Flags for gating mechanisms

    Returns:
        Updated carry and output for this timestep
    """
    (h,) = carry
    q, k, v, g, g_gamma, gk, gv = inputs

    if use_g:
        decay = jnp.exp(g)[:, :, None]
        h = h * decay

    if use_g_gamma:
        decay = jnp.exp(g_gamma)[:, None, None]
        h = h * decay

    if use_gk:
        gk_decay = jnp.exp(gk)[:, :, None]
        h = h * gk_decay

    if use_gv:
        gv_decay = jnp.exp(gv)[:, None, :]
        h = h * gv_decay

    h = h + k[:, :, None] * v[:, None, :]

    q_scaled = q * softmax_scale
    o = jnp.sum(h * q_scaled[:, :, None], axis=1)

    return (h,), o


def _recurrent_attention_fwd(
    q: Float[Array, "batch seq_len num_heads head_dim"],
    k: Float[Array, "batch seq_len num_heads head_dim"],
    v: Float[Array, "batch seq_len num_heads head_dim"],
    g: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "num_heads"] | None = None,
    gk: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    gv: Float[Array, "batch seq_len num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "batch num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
) -> tuple[Float[Array, "batch seq_len num_heads head_dim"], Float[Array, "batch num_heads head_dim head_dim"]]:
    """
    Forward pass for recurrent linear attention.

    Processes sequences sequentially with O(N) complexity by maintaining
    a hidden state h that accumulates key-value information.

    Args:
        q, k, v: Query, key, value tensors [batch, seq_len, num_heads, head_dim]
        g: Optional gate for GLA-style gating [batch, seq_len, num_heads, head_dim]
        g_gamma: Optional per-head decay factor [num_heads]
        gk, gv: Optional gates for keys/values [batch, seq_len, num_heads, head_dim]
        softmax_scale: Query scaling factor
        initial_state: Initial hidden state [batch, num_heads, head_dim, head_dim]
        reverse: If True, process sequence in reverse

    Returns:
        Tuple of (output, final_state)
    """
    batch, seq_len, num_heads, head_dim = q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)

    use_g = g is not None
    use_g_gamma = g_gamma is not None
    use_gk = gk is not None
    use_gv = gv is not None

    if reverse:
        q = jnp.flip(q, axis=1)
        k = jnp.flip(k, axis=1)
        v = jnp.flip(v, axis=1)
        if use_g:
            g = jnp.flip(g, axis=1)
        if use_gk:
            gk = jnp.flip(gk, axis=1)
        if use_gv:
            gv = jnp.flip(gv, axis=1)

    if g is None:
        g = jnp.zeros((batch, seq_len, num_heads, head_dim))
    if g_gamma is None:
        g_gamma = jnp.zeros((num_heads,))
    if gk is None:
        gk = jnp.zeros((batch, seq_len, num_heads, head_dim))
    if gv is None:
        gv = jnp.zeros((batch, seq_len, num_heads, head_dim))

    def process_batch(q_b, k_b, v_b, g_b, gk_b, gv_b, h0):
        """Process a single batch element."""

        g_gamma_seq = jnp.tile(g_gamma[None, :], (seq_len, 1))

        def scan_fn(carry, inputs):
            (h,) = carry
            (h_new,), o = _recurrent_attention_step((h,), inputs, softmax_scale, use_g, use_g_gamma, use_gk, use_gv)

            return (h_new,), (h_new, o)

        scan_inputs = (q_b, k_b, v_b, g_b, g_gamma_seq, gk_b, gv_b)

        (h_final,), (hidden_states, outputs) = jax.lax.scan(scan_fn, (h0,), scan_inputs)

        return outputs, hidden_states, h_final

    if initial_state is not None:
        h0_batch = initial_state
    else:
        h0_batch = jnp.zeros((batch, num_heads, head_dim, head_dim))

    outputs, hidden_states, final_states = jax.vmap(process_batch)(q, k, v, g, gk, gv, h0_batch)

    if reverse:
        outputs = jnp.flip(outputs, axis=1)
        hidden_states = jnp.flip(hidden_states, axis=1)

    return outputs, hidden_states, final_states


def _recurrent_attention_varlen_fwd(
    q: Float[Array, "total_tokens num_heads head_dim"],
    k: Float[Array, "total_tokens num_heads head_dim"],
    v: Float[Array, "total_tokens num_heads head_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"],
    g: Float[Array, "total_tokens num_heads head_dim"] | None = None,
    g_gamma: Float[Array, "num_heads"] | None = None,
    gk: Float[Array, "total_tokens num_heads head_dim"] | None = None,
    gv: Float[Array, "total_tokens num_heads head_dim"] | None = None,
    softmax_scale: float | None = None,
    initial_state: Float[Array, "num_seqs num_heads head_dim head_dim"] | None = None,
    reverse: bool = False,
) -> tuple[Float[Array, "total_tokens num_heads head_dim"], Float[Array, "num_seqs num_heads head_dim head_dim"]]:
    """
    Forward pass for recurrent linear attention with variable-length sequences.

    Args:
        q, k, v: Query, key, value tensors [total_tokens, num_heads, head_dim]
        cu_seqlens: Cumulative sequence lengths [num_seqs + 1]
        ... (other args same as fixed-length version)

    Returns:
        Tuple of (output, final_states)
    """
    num_seqs = len(cu_seqlens) - 1
    head_dim = q.shape[2]

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(jnp.float32)

    def process_sequence(seq_idx):
        """Process a single variable-length sequence."""
        start = cu_seqlens[seq_idx]
        end = cu_seqlens[seq_idx + 1]

        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        g_seq = g[start:end] if g is not None else None
        gk_seq = gk[start:end] if gk is not None else None
        gv_seq = gv[start:end] if gv is not None else None

        h0 = initial_state[seq_idx] if initial_state is not None else None

        q_batch = q_seq[None, ...]
        k_batch = k_seq[None, ...]
        v_batch = v_seq[None, ...]
        g_batch = g_seq[None, ...] if g_seq is not None else None
        gk_batch = gk_seq[None, ...] if gk_seq is not None else None
        gv_batch = gv_seq[None, ...] if gv_seq is not None else None
        h0_batch = h0[None, ...] if h0 is not None else None

        o_batch, h_final_batch = _recurrent_attention_fwd(
            q_batch,
            k_batch,
            v_batch,
            g=g_batch,
            g_gamma=g_gamma,
            gk=gk_batch,
            gv=gv_batch,
            softmax_scale=softmax_scale,
            initial_state=h0_batch,
            reverse=reverse,
        )

        return o_batch[0], h_final_batch[0]

    outputs_list = []
    final_states_list = []

    for seq_idx in range(num_seqs):
        o_seq, h_final = process_sequence(seq_idx)
        outputs_list.append(o_seq)
        final_states_list.append(h_final)

    outputs = jnp.concatenate(outputs_list, axis=0)
    final_states = jnp.stack(final_states_list, axis=0)

    return outputs, final_states
