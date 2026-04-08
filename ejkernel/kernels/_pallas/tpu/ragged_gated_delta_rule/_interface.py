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

"""TPU Pallas interface for Ragged Gated Delta Rule.

Registers the Pallas TPU implementation under ``"ragged_gated_delta_rule"``
for ``Platform.PALLAS, Backend.TPU``. The decode path uses a Pallas kernel
with per-token parallelism. The prefill path falls back to XLA chunked.
"""

from __future__ import annotations

import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.ragged_gated_delta_rule._xla_impl_fwd import (
    _ragged_gdr_chunked_prefill,
)
from ._pallas_impl_fwd import run_ragged_gdr_decode_pallas


def _decode_path(query, key, value, beta, decay, recurrent_state, state_indices, use_qk_l2norm):
    """Execute the Pallas decode path: L2 norm → gather → kernel → scatter."""
    if use_qk_l2norm:
        from ...._xla.gated_delta_rule._xla_impl_fwd import _l2norm

        query = _l2norm(query, axis=-1, eps=1e-6)
        key = _l2norm(key, axis=-1, eps=1e-6)

    gathered_state = recurrent_state[state_indices]

    output, updated_per_token_state = run_ragged_gdr_decode_pallas(
        query,
        key,
        value,
        beta,
        decay,
        gathered_state,
        use_l2norm=False,
    )

    updated_state = recurrent_state.at[state_indices].set(updated_per_token_state)
    return output, updated_state


def ragged_gated_delta_rule_decode(
    query: Float[Array, "num_tokens num_heads qk_head_dim"],
    key: Float[Array, "num_tokens num_heads qk_head_dim"],
    value: Float[Array, "num_tokens num_heads v_head_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads"],
    recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
    state_indices: Int[Array, "num_requests"],
    *,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "num_tokens num_heads v_head_dim"],
    Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
]:
    """Direct decode-only Pallas path without ``lax.cond`` dispatch."""
    return _decode_path(query, key, value, beta, decay, recurrent_state, state_indices, use_qk_l2norm)


@kernel_registry.register("ragged_gated_delta_rule", Platform.PALLAS, Backend.TPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def ragged_gated_delta_rule(
    query: Float[Array, "num_tokens num_heads qk_head_dim"],
    key: Float[Array, "num_tokens num_heads qk_head_dim"],
    value: Float[Array, "num_tokens num_heads v_head_dim"],
    beta: Float[Array, "num_tokens num_heads"],
    decay: Float[Array, "num_tokens num_heads"] | None,
    recurrent_state: Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
    query_start_loc: Int[Array, "num_requests_plus_1"],
    state_indices: Int[Array, "num_requests"],
    *,
    chunk_size: int = 64,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "num_tokens num_heads v_head_dim"],
    Float[Array, "num_slots num_heads qk_head_dim v_head_dim"],
]:
    """Ragged GDR with Pallas TPU decode kernel and XLA chunked prefill fallback."""
    if decay is None:
        decay = jnp.zeros_like(beta)

    seq_lengths = query_start_loc[1:] - query_start_loc[:-1]
    is_all_decode = jnp.all(seq_lengths <= 1)

    # _decode_path gathers recurrent_state[state_indices] and passes it
    # to a Pallas kernel whose output shape is (query.shape[0], ...) with
    # input_output_aliases={5:1}.  Both must match, so state_indices must
    # have exactly num_tokens entries.  lax.cond traces both branches, so
    # num_tokens may differ from len(state_indices) at trace time.
    num_tokens = query.shape[0]
    num_si = state_indices.shape[0]
    if num_tokens > num_si:
        decode_state_indices = jnp.pad(state_indices, (0, num_tokens - num_si))
    elif num_tokens < num_si:
        decode_state_indices = state_indices[:num_tokens]
    else:
        decode_state_indices = state_indices

    def decode_fn(_):
        return _decode_path(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            decode_state_indices,
            use_qk_l2norm,
        )

    def prefill_fn(_):
        new_state, out = _ragged_gdr_chunked_prefill(
            query,
            key,
            value,
            beta,
            decay,
            recurrent_state,
            query_start_loc,
            state_indices,
            chunk_size,
            use_qk_l2norm,
        )
        return out, new_state

    return lax.cond(is_all_decode, decode_fn, prefill_fn, operand=None)
