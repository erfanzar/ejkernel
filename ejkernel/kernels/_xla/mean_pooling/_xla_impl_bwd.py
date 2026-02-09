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


"""Backward rules for XLA mean pooling custom VJP."""

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _mean_pooling_fwd(
    x: Float[Array, "... hidden_dim"],
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None = None,
) -> tuple[Float[Array, "batch hidden_dim"], tuple]:
    out = _mean_pooling_core(x, cu_seqlens)
    residual = (x.shape, cu_seqlens)
    return out, residual


def _mean_pooling_bwd(
    cu_seqlens: Int[Array, "num_seqs_plus_one"] | None,
    residual: tuple,
    g: Float[Array, "batch hidden_dim"],
) -> tuple[Float[Array, "... hidden_dim"]]:
    x_shape, _ = residual

    if cu_seqlens is not None:
        num_seqs = len(cu_seqlens) - 1

        def grad_sequence(i):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            seq_len = end - start
            return jnp.tile(g[i] / seq_len, (seq_len, 1))

        dx_list = [grad_sequence(i) for i in range(num_seqs)]
        dx = jnp.concatenate(dx_list, axis=0)
    else:
        seq_len = x_shape[1]
        dx = jnp.tile(g[:, None, :], (1, seq_len, 1)) / seq_len

    return (dx,)


__all__ = ("_mean_pooling_bwd", "_mean_pooling_fwd")
