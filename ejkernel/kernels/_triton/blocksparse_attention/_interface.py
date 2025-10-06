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


import functools
import math

import jax
import jaxtyping
from beartype import beartype
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array, Bool, DTypeLike, Float, Int

from ejkernel.callib import ejit

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_bwd import _blocksparse_bwd_attention_kernel_call
from ._triton_impl_fwd import _blocksparse_fwd_attention_kernel_call


def _transpose_layout_q2k_to_k2q(layout_q2k: jnp.ndarray, degree_q2k: jnp.ndarray, n_k_blocks: int, max_degree_k: int):
    lists = [[] for _ in range(n_k_blocks)]
    n_q_blocks, _max_degree_q = layout_q2k.shape
    for qb in range(n_q_blocks):
        deg = int(degree_q2k[qb])
        for li in range(deg):
            kb = int(layout_q2k[qb, li])
            if 0 <= kb < n_k_blocks:
                lists[kb].append(qb)

    layout_k2q = -jnp.ones((n_k_blocks, max_degree_k), dtype=jnp.int32)
    degree_k2q = jnp.zeros((n_k_blocks,), dtype=jnp.int32)
    for kb in range(n_k_blocks):
        deg = min(len(lists[kb]), max_degree_k)
        degree_k2q = degree_k2q.at[kb].set(deg)
        if deg > 0:
            layout_k2q = layout_k2q.at[kb, :deg].set(jnp.array(lists[kb][:deg], dtype=jnp.int32))

    layout_k2q = jnp.where(layout_k2q < 0, 0, layout_k2q)
    return layout_k2q, degree_k2q


def _jax_fwd_blocksparse_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    layout: Int[Array, "n_q_blocks max_degree"] | None = None,
    degree: Int[Array, "n_q_blocks"] | None = None,
    max_degree: int = 0,
    force_block_m: int = 0,
    force_block_n: int = 0,
):
    out, lse = _blocksparse_fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        layout=layout,
        degree=degree,
        max_degree=max_degree,
        force_block_m=force_block_m,
        force_block_n=force_block_n,
    )

    return out, (
        query,
        key,
        value,
        bias,
        attention_mask,
        out,
        lse,
        dropout_seed,
        cum_seqlens_q,
        cum_seqlens_k,
        layout,
        degree,
        max_degree,
        sliding_window,
        force_block_m,
        force_block_n,
        logits_soft_cap,
        softmax_aux,
    )


def _jax_bwd_blocksparse_call(
    softmax_scale: float | None,
    dropout_prob: float,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    logits_soft_cap: float | None,
    residual: tuple[Float[Array, "..."], ...],
    dO: Float[Array, "batch seq_len num_heads head_dim"],
):
    (
        query,
        key,
        value,
        bias,
        attention_mask,
        out,
        lse,
        dropout_seed,
        cum_seqlens_q,
        cum_seqlens_k,
        layout_q2k,
        degree_q2k,
        max_degree,
        sliding_window_fwd,
        force_block_m,
        force_block_n,
        logits_soft_cap_fwd,
        softmax_aux,
    ) = residual

    window = sliding_window if sliding_window is not None else sliding_window_fwd
    cap = logits_soft_cap if logits_soft_cap is not None else logits_soft_cap_fwd

    if layout_q2k is not None:
        if force_block_m == 0 or force_block_n == 0:
            raise ValueError("blocksparse backward requires force_block_m/force_block_n when using layout")
        QSeq = query.shape[1]
        KSeq = key.shape[1]
        _n_q_blocks = math.ceil(QSeq / force_block_m)
        n_k_blocks = math.ceil(KSeq / force_block_n)
        max_degree_k = max_degree
        layout_k2q, degree_k2q = _transpose_layout_q2k_to_k2q(layout_q2k, degree_q2k, n_k_blocks, max_degree_k)
    else:
        layout_k2q = degree_k2q = None
        max_degree_k = 0

    dq, dk, dv = _blocksparse_bwd_attention_kernel_call(
        dO=dO,
        q=query,
        k=key,
        v=value,
        bias=bias,
        attention_mask=attention_mask,
        o=out,
        M=lse,
        dropout_prob=dropout_prob,
        causal=causal,
        softmax_scale=softmax_scale,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=window,
        logits_soft_cap=cap,
        softmax_aux=softmax_aux,
        layout_q2k=layout_q2k,
        degree_q2k=degree_q2k,
        layout_k2q=layout_k2q,
        degree_k2q=degree_k2q,
        max_degree_q=(max_degree or 0),
        max_degree_k=(max_degree_k or 0),
        force_block_m=force_block_m,
        force_block_n=force_block_n,
    )
    return dq, dk, dv, None, None, None, None, None, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18))
@ejit(static_argnums=(5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18))
def blocksparse_attention_call(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    layout: Int[Array, "n_q_blocks max_degree"] | None = None,
    degree: Int[Array, "n_q_blocks"] | None = None,
    max_degree: int = 0,
    force_block_m: int = 0,
    force_block_n: int = 0,
) -> Float[Array, "batch seq_len_q num_heads head_dim"]:
    return _blocksparse_fwd_attention_kernel_call(
        q=query,
        k=key,
        v=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        layout=layout,
        degree=degree,
        max_degree=max_degree,
        force_block_m=force_block_m,
        force_block_n=force_block_n,
    )


blocksparse_attention_call.defvjp(_jax_fwd_blocksparse_call, _jax_bwd_blocksparse_call)


@kernel_registry.register("blocksparse_attention_GPU", Platform.TRITON, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def blocksparse_attention(
    query: Float[Array, "batch seq_len_q num_heads head_dim"],
    key: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    value: Float[Array, "batch seq_len_k num_kv_heads head_dim"],
    attention_mask: Bool[Array, "batch seq_len"] | None = None,
    bias: Float[Array, "batch num_heads seq_len_q seq_len_k"] | None = None,
    softmax_scale: float | None = None,
    dropout_prob: float = 0.0,
    causal: bool = False,
    dropout_seed: int | None = None,
    cum_seqlens_q: Int[Array, "batch_plus_one"] | None = None,
    cum_seqlens_k: Int[Array, "batch_plus_one"] | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: Float[Array, "num_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    layout: Int[Array, "n_q_blocks max_degree"] | None = None,
    degree: Int[Array, "n_q_blocks"] | None = None,
    max_degree: int = 0,
    force_block_m: int = 0,
    force_block_n: int = 0,
    precision: lax.PrecisionLike = jax.lax.Precision.DEFAULT,
    logits_dtype: DTypeLike = jnp.float32,
    *,
    q_segment_ids: Int[Array, "batch seq_len_q"] | None = None,
    kv_segment_ids: Int[Array, "batch seq_len_k"] | None = None,
    debug: bool = False,
) -> tuple[Float[Array, "batch seq_len_q num_heads head_dim"], Float[Array, "batch num_heads seq_len_q"]]:
    del precision, logits_dtype, debug
    if q_segment_ids is not None or kv_segment_ids is not None:
        raise NotImplementedError("segment_ids are not supported in Triton implementation.")

    out, residuals = blocksparse_attention_call(
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        bias=bias,
        softmax_scale=softmax_scale,
        dropout_prob=dropout_prob,
        causal=causal,
        dropout_seed=dropout_seed,
        cum_seqlens_q=cum_seqlens_q,
        cum_seqlens_k=cum_seqlens_k,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        softmax_aux=softmax_aux,
        layout=layout,
        degree=degree,
        max_degree=max_degree,
        force_block_m=force_block_m,
        force_block_n=force_block_n,
    )

    lse = residuals[6]
    return out, lse
