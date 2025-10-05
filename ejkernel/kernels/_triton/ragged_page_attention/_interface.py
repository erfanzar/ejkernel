# Copyright 2023 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

from ejkernel.callib import cdiv, triton_call

from ..._registry import Backend, Platform, kernel_registry
from ._triton_impl_fwd import _ragged_paged_attn_prefetch_kernel_combined

# Default mask value matching Pallas implementation
DEFAULT_MASK_VALUE = -2.381976426469702e38


@kernel_registry.register("ragged_page_attention", Platform.TRITON, Backend.GPU)
def ragged_page_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    kv_pages: Float[Array, "num_pages page_size num_combined_kv_heads head_dim"],
    context_lens: Int[Array, "num_seqs"],
    block_tables: Int[Array, "num_seqs pages_per_seq"],
    query_start_loc: Int[Array, "num_seqs_plus_one"],
    num_seqs: Array | int,
    *,
    softmax_scale: float | None = None,
    logit_soft_cap: float | None = None,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    optimized: bool = False,
    sliding_window: int | None = None,
    softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    del compute_dtype, optimized
    T, QH, D = queries.shape
    if softmax_aux is not None:
        raise NotImplementedError("`softmax_aux` is not implemented in triton impl yet!")
    # Validate unsupported TPU-specific parameters
    if num_queries_per_block is not None:
        raise NotImplementedError("num_queries_per_block is TPU-specific and not supported on GPU")
    if vmem_limit_bytes is not None:
        raise NotImplementedError("vmem_limit_bytes is TPU-specific and not supported on GPU")

    # Handle parameters with defaults
    if softmax_scale is None:
        softmax_scale = 1.0 / (D**0.5)  # Default to 1/sqrt(head_dim)

    kv_pages_per_block = num_kv_pages_per_block if num_kv_pages_per_block is not None else 8

    # Default mask_value to large negative number (match Pallas default)
    if mask_value is None:
        mask_value = -2.381976426469702e38

    # Set feature flags
    use_soft_cap = logit_soft_cap is not None
    soft_cap_value = logit_soft_cap if use_soft_cap else 1.0  # dummy value when not used

    use_sliding_window = sliding_window is not None
    sliding_window_size = sliding_window if use_sliding_window else 0  # dummy value when not used
    _P, PS, C, Dk = kv_pages.shape
    assert D == Dk, "head_size mismatch"
    assert C % 2 == 0, "combined kv heads must be even"
    KVH = C // 2
    assert QH % KVH == 0
    QHG = QH // KVH
    pages_per_seq_max = int(block_tables.shape[1])

    q4 = queries.reshape(T, KVH, QHG, D)
    T_padded = max(T, 1)
    if T_padded > T:
        pad = jnp.zeros((T_padded - T, KVH, QHG, D), dtype=q4.dtype)
        q4 = jnp.concatenate([q4, pad], axis=0)

    # per-row metadata
    starts = query_start_loc[:-1]
    ends = query_start_loc[1:]
    q_lens = (ends - starts).astype(jnp.int32)

    t_idx = jnp.arange(T_padded, dtype=jnp.int32)
    t_clamped = jnp.minimum(t_idx, jnp.int32(max(T - 1, 0)))
    row_seq = jnp.searchsorted(ends, t_clamped, side="right").astype(jnp.int32)

    row_start = starts[row_seq]
    row_qlen = q_lens[row_seq]
    row_kvlen = context_lens[row_seq]
    row_qoff = t_idx - row_start
    row_firstk = (row_kvlen - row_qlen + row_qoff).astype(jnp.int32)

    # active rows gating
    ns_dev = jnp.asarray(num_seqs, dtype=jnp.int32)  # 0-d device scalar
    row_valid = (t_idx < T) & (row_seq < ns_dev)

    KV_PAGES_PER_BLOCK = int(kv_pages_per_block)
    MAX_KV_SUPERBLOCKS = cdiv(pages_per_seq_max, KV_PAGES_PER_BLOCK)

    out_shape = jax.ShapeDtypeStruct((T_padded, KVH, QHG, D), jnp.float32)

    def grid(meta):
        return (T_padded, KVH, QHG)

    metaparams = dict(
        grid=grid,
        T=T_padded,
        KVH=KVH,
        QHG=QHG,
        D=D,
        PS=PS,
        PAGES_PER_SEQ_MAX=pages_per_seq_max,
        KV_PAGES_PER_BLOCK=KV_PAGES_PER_BLOCK,
        MAX_KV_SUPERBLOCKS=int(MAX_KV_SUPERBLOCKS),
        SCALE=softmax_scale,
        SOFT_CAP=soft_cap_value,
        USE_SOFT_CAP=use_soft_cap,
        SLIDING_WINDOW_SIZE=sliding_window_size,
        USE_SLIDING_WINDOW=use_sliding_window,
        MASK_VALUE=mask_value,
    )

    # Remove hardcoded num_warps and num_stages to let autotune decide
    out4_padded = triton_call(
        q4,
        kv_pages,
        block_tables.astype(jnp.int32),
        row_seq.astype(jnp.int32),
        row_firstk.astype(jnp.int32),
        row_kvlen.astype(jnp.int32),
        row_valid.astype(jnp.bool_),
        kernel=_ragged_paged_attn_prefetch_kernel_combined,
        out_shape=out_shape,
        **metaparams,
    )

    return out4_padded[:T].reshape(T, QH, D).astype(queries.dtype)
