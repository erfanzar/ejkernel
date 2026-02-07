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

"""Unified attention interface registered for the CuTe platform."""

from __future__ import annotations

import jax
import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float, Int32

from ejkernel.errors import EjkernelRuntimeError

from ..._registry import Backend, Platform, kernel_registry
from ._cute_impl import unified_attention_cute


def _is_gpu_array_or_unknown(arr: Array) -> bool:
    """Return True when placement is on GPU or cannot be inferred (e.g. tracer)."""
    device_attr = getattr(arr, "device", None)
    if device_attr is None:
        return jax.default_backend() in {"gpu", "cuda"}
    try:
        dev = device_attr() if callable(device_attr) else device_attr
    except Exception:
        return jax.default_backend() in {"gpu", "cuda"}
    if dev is None:
        return jax.default_backend() in {"gpu", "cuda"}
    return getattr(dev, "platform", None) in {"gpu", "cuda"}


@kernel_registry.register("unified_attention", Platform.CUTE, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def unified_attention(
    queries: Float[Array, "total_tokens num_q_heads head_dim"],
    key_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    value_cache: Float[Array, "num_blocks block_size num_kv_heads head_dim"],
    kv_lens: Int32[Array, "num_seqs"],
    block_tables: Int32[Array, "num_seqs max_blocks_per_seq"],
    query_start_loc: Int32[Array, "num_seqs_plus_1"],
    alibi_slopes: Float[Array, "num_q_heads"] | None = None,
    qq_bias: Float[Array, "num_query_tokens num_query_tokens"] | None = None,
    softmax_aux: Float[Array, "num_q_heads"] | None = None,
    *,
    softmax_scale: float | None = None,
    causal: bool = True,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    seq_threshold_3d: int | None = None,
    num_par_softmax_segments: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> Float[Array, "total_tokens num_q_heads head_dim"]:
    """Compute unified paged attention for the CuTe platform entry point.

    This function keeps the same signature and feature semantics as the Triton
    and XLA registrations so operation-level dispatch remains fully compatible.
    Internally it executes the Triton unified-attention implementation and
    raises when that implementation is unavailable.

    Args:
        queries: Packed query tensor, shape ``(total_tokens, num_q_heads, head_dim)``.
        key_cache: Paged key cache, shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Paged value cache with the same shape as ``key_cache``.
        kv_lens: Sequence KV lengths, shape ``(num_seqs,)``, dtype ``int32``.
        block_tables: Page table mapping, shape
            ``(num_seqs, max_blocks_per_seq)``, dtype ``int32``.
        query_start_loc: Prefix offsets of packed queries, shape
            ``(num_seqs + 1,)``, dtype ``int32``.
        alibi_slopes: Optional ALiBi slopes, shape ``(num_q_heads,)``.
        qq_bias: Optional query-query bias matrix.
        softmax_aux: Optional sink logits, shape ``(num_q_heads,)``.
        softmax_scale: Optional score scale. Defaults to ``1 / sqrt(head_dim)``.
        causal: Whether attention is causal. Must be ``True``.
        sliding_window: Optional local-attention window.
        logits_soft_cap: Optional tanh soft-capping value.
        seq_threshold_3d: Optional Triton decode-kernel threshold.
        num_par_softmax_segments: Optional Triton segmented-softmax factor.
        num_warps: Optional Triton kernel override.
        num_stages: Optional Triton kernel override.

    Returns:
        Attention output tensor with shape ``(total_tokens, num_q_heads, head_dim)``.

    Raises:
        EjkernelRuntimeError: If inputs are not placed on a GPU device.
    """
    if not _is_gpu_array_or_unknown(queries):
        raise EjkernelRuntimeError("unified_attention (platform=cute) requires GPU-resident inputs.")

    return unified_attention_cute(
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        kv_lens=kv_lens,
        query_start_loc=query_start_loc,
        softmax_scale=softmax_scale,
        causal=causal,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        seq_threshold_3d=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        softmax_aux=softmax_aux,
        num_warps=num_warps,
        num_stages=num_stages,
    )
