#!/usr/bin/env python3
"""Shared benchmark helpers for ejkernel operations."""

from __future__ import annotations

import importlib.util
import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from ejkernel.benchmarks import Benchmark
from ejkernel.kernels._registry import Backend, kernel_registry
from ejkernel.modules import operations as ops
from ejkernel.quantization import prepack_quantized_weights
from ejkernel.utils import make_dummy_rpa_inputs


@dataclass
class OpBenchmarkSpec:
    """Specification for a single operation benchmark.

    Encapsulates everything needed to benchmark a kernel operation: the
    operation callable, how to generate inputs, which configurations to
    sweep, and how to wrap the operation for each platform.

    Attributes:
        op_name: Human-readable name used for logging and plot filenames.
        algorithm: Registry key used to look up kernel implementations.
        op_fn: The operation callable to benchmark.
        input_generator: Factory that produces input tensors from a config dict.
        configs: List of configuration dicts defining the parameter sweep.
        static_kwargs: Argument names treated as compile-time constants.
        bench_bwd: Whether to also benchmark the backward pass.
        needs_platform: Whether the operation requires a ``platform`` argument.
        wrapper_factory: Optional factory to build a platform-specific wrapper
            around ``op_fn``.  When ``None``, the default wrapper is used.
    """

    op_name: str
    algorithm: str
    op_fn: Callable[..., Any]
    input_generator: Callable[[dict[str, Any]], tuple[Any, ...]]
    configs: list[dict[str, Any]]
    static_kwargs: list[str] | None = None
    bench_bwd: bool = False
    needs_platform: bool = True
    wrapper_factory: Callable[[Callable[..., Any], str], Callable[..., Any]] | None = None


def _default_dtype() -> jnp.dtype:
    """Return a sensible default dtype for the current JAX backend.

    Returns:
        ``bfloat16`` on TPU, ``float32`` on CPU, ``float16`` on GPU.
    """
    backend = jax.default_backend()
    if backend == "tpu":
        return jnp.bfloat16
    if backend == "cpu":
        return jnp.float32
    return jnp.float16


def _available_platforms(algorithm: str) -> list[str]:
    """List kernel platforms available for *algorithm* on the current backend.

    Filters out the ``triton`` platform when the triton package is not
    installed.

    Args:
        algorithm: Registry algorithm name (e.g. ``"flash_attention"``).

    Returns:
        Sorted, deduplicated list of platform name strings.
    """
    backend = Backend(jax.default_backend())
    platforms: list[str] = []
    for spec in kernel_registry.list_implementations(algorithm):
        if spec.backend not in (Backend.ANY, backend):
            continue
        platforms.append(spec.platform.value)
    if "triton" in platforms and importlib.util.find_spec("triton") is None:
        platforms = [p for p in platforms if p != "triton"]
    return sorted(set(platforms))


def _parse_platform_list(value: str | None) -> list[str]:
    """Parse a comma-or-space-separated platform string into a list.

    Args:
        value: Raw string such as ``"triton, cuda"`` or ``None``.

    Returns:
        List of non-empty platform name strings, or empty list if *value*
        is falsy.
    """
    if not value:
        return []
    return [item for item in value.replace(",", " ").split() if item]


def _ignored_platforms(extra: list[str] | None = None) -> set[str]:
    """Collect the set of platforms to skip during benchmarking.

    Reads ``EJKERNEL_BENCH_IGNORE_PLATFORMS`` from the environment and
    merges in any *extra* names supplied by the caller.

    Args:
        extra: Additional platform names to ignore.

    Returns:
        Lower-cased set of platform names to exclude.
    """
    env_value = os.getenv("EJKERNEL_BENCH_IGNORE_PLATFORMS")
    items = _parse_platform_list(env_value)
    if extra:
        items.extend(extra)
    return {item.lower() for item in items}


def _wrap_op(op_fn: Callable[..., Any], platform: str) -> Callable[..., Any]:
    """Wrap *op_fn* to inject ``platform`` and unwrap tuple outputs.

    Args:
        op_fn: The operation callable.
        platform: Platform name passed as a keyword argument.

    Returns:
        A wrapper that calls *op_fn* with ``platform=platform`` and returns
        the first element if the result is a tuple.
    """

    def _fn(*args):
        out = op_fn(*args, platform=platform)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _wrap_op_with_kwargs(op_fn: Callable[..., Any], platform: str, **fixed_kwargs) -> Callable[..., Any]:
    """Wrap *op_fn* with ``platform`` and additional fixed keyword arguments.

    Args:
        op_fn: The operation callable.
        platform: Platform name passed as a keyword argument.
        **fixed_kwargs: Extra keyword arguments forwarded on every call.

    Returns:
        A wrapper that calls *op_fn* with the given kwargs and unwraps
        tuple outputs.
    """

    def _fn(*args):
        out = op_fn(*args, platform=platform, **fixed_kwargs)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _wrap_op_no_platform(op_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap *op_fn* without injecting a platform, unwrapping tuple outputs.

    Args:
        op_fn: The operation callable.

    Returns:
        A wrapper that returns the first element when *op_fn* produces a tuple.
    """

    def _fn(*args):
        out = op_fn(*args)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _wrap_attention_like(op_fn: Callable[..., Any], platform: str) -> Callable[..., Any]:
    """Wrap an attention-like op to accept positional (q, k, v, causal, sliding_window).

    Args:
        op_fn: An attention operation that takes keyword args for ``causal``,
            ``sliding_window``, and ``platform``.
        platform: Platform name injected on every call.

    Returns:
        A wrapper accepting ``(q, k, v, causal, sliding_window)`` positionally.
    """

    def _fn(q, k, v, causal, sliding_window):
        out = op_fn(q, k, v, causal=causal, sliding_window=sliding_window, platform=platform)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _apply_native_sparse_op(*args, platform: str):
    """Dispatch apply_native_sparse_attention through the kernel registry.

    Args:
        *args: Positional arguments forwarded to the kernel implementation.
        platform: Platform name used to resolve the implementation.

    Returns:
        The result of the resolved kernel implementation.
    """
    backend = Backend(jax.default_backend())
    impl = kernel_registry.get("apply_native_sparse_attention", platform=platform, backend=backend)
    return impl(*args)


def _limit_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Truncate *configs* to the limit set by ``EJKERNEL_BENCH_CONFIG_LIMIT``.

    When the environment variable is unset or not a positive integer the
    full list is returned unchanged.

    Args:
        configs: Full list of benchmark configuration dicts.

    Returns:
        Possibly truncated list of configurations.
    """
    limit = os.getenv("EJKERNEL_BENCH_CONFIG_LIMIT")
    if limit is None:
        return configs
    try:
        cap = int(limit)
    except ValueError:
        return configs
    if cap <= 0:
        return configs
    return configs[:cap]


def _grid(**options: list[Any]) -> list[dict[str, Any]]:
    """Build the Cartesian product of *options* as a list of config dicts.

    Args:
        **options: Mapping of parameter name to a list of values to sweep.

    Returns:
        List of dicts, one per combination in the Cartesian product.
    """
    keys = list(options.keys())
    values = [options[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def _cfgs_mha():
    """Generate benchmark configs for multi-head attention operations."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[256, 512, 1024, 2048, 4096],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        causal=[False, True],
        sliding=[None, (256, 256)],
    )
    return _limit_configs(configs)


def _cfgs_blocksparse():
    """Generate benchmark configs for block-sparse attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[256, 512, 1024, 2048, 4096],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        causal=[True],
        sliding=[None, (128, 128), (256, 256)],
    )
    return _limit_configs(configs)


def _cfgs_native_sparse():
    """Generate benchmark configs for native sparse attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[256, 512, 1024],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        block_counts=[8, 16, 32],
    )
    return _limit_configs(configs)


def _cfgs_apply_native_sparse():
    """Generate benchmark configs for apply-native-sparse attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[256, 512, 1024],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        block_size=[16, 32],
        num_selected_blocks=[4, 8, 16],
    )
    return _limit_configs(configs)


def _cfgs_decode_attention():
    """Generate benchmark configs for paged decode attention."""
    configs = _grid(
        batch=[1, 2, 4, 8],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        page_size=[4, 8, 16],
        max_pages=[8, 16, 32],
    )
    return _limit_configs(configs)


def _cfgs_ragged_decode():
    """Generate benchmark configs for ragged decode attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512, 1024],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_page_attention():
    """Generate benchmark configs for page attention."""
    configs = _grid(
        batch=[1, 2, 4, 8],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        block_size=[16, 32],
        pages_per_seq=[2, 4, 8],
    )
    return _limit_configs(configs)


def _cfgs_prefill_page_attention():
    """Generate benchmark configs for prefill page attention."""
    configs = _grid(
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        chunk_size=[64, 128, 256],
        page_size=[16, 32],
        total_pages=[8, 16, 32],
    )
    return _limit_configs(configs)


def _cfgs_chunked_prefill():
    """Generate benchmark configs for chunked prefill paged decode."""
    configs = _grid(
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        num_seqs=[2, 4, 8],
        q_len=[8, 16, 32],
        block_size=[16, 32],
        pages_per_seq=[2, 4],
    )
    return _limit_configs(configs)


def _cfgs_rpa_v2():
    """Generate benchmark configs for ragged page attention v2."""
    configs = _grid(
        num_seqs=[2, 4, 8],
        pages_per_seq=[2, 4],
        page_size=[8, 16, 32],
        qheads=[4, 8],
        kvheads=[2, 4],
        dim=[32, 64, 128],
    )
    return _limit_configs(configs)


def _cfgs_rpa_v3():
    """Generate benchmark configs for ragged page attention v3."""
    configs = _grid(
        num_seqs=[8],
        pages_per_seq=[128],
        page_size=[32, 64],
        qheads=[8],
        kvheads=[4],
        dim=[128],
        total_q=[2048],
    )
    return _limit_configs(configs)


def _cfgs_unified():
    """Generate hand-picked benchmark configs for unified attention."""
    return [
        {"num_seqs": 2, "qheads": 4, "kvheads": 2, "dim": 64, "block_size": 16, "kv_lens": [16, 12], "q_lens": [16, 12]},
        {
            "num_seqs": 4,
            "qheads": 8,
            "kvheads": 4,
            "dim": 64,
            "block_size": 16,
            "kv_lens": [32, 24, 28, 16],
            "q_lens": [32, 24, 28, 16],
        },
        {
            "num_seqs": 8,
            "qheads": 8,
            "kvheads": 4,
            "dim": 128,
            "block_size": 16,
            "kv_lens": [64, 48, 56, 32, 40, 44, 52, 36],
            "q_lens": [64, 48, 56, 32, 40, 44, 52, 36],
        },
    ]


def _cfgs_kernel_delta():
    """Generate benchmark configs for kernel delta attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        heads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_gdr():
    """Generate benchmark configs for gated delta rule."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512, 1024],
        heads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_lightning():
    """Generate benchmark configs for lightning attention."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[256, 512, 1024],
        qheads=[4, 8],
        kvheads=[4, 8],
        dim=[64, 128],
        causal=[False, True],
        sliding=[None],
        layer_idx=[0, 6, 12],
        num_layers=[24],
    )
    return _limit_configs(configs)


def _cfgs_grouped_matmul():
    """Generate benchmark configs for grouped matrix multiplication."""
    configs = _grid(
        groups=[4, 8, 16],
        m_per_group=[32, 64],
        k=[64, 128, 256],
        n=[64, 128, 256],
    )
    return _limit_configs(configs)


def _cfgs_quantized_matmul():
    """Generate benchmark configs for quantized matrix multiplication."""
    configs = _grid(
        m=[128, 512, 2048, 4096],
        k=[4096, 8192],
        n=[128, 512, 1024, 4096],
        mode=["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"],
        dtype=["bf16"],
    )
    return _limit_configs(configs)


def _cfgs_mean_pooling():
    """Generate benchmark configs for mean pooling."""
    configs = _grid(
        batch=[1, 2, 4, 8],
        seq=[256, 512, 1024, 2048],
        dim=[256, 512, 1024],
    )
    return _limit_configs(configs)


def _cfgs_rwkv4():
    """Generate benchmark configs for RWKV-4 recurrence."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        chans=[256, 512],
    )
    return _limit_configs(configs)


def _cfgs_rwkv6():
    """Generate benchmark configs for RWKV-6 recurrence."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        heads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_rwkv7():
    """Generate benchmark configs for RWKV-7 recurrence."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        heads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_rwkv7_mul():
    """Generate benchmark configs for RWKV-7 multiplicative recurrence."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        heads=[4, 8],
        dim=[64, 128],
    )
    return _limit_configs(configs)


def _cfgs_state_space_v1():
    """Generate benchmark configs for state-space model v1 (Mamba-1)."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        intermediate_size=[64, 128, 256],
        ssm_state_size=[16, 32],
    )
    return _limit_configs(configs)


def _cfgs_state_space_v2():
    """Generate benchmark configs for state-space model v2 (Mamba-2)."""
    configs = _grid(
        batch=[1, 2, 4],
        seq=[128, 256, 512],
        heads=[4, 8],
        dim=[32, 64],
        ssm_state_size=[16, 32],
        n_groups=[1],
    )
    return _limit_configs(configs)


def _rand_inputs(config: dict[str, Any], *shapes: tuple[int, ...], dtype: jnp.dtype | None = None):
    """Generate a list of random normal tensors with the given shapes.

    Args:
        config: Benchmark config dict; ``seed`` is read if present.
        *shapes: One shape tuple per desired output tensor.
        dtype: Element type; falls back to ``_default_dtype()``.

    Returns:
        List of JAX arrays, one per shape.
    """
    key = jax.random.PRNGKey(config.get("seed", 0))
    dtype = _default_dtype() if dtype is None else dtype
    keys = jax.random.split(key, len(shapes))
    return [jax.random.normal(k, shape, dtype=dtype) for k, shape in zip(keys, shapes, strict=True)]


def _gen_mha_inputs(config: dict[str, Any]):
    """Generate random q, k, v tensors and attention flags for MHA benchmarks."""
    batch = config["batch"]
    seq = config["seq"]
    qh = config["qheads"]
    kvh = config.get("kvheads", qh)
    dim = config["dim"]
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(config, (batch, seq, qh, dim), (batch, seq, kvh, dim), (batch, seq, kvh, dim), dtype=dtype)
    return q, k, v, config["causal"], config["sliding"]


def _gen_blocksparse_inputs(config: dict[str, Any]):
    """Generate random inputs for block-sparse attention (heads-first layout)."""
    batch = config["batch"]
    seq = config["seq"]
    qh = config["qheads"]
    kvh = config.get("kvheads", qh)
    dim = config["dim"]
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(
        config,
        (batch, qh, seq, dim),
        (batch, kvh, seq, dim),
        (batch, kvh, seq, dim),
        dtype=dtype,
    )
    return q, k, v, config["causal"], config["sliding"]


def _gen_native_sparse_inputs(config: dict[str, Any]):
    """Generate random q, k, v tensors and block_counts for native sparse attention."""
    batch = config["batch"]
    seq = config["seq"]
    qh = config["qheads"]
    kvh = config.get("kvheads", qh)
    dim = config["dim"]
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(config, (batch, seq, qh, dim), (batch, seq, kvh, dim), (batch, seq, kvh, dim), dtype=dtype)
    return q, k, v, config.get("block_counts", 16)


def _gen_apply_native_sparse_inputs(config: dict[str, Any]):
    """Generate inputs for apply-native-sparse attention including block indices."""
    batch = config["batch"]
    seq = config["seq"]
    qh = config["qheads"]
    kvh = config.get("kvheads", qh)
    dim = config["dim"]
    block_size = config.get("block_size", 16)
    num_selected = config.get("num_selected_blocks", 4)
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(config, (batch, seq, qh, dim), (batch, seq, kvh, dim), (batch, seq, kvh, dim), dtype=dtype)
    num_blocks = (seq + block_size - 1) // block_size
    key = jax.random.PRNGKey(config.get("seed", 0) ^ 0xA5A5)
    block_indices = jax.random.randint(
        key,
        (batch, seq, kvh, num_selected),
        minval=0,
        maxval=num_blocks,
        dtype=jnp.int32,
    )
    block_counts = num_selected
    return q, k, v, block_indices, block_counts, block_size


def _gen_decode_attention_inputs(config: dict[str, Any]):
    """Generate query, paged KV buffers, token map, and seq lengths for decode attention."""
    batch = config["batch"]
    heads = config["qheads"]
    kvh = config.get("kvheads", heads)
    dim = config["dim"]
    page_size = config.get("page_size", 4)
    max_pages = config.get("max_pages", 8)
    total_tokens = page_size * max_pages
    dtype = config.get("dtype", _default_dtype())
    q, k_buf, v_buf = _rand_inputs(
        config,
        (batch, heads, dim),
        (total_tokens, kvh, dim),
        (total_tokens, kvh, dim),
        dtype=dtype,
    )
    req_to_tokens = jnp.tile(jnp.arange(max_pages, dtype=jnp.int32)[None, :], (batch, 1))
    seq_lens = jnp.full((batch,), total_tokens, dtype=jnp.int32)
    return q, k_buf, v_buf, req_to_tokens, seq_lens


def _gen_ragged_decode_inputs(config: dict[str, Any]):
    """Generate query, KV tensors, and sequence start/end for ragged decode attention."""
    batch = config["batch"]
    seq = config["seq"]
    heads = config["qheads"]
    kvh = config.get("kvheads", heads)
    dim = config["dim"]
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(config, (batch, heads, dim), (batch, seq, kvh, dim), (batch, seq, kvh, dim), dtype=dtype)
    seq_start = jnp.zeros((batch,), dtype=jnp.int32)
    seq_end = jnp.full((batch,), seq, dtype=jnp.int32)
    return q, k, v, seq_start, seq_end


def _gen_page_attention_inputs(config: dict[str, Any]):
    """Generate query, paged KV caches, context lengths, and block tables for page attention."""
    num_seqs = config["batch"]
    heads = config["qheads"]
    kvh = config.get("kvheads", heads)
    dim = config["dim"]
    block_size = config.get("block_size", 16)
    pages_per_seq = config.get("pages_per_seq", 4)
    num_blocks = num_seqs * pages_per_seq
    dtype = config.get("dtype", _default_dtype())
    query = jax.random.normal(jax.random.PRNGKey(config.get("seed", 0)), (num_seqs, heads, dim), dtype=dtype)
    key_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 1)), (num_blocks, kvh, block_size, dim), dtype=dtype
    )
    value_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 2)), (num_blocks, kvh, block_size, dim), dtype=dtype
    )
    context_lens = jnp.full((num_seqs,), block_size * pages_per_seq, dtype=jnp.int32)
    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    return query, key_cache, value_cache, context_lens, block_tables


def _gen_prefill_page_attention_inputs(config: dict[str, Any]):
    """Generate query chunk, paged KV caches, context length, and page indices for prefill page attention."""
    chunk_size = config.get("chunk_size", 128)
    heads = config["qheads"]
    kvh = config.get("kvheads", heads)
    dim = config["dim"]
    page_size = config.get("page_size", 16)
    total_pages = config.get("total_pages", 8)
    dtype = config.get("dtype", _default_dtype())
    query = jax.random.normal(jax.random.PRNGKey(config.get("seed", 0)), (chunk_size, heads, dim), dtype=dtype)
    key_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 1)), (kvh, total_pages, page_size, dim), dtype=dtype
    )
    value_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 2)), (kvh, total_pages, page_size, dim), dtype=dtype
    )
    context_len = jnp.array([total_pages * page_size], dtype=jnp.int32)
    page_indices = jnp.arange(total_pages, dtype=jnp.int32)
    return query, key_cache, value_cache, context_len, page_indices


def _gen_chunked_prefill_inputs(config: dict[str, Any]):
    """Generate q/k/v, paged KV caches, KV lengths, block tables, and query offsets for chunked prefill."""
    num_seqs = config.get("num_seqs", 2)
    q_len = config.get("q_len", 8)
    heads = config["qheads"]
    kvh = config.get("kvheads", heads)
    dim = config["dim"]
    block_size = config.get("block_size", 16)
    pages_per_seq = config.get("pages_per_seq", 2)
    num_blocks = num_seqs * pages_per_seq
    total_tokens = num_seqs * q_len
    dtype = config.get("dtype", _default_dtype())
    q, k, v = _rand_inputs(
        config,
        (total_tokens, heads, dim),
        (total_tokens, kvh, dim),
        (total_tokens, kvh, dim),
        dtype=dtype,
    )
    key_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 1)), (num_blocks, block_size, kvh, dim), dtype=dtype
    )
    value_cache = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 2)), (num_blocks, block_size, kvh, dim), dtype=dtype
    )
    kv_lens = jnp.full((num_seqs,), q_len, dtype=jnp.int32)
    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    query_start_loc = jnp.arange(0, total_tokens + 1, q_len, dtype=jnp.int32)
    return q, k, v, key_cache, value_cache, kv_lens, block_tables, query_start_loc


def _make_kv_pages(num_pages: int, page_size: int, kv_heads: int, head_dim: int, seed: int, dtype: jnp.dtype):
    """Create interleaved KV page tensor of shape ``(num_pages, page_size, kv_heads*2, head_dim)``."""
    k = jax.random.normal(
        jax.random.PRNGKey(seed), (num_pages, page_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)
    v = jax.random.normal(
        jax.random.PRNGKey(seed ^ 0xA5A5), (num_pages, page_size, kv_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)
    kv = jnp.stack([k, v], axis=3).reshape(num_pages, page_size, kv_heads * 2, head_dim)
    return kv


def _gen_rpa_v2_inputs(config: dict[str, Any]):
    """Generate inputs for ragged page attention v2 including KV pages and query offsets."""
    num_seqs = config.get("num_seqs", 2)
    pages_per_seq = config.get("pages_per_seq", 2)
    page_size = config.get("page_size", 8)
    kv_heads = config.get("kvheads", 2)
    q_heads = config.get("qheads", 4)
    head_dim = config.get("dim", 32)
    dtype = config.get("dtype", _default_dtype())
    num_pages = num_seqs * pages_per_seq

    context_lens = jnp.full((num_seqs,), page_size * pages_per_seq, dtype=jnp.int32)
    q_lens = context_lens
    query_start_loc = jnp.concatenate([jnp.array([0], dtype=jnp.int32), jnp.cumsum(q_lens, dtype=jnp.int32)])
    total_tokens = int(query_start_loc[-1])

    queries = jax.random.normal(
        jax.random.PRNGKey(config.get("seed", 0)), (total_tokens, q_heads, head_dim), dtype=jnp.float32
    ).astype(dtype)
    kv_pages = _make_kv_pages(num_pages, page_size, kv_heads, head_dim, seed=1, dtype=dtype)
    block_tables = jnp.arange(num_pages, dtype=jnp.int32).reshape(num_seqs, pages_per_seq)
    softmax_aux = jnp.zeros((q_heads,), dtype=jnp.float32)
    return queries, kv_pages, context_lens, block_tables, query_start_loc, num_seqs, softmax_aux


def _gen_rpa_v3_inputs(config: dict[str, Any]):
    """Generate inputs for ragged page attention v3 using the ``make_dummy_rpa_inputs`` helper."""
    inputs = make_dummy_rpa_inputs(
        rng_seed=config.get("seed", 0),
        num_seqs=config.get("num_seqs", 2),
        pages_per_seq=config.get("pages_per_seq", 2),
        page_size=config.get("page_size", 16),
        num_q_heads=config.get("qheads", 4),
        num_kv_heads=config.get("kvheads", 2),
        head_dim=config.get("dim", 64),
        kv_dtype=config.get("dtype", _default_dtype()),
        total_q=config.get("total_q", 8),
    )
    softmax_aux = jnp.zeros((inputs["queries"].shape[1],), dtype=jnp.float32)
    return (
        inputs["queries"],
        inputs["keys"],
        inputs["values"],
        inputs["kv_cache"],
        inputs["kv_lens"],
        inputs["block_tables"],
        inputs["query_start_loc"],
        inputs["distribution"],
        softmax_aux,
    )


def _make_unified_inputs(config: dict[str, Any]):
    """Generate queries, paged KV caches, lengths, block tables, and query offsets for unified attention."""
    num_seqs = config.get("num_seqs", 2)
    q_heads = config.get("qheads", 4)
    kv_heads = config.get("kvheads", 2)
    head_dim = config.get("dim", 64)
    block_size = config.get("block_size", 16)
    kv_lens = config.get("kv_lens", [16, 12])
    if len(kv_lens) != num_seqs:
        kv_lens = [kv_lens[0]] * num_seqs
    q_lens = config.get("q_lens", kv_lens)
    max_kv = max(kv_lens)
    max_blocks = (max_kv + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks

    block_tables = jnp.arange(num_blocks, dtype=jnp.int32).reshape(num_seqs, max_blocks)
    kv_lens_arr = jnp.array(kv_lens, dtype=jnp.int32)
    query_start = [0]
    for q in q_lens:
        query_start.append(query_start[-1] + int(q))
    query_start_loc = jnp.array(query_start, dtype=jnp.int32)
    total_tokens = int(query_start_loc[-1])

    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3 = jax.random.split(key, 3)
    queries = jax.random.normal(k1, (total_tokens, q_heads, head_dim), dtype=jnp.float32).astype(dtype)
    key_cache = jax.random.normal(k2, (num_blocks, block_size, kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    value_cache = jax.random.normal(k3, (num_blocks, block_size, kv_heads, head_dim), dtype=jnp.float32).astype(dtype)
    return queries, key_cache, value_cache, kv_lens_arr, block_tables, query_start_loc


def _gen_grouped_matmul_inputs(config: dict[str, Any]):
    """Generate LHS, RHS, and group_sizes tensors for grouped matmul."""
    groups = config.get("groups", 4)
    m_per = config.get("m_per_group", 32)
    k = config.get("k", 64)
    n = config.get("n", 64)
    dtype = config.get("dtype", _default_dtype())
    m = groups * m_per
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2 = jax.random.split(key, 2)
    lhs = jax.random.normal(k1, (m, k), dtype=dtype)
    rhs = jax.random.normal(k2, (groups, k, n), dtype=dtype)
    group_sizes = jnp.full((groups,), m_per, dtype=jnp.int32)
    return lhs, rhs, group_sizes


def _gen_quantized_matmul_inputs(config: dict[str, Any]):
    """Generate activation, quantized weight, scales, biases, and mode for quantized matmul."""
    m = config.get("m", 64)
    k = config.get("k", 64)
    n = config.get("n", 64)
    mode = config.get("mode", "affine")
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2 = jax.random.split(key, 2)
    x = jax.random.normal(k1, (m, k), dtype=dtype)
    w = jax.random.normal(k2, (n, k), dtype=dtype)
    packed = prepack_quantized_weights(w, mode=mode)
    if mode == "affine":
        w_q, scales, biases = packed
    else:
        w_q, scales = packed
        biases = None
    return x, w_q, scales, biases, mode


def _gen_rwkv4_inputs(config: dict[str, Any]):
    """Generate w, u, k, v tensors for RWKV-4 WKV recurrence."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    chans = config.get("chans", 256)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    w = jax.random.normal(k1, (chans,), dtype=jnp.float32).astype(dtype)
    u = jax.random.normal(k2, (chans,), dtype=jnp.float32).astype(dtype)
    k = jax.random.normal(k3, (batch, seq, chans), dtype=dtype)
    v = jax.random.normal(k4, (batch, seq, chans), dtype=dtype)
    return w, u, k, v


def _gen_rwkv6_inputs(config: dict[str, Any]):
    """Generate r, k, v, w, u tensors for RWKV-6 recurrence."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 64)
    vdim = config.get("vdim", dim)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    r = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, seq, heads, dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, seq, heads, vdim), dtype=dtype)
    w = jax.random.normal(k4, (batch, seq, heads, dim), dtype=dtype)
    u = jax.random.normal(k5, (heads, dim), dtype=dtype)
    return r, k, v, w, u


def _gen_rwkv7_inputs(config: dict[str, Any]):
    """Generate r, w, k, v, a, b tensors for RWKV-7 recurrence."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 64)
    vdim = config.get("vdim", dim)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    r = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    w = jax.random.normal(k2, (batch, seq, heads, dim), dtype=dtype)
    k = jax.random.normal(k3, (batch, seq, heads, dim), dtype=dtype)
    v = jax.random.normal(k4, (batch, seq, heads, vdim), dtype=dtype)
    a = jax.random.normal(k5, (batch, seq, heads, dim), dtype=dtype)
    b = jax.random.normal(k6, (batch, seq, heads, dim), dtype=dtype)
    return r, w, k, v, a, b


def _gen_rwkv7_mul_inputs(config: dict[str, Any]):
    """Generate r, w, k, v, kk, a tensors for RWKV-7 multiplicative recurrence."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 64)
    vdim = config.get("vdim", dim)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    r = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    w = jax.random.normal(k2, (batch, seq, heads, dim), dtype=dtype)
    k = jax.random.normal(k3, (batch, seq, heads, dim), dtype=dtype)
    v = jax.random.normal(k4, (batch, seq, heads, vdim), dtype=dtype)
    kk = jax.random.normal(k5, (batch, seq, heads, dim), dtype=dtype)
    a = jax.random.normal(k6, (batch, seq, heads, dim), dtype=dtype)
    return r, w, k, v, kk, a


def _gen_state_space_v1_inputs(config: dict[str, Any]):
    """Generate hidden, A, B, C, D, dt tensors for state-space model v1."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    inter = config.get("intermediate_size", 64)
    ssm = config.get("ssm_state_size", 16)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    hidden = jax.random.normal(k1, (batch, seq, inter), dtype=dtype)
    A = jax.random.normal(k2, (inter, ssm), dtype=jnp.float32).astype(dtype)
    B = jax.random.normal(k3, (batch, seq, ssm), dtype=dtype)
    C = jax.random.normal(k4, (batch, seq, ssm), dtype=dtype)
    D = jax.random.normal(k5, (inter,), dtype=jnp.float32).astype(dtype)
    dt = jax.random.normal(k6, (batch, seq, inter), dtype=dtype)
    return hidden, A, B, C, D, dt


def _gen_state_space_v2_inputs(config: dict[str, Any]):
    """Generate x, A, B, C, D, dt tensors for state-space model v2."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 32)
    ssm = config.get("ssm_state_size", 16)
    n_groups = config.get("n_groups", 1)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    x = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    A = jax.random.normal(k2, (heads,), dtype=jnp.float32).astype(dtype)
    B = jax.random.normal(k3, (batch, seq, n_groups, ssm), dtype=dtype)
    C = jax.random.normal(k4, (batch, seq, n_groups, ssm), dtype=dtype)
    D = jax.random.normal(k5, (heads,), dtype=jnp.float32).astype(dtype)
    dt = jax.random.normal(jax.random.PRNGKey(config.get("seed", 6)), (batch, seq, heads), dtype=dtype)
    return x, A, B, C, D, dt


def _gen_simple_seq_inputs(config: dict[str, Any]):
    """Generate a single random ``(batch, seq, dim)`` tensor for simple sequence ops."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    dim = config.get("dim", 256)
    dtype = config.get("dtype", _default_dtype())
    x = jax.random.normal(jax.random.PRNGKey(config.get("seed", 0)), (batch, seq, dim), dtype=dtype)
    return (x,)


def _gen_kernel_delta_inputs(config: dict[str, Any]):
    """Generate q, k, v, beta tensors for kernel delta attention."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 64)
    vdim = config.get("vdim", dim)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    q = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, seq, heads, dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, seq, heads, vdim), dtype=dtype)
    beta = jax.random.uniform(k4, (batch, seq, heads), minval=0.0, maxval=1.0, dtype=jnp.float32)
    return q, k, v, beta


def _gen_gdr_inputs(config: dict[str, Any]):
    """Generate q, k, v, beta, decay tensors for gated delta rule."""
    batch = config.get("batch", 2)
    seq = config.get("seq", 128)
    heads = config.get("heads", 4)
    dim = config.get("dim", 64)
    dtype = config.get("dtype", _default_dtype())
    key = jax.random.PRNGKey(config.get("seed", 0))
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    q = jax.random.normal(k1, (batch, seq, heads, dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, seq, heads, dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, seq, heads, dim), dtype=dtype)
    beta = jax.nn.sigmoid(jax.random.normal(k4, (batch, seq, heads), dtype=jnp.float32))
    decay = jax.random.normal(k5, (batch, seq, heads), dtype=jnp.float32) * -0.01
    return q, k, v, beta, decay


def _gen_lightning_inputs(config: dict[str, Any]):
    """Generate q, k, v, layer_idx, num_layers for lightning attention."""
    q, k, v, _, _ = _gen_mha_inputs(config)
    layer_idx = config.get("layer_idx", 0)
    num_layers = config.get("num_layers", 24)
    return q, k, v, layer_idx, num_layers


def _gen_gla_inputs(config: dict[str, Any]):
    """Generate q, k, v tensors for gated linear attention."""
    q, k, v, _, _ = _gen_mha_inputs(config)
    return q, k, v


def _gen_recurrent_inputs(config: dict[str, Any]):
    """Generate q, k, v tensors for recurrent attention."""
    q, k, v, _, _ = _gen_mha_inputs(config)
    return q, k, v


def _gen_unified_inputs(config: dict[str, Any]):
    """Generate inputs for unified attention (delegates to ``_make_unified_inputs``)."""
    return _make_unified_inputs(config)


def _build_algorithms(
    spec: OpBenchmarkSpec,
    *,
    ignore_platforms: set[str] | None = None,
) -> dict[str, Callable[..., Any]]:
    """Build a mapping of platform name to wrapped callable for benchmarking.

    Args:
        spec: The benchmark specification to derive algorithms from.
        ignore_platforms: Set of platform names (lower-cased) to exclude.

    Returns:
        Dict mapping each available platform name to its wrapped operation callable.
    """
    platforms = _available_platforms(spec.algorithm)
    if ignore_platforms:
        platforms = [platform for platform in platforms if platform.lower() not in ignore_platforms]
    if not platforms:
        return {}
    algorithms: dict[str, Callable[..., Any]] = {}
    for platform in platforms:
        if spec.wrapper_factory is not None:
            fn = spec.wrapper_factory(spec.op_fn, platform)
        elif spec.needs_platform:
            fn = _wrap_op(spec.op_fn, platform)
        else:
            fn = _wrap_op_no_platform(spec.op_fn)
        algorithms[platform] = fn
    return algorithms


def run_benchmark(op_name: str, *, ignore_platforms: list[str] | None = None) -> int:
    """Run the full benchmark suite for the given operation.

    Looks up the ``OpBenchmarkSpec`` from the ``SPECS`` registry, builds
    platform-specific callables, and executes the benchmark with warmup
    and timing iterations.  Results are printed and a plot is saved to
    ``benchmark_plots/<op_name>``.

    Args:
        op_name: Name of the operation as registered in ``SPECS``.
        ignore_platforms: Additional platforms to skip.

    Returns:
        0 on success, 1 if the spec is missing or no implementations are found.
    """
    spec = SPECS.get(op_name)
    if spec is None:
        print(f"No benchmark spec registered for {op_name}")
        return 1

    algorithms = _build_algorithms(spec, ignore_platforms=_ignored_platforms(ignore_platforms))
    if not algorithms:
        print(f"No implementations found for {spec.algorithm} on this backend.")
        return 1

    bench = Benchmark(
        algorithms=algorithms,
        configs=spec.configs,
        input_generator=spec.input_generator,
        warmup=5,
        iterations=30,
        bench_bwd=spec.bench_bwd,
        static_kwargs=spec.static_kwargs,
        unpack_inputs=True,
    )

    bench.run(verbose=True)
    bench.plot(f"benchmark_plots/{spec.op_name}")
    return 0


def _quantized_wrapper(op_fn: Callable[..., Any], platform: str):
    """Wrap a quantized matmul op to accept ``mode`` positionally and inject ``platform``."""

    def _fn(x, w, scales, biases, mode):
        out = op_fn(x, w, scales, biases, mode=mode, platform=platform)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _lightning_wrapper(op_fn: Callable[..., Any], platform: str):
    """Wrap a lightning attention op to pass ``layer_idx``, ``num_layers``, and ``platform`` as kwargs."""

    def _fn(q, k, v, layer_idx, num_layers):
        out = op_fn(q, k, v, layer_idx=layer_idx, num_layers=num_layers, platform=platform)
        return out[0] if isinstance(out, tuple) else out

    return _fn


def _grouped_matmul_v2_wrapper(op_fn: Callable[..., Any], platform: str):
    """Wrap a grouped matmul op to enable v2 mode and inject ``platform``."""

    def _fn(lhs, rhs, group_sizes):
        out = op_fn(lhs, rhs, group_sizes, use_v2=True, platform=platform)
        return out[0] if isinstance(out, tuple) else out

    return _fn


SPECS: dict[str, OpBenchmarkSpec] = {
    "attention": OpBenchmarkSpec(
        op_name="attention",
        algorithm="attention",
        op_fn=ops.attention,
        input_generator=_gen_mha_inputs,
        configs=_cfgs_mha(),
        static_kwargs=["causal", "sliding_window"],
        wrapper_factory=_wrap_attention_like,
    ),
    "flash_attention": OpBenchmarkSpec(
        op_name="flash_attention",
        algorithm="flash_attention",
        op_fn=ops.flash_attention,
        input_generator=_gen_mha_inputs,
        configs=_cfgs_mha(),
        static_kwargs=["causal", "sliding_window"],
        wrapper_factory=_wrap_attention_like,
    ),
    "scaled_dot_product_attention": OpBenchmarkSpec(
        op_name="scaled_dot_product_attention",
        algorithm="scaled_dot_product_attention",
        op_fn=ops.scaled_dot_product_attention,
        input_generator=_gen_mha_inputs,
        configs=_cfgs_mha(),
        static_kwargs=["causal", "sliding_window"],
        wrapper_factory=_wrap_attention_like,
    ),
    "blocksparse_attention": OpBenchmarkSpec(
        op_name="blocksparse_attention",
        algorithm="blocksparse_attention",
        op_fn=ops.blocksparse_attention,
        input_generator=_gen_blocksparse_inputs,
        configs=_cfgs_blocksparse(),
        static_kwargs=["causal", "sliding_window"],
        wrapper_factory=_wrap_attention_like,
    ),
    "native_sparse_attention": OpBenchmarkSpec(
        op_name="native_sparse_attention",
        algorithm="native_sparse_attention",
        op_fn=ops.native_sparse_attention,
        input_generator=_gen_native_sparse_inputs,
        configs=_cfgs_native_sparse(),
    ),
    "apply_native_sparse_attention": OpBenchmarkSpec(
        op_name="apply_native_sparse_attention",
        algorithm="apply_native_sparse_attention",
        op_fn=_apply_native_sparse_op,
        input_generator=_gen_apply_native_sparse_inputs,
        configs=_cfgs_apply_native_sparse(),
    ),
    "decode_attention": OpBenchmarkSpec(
        op_name="decode_attention",
        algorithm="decode_attention",
        op_fn=ops.decode_attention,
        input_generator=_gen_decode_attention_inputs,
        configs=_cfgs_decode_attention(),
    ),
    "ragged_decode_attention": OpBenchmarkSpec(
        op_name="ragged_decode_attention",
        algorithm="ragged_decode_attention",
        op_fn=ops.ragged_decode_attention,
        input_generator=_gen_ragged_decode_inputs,
        configs=_cfgs_ragged_decode(),
    ),
    "page_attention": OpBenchmarkSpec(
        op_name="page_attention",
        algorithm="page_attention",
        op_fn=ops.page_attention,
        input_generator=_gen_page_attention_inputs,
        configs=_cfgs_page_attention(),
    ),
    "prefill_page_attention": OpBenchmarkSpec(
        op_name="prefill_page_attention",
        algorithm="prefill_page_attention",
        op_fn=ops.prefill_page_attention,
        input_generator=_gen_prefill_page_attention_inputs,
        configs=_cfgs_prefill_page_attention(),
    ),
    "chunked_prefill_paged_decode": OpBenchmarkSpec(
        op_name="chunked_prefill_paged_decode",
        algorithm="chunked_prefill_paged_decode",
        op_fn=ops.chunked_prefill_paged_decode,
        input_generator=_gen_chunked_prefill_inputs,
        configs=_cfgs_chunked_prefill(),
    ),
    "ragged_page_attention_v2": OpBenchmarkSpec(
        op_name="ragged_page_attention_v2",
        algorithm="ragged_page_attention_v2",
        op_fn=ops.ragged_page_attention_v2,
        input_generator=_gen_rpa_v2_inputs,
        configs=_cfgs_rpa_v2(),
    ),
    "ragged_page_attention_v3": OpBenchmarkSpec(
        op_name="ragged_page_attention_v3",
        algorithm="ragged_page_attention_v3",
        op_fn=ops.ragged_page_attention_v3,
        input_generator=_gen_rpa_v3_inputs,
        configs=_cfgs_rpa_v3(),
    ),
    "unified_attention": OpBenchmarkSpec(
        op_name="unified_attention",
        algorithm="unified_attention",
        op_fn=ops.unified_attention,
        input_generator=_gen_unified_inputs,
        configs=_cfgs_unified(),
    ),
    "gla_attention": OpBenchmarkSpec(
        op_name="gla_attention",
        algorithm="gla",
        op_fn=ops.gla_attention,
        input_generator=_gen_gla_inputs,
        configs=_cfgs_mha(),
    ),
    "recurrent_attention": OpBenchmarkSpec(
        op_name="recurrent_attention",
        algorithm="recurrent",
        op_fn=ops.recurrent_attention,
        input_generator=_gen_recurrent_inputs,
        configs=_cfgs_mha(),
    ),
    "kernel_delta_attention": OpBenchmarkSpec(
        op_name="kernel_delta_attention",
        algorithm="kernel_delta_attention",
        op_fn=ops.kernel_delta_attention,
        input_generator=_gen_kernel_delta_inputs,
        configs=_cfgs_kernel_delta(),
    ),
    "gated_delta_rule": OpBenchmarkSpec(
        op_name="gated_delta_rule",
        algorithm="gated_delta_rule",
        op_fn=ops.gated_delta_rule,
        input_generator=_gen_gdr_inputs,
        configs=_cfgs_gdr(),
        bench_bwd=True,
    ),
    "lightning_attention": OpBenchmarkSpec(
        op_name="lightning_attention",
        algorithm="lightning_attn",
        op_fn=ops.lightning_attention,
        input_generator=_gen_lightning_inputs,
        configs=_cfgs_lightning(),
        wrapper_factory=_lightning_wrapper,
        static_kwargs=["layer_idx", "num_layers"],
    ),
    "ring_attention": OpBenchmarkSpec(
        op_name="ring_attention",
        algorithm="ring_attention",
        op_fn=ops.ring_attention,
        input_generator=_gen_mha_inputs,
        configs=_cfgs_mha(),
        static_kwargs=["causal", "sliding_window"],
        wrapper_factory=_wrap_attention_like,
    ),
    "grouped_matmul": OpBenchmarkSpec(
        op_name="grouped_matmul",
        algorithm="grouped_matmul",
        op_fn=ops.grouped_matmul,
        input_generator=_gen_grouped_matmul_inputs,
        configs=_cfgs_grouped_matmul(),
    ),
    "grouped_matmulv2": OpBenchmarkSpec(
        op_name="grouped_matmulv2",
        algorithm="grouped_matmulv2",
        op_fn=ops.grouped_matmul,
        input_generator=_gen_grouped_matmul_inputs,
        configs=_cfgs_grouped_matmul(),
        wrapper_factory=_grouped_matmul_v2_wrapper,
    ),
    "quantized_matmul": OpBenchmarkSpec(
        op_name="quantized_matmul",
        algorithm="quantized_matmul",
        op_fn=ops.quantized_matmul,
        input_generator=_gen_quantized_matmul_inputs,
        configs=_cfgs_quantized_matmul(),
        wrapper_factory=_quantized_wrapper,
        static_kwargs=["mode"],
        bench_bwd=True,
    ),
    "mean_pooling": OpBenchmarkSpec(
        op_name="mean_pooling",
        algorithm="mean_pooling",
        op_fn=ops.mean_pooling,
        input_generator=_gen_simple_seq_inputs,
        configs=_cfgs_mean_pooling(),
    ),
    "rwkv4": OpBenchmarkSpec(
        op_name="rwkv4",
        algorithm="rwkv4",
        op_fn=ops.rwkv4,
        input_generator=_gen_rwkv4_inputs,
        configs=_cfgs_rwkv4(),
    ),
    "rwkv6": OpBenchmarkSpec(
        op_name="rwkv6",
        algorithm="rwkv6",
        op_fn=ops.rwkv6,
        input_generator=_gen_rwkv6_inputs,
        configs=_cfgs_rwkv6(),
    ),
    "rwkv7": OpBenchmarkSpec(
        op_name="rwkv7",
        algorithm="rwkv7",
        op_fn=ops.rwkv7,
        input_generator=_gen_rwkv7_inputs,
        configs=_cfgs_rwkv7(),
    ),
    "rwkv7_mul": OpBenchmarkSpec(
        op_name="rwkv7_mul",
        algorithm="rwkv7_mul",
        op_fn=ops.rwkv7_mul,
        input_generator=_gen_rwkv7_mul_inputs,
        configs=_cfgs_rwkv7_mul(),
    ),
    "state_space_v1": OpBenchmarkSpec(
        op_name="state_space_v1",
        algorithm="state_space_v1",
        op_fn=ops.state_space_v1,
        input_generator=_gen_state_space_v1_inputs,
        configs=_cfgs_state_space_v1(),
    ),
    "state_space_v2": OpBenchmarkSpec(
        op_name="state_space_v2",
        algorithm="state_space_v2",
        op_fn=ops.state_space_v2,
        input_generator=_gen_state_space_v2_inputs,
        configs=_cfgs_state_space_v2(),
    ),
}
