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


"""Grouped matrix multiplication kernels for TPU written in Pallas."""

import dataclasses
import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def buffered_pallas_call(
    kernel: Callable[..., Any],
    out_shape: jax.ShapeDtypeStruct,
    grid_spec: pltpu.PrefetchScalarGridSpec,
    compiler_params: pltpu.CompilerParams,
    input_buffer_count: Sequence[int] | None = None,
    **kw,
):
    """Create a buffered Pallas call for TPU with custom prefetch and pipeline configuration.

    This function wraps a Pallas kernel with TPU-specific optimizations including:
    - Scalar memory prefetching for grid parameters
    - Input buffering for hiding memory latency
    - Pipeline scheduling with emit_pipeline for overlapping compute and memory ops
    - Proper memory space mapping (SMEM, HBM) for TPU memory hierarchy

    The wrapper handles the complexity of:
    1. Binding scalar prefetch values from SMEM to kernel index maps
    2. Configuring input buffer counts for double/triple buffering
    3. Setting up proper memory spaces for different argument types
    4. Integrating with TPU's emit_pipeline for efficient execution

    Args:
        kernel: The Pallas kernel function to execute. Should accept:
            - Scalar prefetch refs (from SMEM)
            - Input/output refs (data to process)
            - Scratch refs (temporary workspace)
        out_shape: Shape and dtype specification for the output tensor(s).
        grid_spec: PrefetchScalarGridSpec defining:
            - num_scalar_prefetch: Number of scalar values to prefetch
            - in_specs: BlockSpec list for input tensors
            - out_specs: BlockSpec for output tensor(s)
            - grid: Tuple of grid dimensions (can be static ints or dynamic arrays)
            - scratch_shapes: Workspace memory shapes
        compiler_params: TPU compiler parameters including dimension_semantics
            for specifying parallelization strategy.
        input_buffer_count: Optional sequence specifying buffer count for each input.
            Must match length of in_specs. Values > 2 enable multi-buffering.
            Default is 2 (double buffering) for all inputs.
        **kw: Additional keyword arguments passed through to pl.pallas_call.

    Returns:
        A callable that executes the buffered kernel when invoked with:
        - Scalar prefetch arguments
        - Regular input arguments

    Raises:
        ValueError: If input_buffer_count length doesn't match in_specs length.

    Example:
        >>> def my_kernel(group_meta, grid_meta, lhs_ref, rhs_ref, out_ref, scratch):
        ...
        ...     pass
        >>> grid_spec = pltpu.PrefetchScalarGridSpec(
        ...     num_scalar_prefetch=2,
        ...     in_specs=[lhs_spec, rhs_spec],
        ...     out_specs=out_spec,
        ...     grid=(tiles_n, tiles_m, tiles_k),
        ...     scratch_shapes=[pltpu.VMEM((128, 128), jnp.float32)]
        ... )
        >>> call_fn = buffered_pallas_call(
        ...     my_kernel,
        ...     out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
        ...     grid_spec=grid_spec,
        ...     compiler_params=pltpu.CompilerParams(...),
        ...     input_buffer_count=[2, 3]
        ... )
        >>> result = call_fn(group_metadata, grid_metadata, lhs, rhs)

    Notes:
        - Scalar prefetch values are stored in SMEM for fast access
        - Input/output data uses HBM (high-bandwidth memory)
        - Buffer counts > 2 increase memory usage but can hide more latency
        - The wrapper automatically handles grid dimension binding
    """
    num_scalar_prefetch = grid_spec.num_scalar_prefetch

    def len_(x):
        return len(x) if isinstance(x, (list, tuple)) else 1

    args_len = len_(grid_spec.in_specs) + len_(grid_spec.out_specs)

    def _augment_blockspec(bs, smem_refs):
        def index_map_(*idxs):
            return bs.index_map(*idxs, *smem_refs)

        return pl.BlockSpec(bs.block_shape, index_map_)

    grid_static = tuple(dim if isinstance(dim, int) else None for dim in grid_spec.grid)
    grid_dynamic = tuple(None if isinstance(dim, int) else jnp.atleast_1d(dim) for dim in grid_spec.grid)

    def _bind_pipeline(spec, count):
        if count == 2:
            return spec
        return dataclasses.replace(spec, pipeline_mode=pl.Buffered(buffer_count=count, use_lookahead=True))

    def pallas_call(*args):
        smem_args = args[:num_scalar_prefetch]

        def pipeline(*args_refs):
            grid = tuple(d[0] if d is not None else s for d, s in zip(args_refs[0], grid_static, strict=False))

            smem_refs = args_refs[1 : num_scalar_prefetch + 1]
            _bind_smem = functools.partial(_augment_blockspec, smem_refs=smem_refs)
            in_specs_ = jax.tree.map(_bind_smem, grid_spec.in_specs)
            if input_buffer_count is not None:
                if len(input_buffer_count) != len(in_specs_):
                    raise ValueError(f"`{input_buffer_count=}` must a list[int] equal in length to {len(in_specs_)=}.")

                in_specs_ = tuple(
                    jax.tree.map(functools.partial(_bind_pipeline, count=c), spec)
                    for spec, c in zip(in_specs_, input_buffer_count, strict=False)
                )
            out_specs_ = jax.tree.map(_bind_smem, grid_spec.out_specs)

            input_output_refs = args_refs[num_scalar_prefetch + 1 : num_scalar_prefetch + args_len + 1]
            scratch_refs = args_refs[num_scalar_prefetch + args_len + 1 :]

            def _pipeline(*args):
                return kernel(*smem_refs, *args, *scratch_refs)

            dim_sem = compiler_params.dimension_semantics
            _emit_pipeline = functools.partial(pltpu.emit_pipeline, dimension_semantics=dim_sem)
            _emit_pipeline(_pipeline, grid=grid, in_specs=in_specs_, out_specs=out_specs_)(*input_output_refs)

        bs_smem = pl.BlockSpec(memory_space=pltpu.SMEM)
        bs_hbm = pl.BlockSpec(memory_space=pltpu.ANY)

        smem_specs = (jax.tree.map(lambda _: bs_smem, grid_dynamic),)
        smem_specs += jax.tree.map(lambda _: bs_smem, smem_args)
        in_specs = jax.tree.map(lambda _: bs_hbm, tuple(grid_spec.in_specs))
        out_specs = jax.tree.map(lambda _: bs_hbm, grid_spec.out_specs)

        params = dataclasses.replace(compiler_params, dimension_semantics=())
        return pl.pallas_call(
            pipeline,
            out_shape,
            compiler_params=params,
            in_specs=smem_specs + in_specs,
            out_specs=out_specs,
            scratch_shapes=grid_spec.scratch_shapes,
            **kw,
        )(grid_dynamic, *args)

    return pallas_call
