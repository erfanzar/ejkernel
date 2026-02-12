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

"""All-gather matmul kernel."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _infer_axis_size(axis_name: str) -> int | None:
    """Infer collective axis size from the active mapped context when available."""
    try:
        return jax.core.concrete_or_error(
            int,
            lax.psum(jnp.array(1, dtype=jnp.int32), axis_name=axis_name),
            f"collective axis '{axis_name}' size must be static.",
        )
    except Exception:
        return None


def _resolve_tp_size(tp_size: int | None, axis_name: str) -> int:
    """Resolve tensor-parallel world size using explicit value, axis context, then global device count."""
    resolved = int(tp_size) if tp_size is not None else (_infer_axis_size(axis_name) or int(jax.device_count()))
    if resolved < 1:
        raise ValueError(f"tp_size must be >= 1, got {resolved}.")
    return resolved


def _local_barrier(left_neighbor, right_neighbor, double_barrier: bool = True):
    """Barrier with neighbors using TPU semaphores."""
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in (left_neighbor, right_neighbor):
        pltpu.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
    pltpu.semaphore_wait(barrier_sem, 2)
    if double_barrier:

        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in (left_neighbor, right_neighbor):
                pltpu.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor,),
                    device_id_type=pltpu.DeviceIdType.MESH,
                )
            pltpu.semaphore_wait(second_barrier, 2)


def _all_gather_kernel(
    x_hbm_ref,
    y_hbm_ref,
    o_hbm_ref,
    x_hbm_scratch_ref,
    x_local_copy_sem,
    y_local_copy_sem,
    o_local_copy_sem,
    send_sems,
    recv_sems,
    x_vmem_scratch_ref,
    y_vmem_scratch_ref,
    o_vmem_scratch_ref,
    acc_vmem_scratch_ref,
    axis_name: str,
    bn: int,
    bk: int,
    rhs_transpose: bool = False,
):
    """Pallas kernel for all-gather.

    Args:
      x_hbm_ref: LHS of the matmul before all-gather.
      y_hbm_ref: RHS of the matmul.
      o_hbm_ref: Output of the matmul.
      x_hbm_scratch_ref: Scratch memory for LHS of the matmul.
      x_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
      y_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
      o_local_copy_sem: DMA semaphore for a local HBM-VMEM copy.
      send_sem: DMA semaphore for the remote send.
      capacity_sem: Capacity semaphore for the remote send.
      recv_sems: DMA semaphore for the remote receive.
      x_vmem_scratch_ref: Scratch memory for LHS of the matmul.
      y_vmem_scratch_ref: Scratch memory for RHS of the matmul.
      o_vmem_scratch_ref: Scratch memory for output of the matmul.
    """
    num_devices = pl.num_programs(0) - 2
    grid_n = pl.num_programs(1)
    grid_k = pl.num_programs(2)
    outer_step = pl.program_id(0)
    bn_i = pl.program_id(1)
    bk_i = pl.program_id(2)
    global_step_id = outer_step * grid_n * grid_k + bn_i * grid_k + bk_i
    mxu_total_steps = num_devices * grid_n * grid_k
    gn_by_gk = grid_n * grid_k
    my_id = lax.axis_index(axis_name)
    left_neighbor = lax.rem(my_id + num_devices - 1, jnp.int32(num_devices))
    right_neighbor = lax.rem(my_id + 1, jnp.int32(num_devices))
    x_hbm_receiving_slot = outer_step
    x_hbm_working_slot = outer_step - 1
    x_vmem_receiving_slot = outer_step % 2
    x_vmem_working_slot = (global_step_id - 1) // gn_by_gk % 2
    o_receiving_slot = lax.rem((global_step_id + grid_k - 1) // grid_k, 2)
    o_working_slot = 1 - o_receiving_slot
    m_per_device, _ = x_hbm_ref.shape
    m_per_device_per_direction = m_per_device // 2

    def _start_or_wait_copy(
        op: jax._src.pallas.mosaic.primitives.AsyncCopyDescriptor,
        wait: bool = False,
    ):
        if wait:
            op.wait()
        else:
            op.start()

    def _do_first_x_local_copy(wait: bool = False):
        k_slice = pl.ds(bk_i * bk, bk)
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_ref.at[:, k_slice],
            dst_ref=x_vmem_scratch_ref.at[x_vmem_receiving_slot, :, k_slice],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_left_local_copy(wait: bool = False):
        k_slice = pl.ds(bk_i * bk, bk)
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot, :m_per_device_per_direction, k_slice],
            dst_ref=x_vmem_scratch_ref.at[x_vmem_receiving_slot, :m_per_device_per_direction, k_slice],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_subsequent_x_right_local_copy(wait: bool = False):
        x_local_copy_op = pltpu.make_async_copy(
            src_ref=x_hbm_scratch_ref.at[
                x_hbm_working_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            dst_ref=x_vmem_scratch_ref.at[
                x_vmem_receiving_slot,
                m_per_device_per_direction:,
                pl.ds(bk_i * bk, bk),
            ],
            sem=x_local_copy_sem,
        )
        _start_or_wait_copy(x_local_copy_op, wait)

    def _do_y_local_copy(wait: bool = False):
        k_slice = pl.ds(bk_i * bk, bk)
        n_slice = pl.ds(bn_i * bn, bn)
        if rhs_transpose:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[n_slice, k_slice],
                dst_ref=y_vmem_scratch_ref.at[n_slice, k_slice],
                sem=y_local_copy_sem,
            )
        else:
            y_local_copy_op = pltpu.make_async_copy(
                src_ref=y_hbm_ref.at[k_slice, n_slice],
                dst_ref=y_vmem_scratch_ref.at[k_slice, n_slice],
                sem=y_local_copy_sem,
            )
        _start_or_wait_copy(y_local_copy_op, wait)

    def _do_first_left_remote_copy(wait: bool = False):
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot, 0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_first_right_remote_copy(wait: bool = False):
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_ref.at[m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_subsequent_left_remote_copy(wait: bool = False):
        left_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot, 0:m_per_device_per_direction],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot, 0:m_per_device_per_direction],
            send_sem=send_sems.at[0, outer_step],
            recv_sem=recv_sems.at[0, outer_step],
            device_id=(left_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(left_remote_copy_op, wait)

    def _do_subsequent_right_remote_copy(wait: bool = False):
        right_remote_copy_op = pltpu.make_async_remote_copy(
            src_ref=x_hbm_scratch_ref.at[x_hbm_working_slot, m_per_device_per_direction:m_per_device],
            dst_ref=x_hbm_scratch_ref.at[x_hbm_receiving_slot, m_per_device_per_direction:m_per_device],
            send_sem=send_sems.at[1, outer_step],
            recv_sem=recv_sems.at[1, outer_step],
            device_id=(right_neighbor,),
            device_id_type=pltpu.DeviceIdType.MESH,
        )
        _start_or_wait_copy(right_remote_copy_op, wait)

    def _do_mxu():
        working_global_step_id = global_step_id - 1
        working_bk_i = working_global_step_id % grid_k
        working_bn_i = working_global_step_id % gn_by_gk // grid_k
        k_slice = pl.ds(working_bk_i * bk, bk)
        n_slice = pl.ds(working_bn_i * bn, bn)

        if grid_k == 1:
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot][...]
                rhs = y_vmem_scratch_ref.at[n_slice, :][...]
                o_vmem_scratch_ref.at[o_receiving_slot][...] = lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
            else:
                o_vmem_scratch_ref.at[o_receiving_slot][...] = jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot][...],
                    y_vmem_scratch_ref.at[:, n_slice][...],
                    preferred_element_type=jnp.float32,
                ).astype(x_vmem_scratch_ref.dtype)
        else:
            if rhs_transpose:
                lhs = x_vmem_scratch_ref.at[x_vmem_working_slot, :, k_slice][...]
                rhs = y_vmem_scratch_ref.at[n_slice, k_slice][...]
                acc_vmem_scratch_ref[...] += lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1,), (1,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
            else:
                acc_vmem_scratch_ref[...] += jnp.dot(
                    x_vmem_scratch_ref.at[x_vmem_working_slot, :, k_slice][...],
                    y_vmem_scratch_ref.at[k_slice, n_slice][...],
                    preferred_element_type=jnp.float32,
                )

            @pl.when(working_bk_i == grid_k - 1)
            def _update():
                o_vmem_scratch_ref.at[o_receiving_slot][...] = acc_vmem_scratch_ref[...].astype(x_vmem_scratch_ref.dtype)

                acc_vmem_scratch_ref[...] = jnp.zeros_like(acc_vmem_scratch_ref)

    def _do_o_local_copy(wait: bool = False):
        working_global_step_id = global_step_id - grid_k - 1
        working_bn_i = (working_global_step_id % gn_by_gk) // grid_k
        n_slice = pl.ds(working_bn_i * bn, bn)
        offset = (global_step_id - 2) // gn_by_gk
        left_o_idx = (my_id + offset) % num_devices
        left_o_idx = left_o_idx * 2
        right_o_idx = (my_id - offset + num_devices) % num_devices
        right_o_idx = right_o_idx * 2 + 1

        o_left_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[o_working_slot, :m_per_device_per_direction],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * left_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        o_right_local_copy_op = pltpu.make_async_copy(
            src_ref=o_vmem_scratch_ref.at[o_working_slot, m_per_device_per_direction:],
            dst_ref=o_hbm_ref.at[
                pl.ds(
                    m_per_device_per_direction * right_o_idx,
                    m_per_device_per_direction,
                ),
                n_slice,
            ],
            sem=o_local_copy_sem,
        )
        _start_or_wait_copy(o_left_local_copy_op, wait)
        _start_or_wait_copy(o_right_local_copy_op, wait)

    @pl.when(global_step_id == 0)
    @jax.named_scope("_start_first_remote_copy")
    def _start_first_remote_copy():
        if grid_k > 1:
            acc_vmem_scratch_ref[...] = jnp.zeros_like(acc_vmem_scratch_ref)
        _local_barrier(left_neighbor, right_neighbor)
        _do_first_left_remote_copy(wait=False)
        _do_first_right_remote_copy(wait=False)

    cond_start_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % gn_by_gk == 0,
    )

    @pl.when(cond_start_subsequent_remote_copy)
    @jax.named_scope("_start_subsequent_remote_copy")
    def _start_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=False)
        _do_subsequent_right_remote_copy(wait=False)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_start_first_local_x_copy")
    def _start_first_x_local_copy():
        _do_first_x_local_copy(wait=False)

    cond_subsequent_x_local_copy = jnp.logical_and(jnp.logical_and(outer_step > 0, outer_step < num_devices), bn_i == 0)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_start_subsequent_x_local_copy")
    def _start_subsequent_x_local_copy():
        _do_subsequent_x_left_local_copy(wait=False)
        _do_subsequent_x_right_local_copy(wait=False)

    @pl.when(outer_step == 0)
    @jax.named_scope("_start_y_local_copy")
    def _start_y_local_copy():
        _do_y_local_copy(wait=False)

    def _get_start_o_local_copy_cond():
        if grid_k == 1:
            return jnp.logical_and(global_step_id >= 2, global_step_id < mxu_total_steps + 2)
        else:
            return jnp.logical_and(
                jnp.logical_and(
                    global_step_id >= grid_k + 1,
                    global_step_id < mxu_total_steps + grid_k + 1,
                ),
                global_step_id % grid_k == 1,
            )

    @pl.when(_get_start_o_local_copy_cond())
    @jax.named_scope("_start_o_local_copy")
    def _start_o_local_copy():
        _do_o_local_copy(wait=False)

    @pl.when(jnp.logical_and(global_step_id >= 1, global_step_id < 1 + mxu_total_steps))
    @jax.named_scope("_mxu")
    def _mxu():
        _do_mxu()

    def _get_wait_o_local_copy_cond():
        if grid_k == 1:
            return jnp.logical_and(global_step_id >= 2, global_step_id < mxu_total_steps + 2)
        else:
            return jnp.logical_and(
                jnp.logical_and(
                    global_step_id >= grid_k + 1,
                    global_step_id < mxu_total_steps + grid_k + 1,
                ),
                global_step_id % grid_k == 0,
            )

    @pl.when(_get_wait_o_local_copy_cond())
    @jax.named_scope("_wait_o_local_copy")
    def _wait_o_local_copy():
        _do_o_local_copy(wait=True)

    @pl.when(outer_step == 0)
    @jax.named_scope("_wait_y_local_copy")
    def _wait_y_local_copy():
        _do_y_local_copy(wait=True)

    @pl.when(jnp.logical_and(outer_step == 0, bn_i == 0))
    @jax.named_scope("_wait_first_x_local_copy")
    def _wait_first_x_local_copy():
        _do_first_x_local_copy(wait=True)

    @pl.when(cond_subsequent_x_local_copy)
    @jax.named_scope("_wait_subsequent_x_local_copy")
    def _wait_subsequent_x_local_copy():
        _do_subsequent_x_left_local_copy(wait=True)
        _do_subsequent_x_right_local_copy(wait=True)

    @pl.when(global_step_id == gn_by_gk - 1)
    @jax.named_scope("_wait_first_remote_copy")
    def _wait_first_remote_copy():
        _do_first_left_remote_copy(wait=True)
        _do_first_right_remote_copy(wait=True)

    cond_wait_subsequent_remote_copy = jnp.logical_and(
        jnp.logical_and(outer_step > 0, outer_step < num_devices - 1),
        global_step_id % gn_by_gk == gn_by_gk - 1,
    )

    @pl.when(cond_wait_subsequent_remote_copy)
    @jax.named_scope("_wait_subsequent_remote_copy")
    def _wait_subsequent_remote_copy():
        _do_subsequent_left_remote_copy(wait=True)
        _do_subsequent_right_remote_copy(wait=True)


def get_vmem_estimate_bytes(
    m,
    n,
    k,
    bn,
    acc_bytes,
    tp_size,
    x_dtype,
    y_dtype,
    out_dtype,
):
    """Returns the total vmem bytes used by the kernel."""
    m_per_device = m // tp_size
    n_per_device = n // tp_size
    y_vmem_bytes = (
        n_per_device
        * k
        * (dtypes.bit_width(y_dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(y_dtype))
        // 8
    )
    total_bytes = (
        2
        * m_per_device
        * k
        * (dtypes.bit_width(x_dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(x_dtype))
        // 8
        + y_vmem_bytes
        + 2
        * m
        * bn
        * (dtypes.bit_width(out_dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(out_dtype))
        // 8
        + acc_bytes
    )
    return total_bytes


def validate_inputs(x, y, tp_size, rhs_transpose=False):
    """Validates the inputs to the all_gather_matmul kernel."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
    if x.dtype != y.dtype:
        raise ValueError(f"Input dtypes must match, got {x.dtype} and {y.dtype}.")
    m_per_device, k = x.shape
    if rhs_transpose:
        n_per_device, k_from_y = y.shape
    else:
        k_from_y, n_per_device = y.shape
    if k != k_from_y:
        raise ValueError(f"Incompatible shapes for matmul: contracting dimension mismatch: {x.shape} and {y.shape}.")

    n = n_per_device * tp_size

    if k % 128 != 0:
        raise ValueError(f"k ({k}) must be divisible by 128.")

    if n % 128 != 0:
        raise ValueError(f"n ({n}) must be divisible by 128.")

    m_per_device_per_direction = m_per_device // 2
    if m_per_device_per_direction % 8 != 0:
        raise ValueError(
            f"x.shape[0] (local m_per_device) must be divisible by 16 for bidirectional ring, got {m_per_device}."
        )

    if m_per_device % 2 != 0:
        raise ValueError(f"x.shape[0] ({m_per_device}) must be divisible by 2.")


def all_gather_matmul(
    x: jax.Array,
    y: jax.Array,
    axis_name: str,
    tp_size: int | None = None,
    collective_id: int | None = 0,
    bn: int | None = None,
    bk: int | None = None,
    rhs_transpose: bool = False,
):
    """Performs all-gather on the input tensor and then a matmul.

    Args:
      x: Local LHS shard of shape [m_per_device, k].
      y: Local RHS shard of shape [k, n_per_device] or [n_per_device, k].
      axis_name: Name of the axis used by the surrounding shard_map/pmap.
      tp_size: Tensor-parallel world size for axis_name.
      collective_id: An integer used for barrier semaphore allocation.
      bn: Number of blocks in the n dimension.
      bk: Number of blocks in the k dimension.
      rhs_transpose: If True, y is transposed.

    Returns:
      all-gather(x, axis=0) @ y_shard
    """
    tp_size = _resolve_tp_size(tp_size, axis_name)
    if tp_size == 1:
        if x.ndim != 2 or y.ndim != 2:
            raise ValueError(f"Inputs must be 2D, got shapes {x.shape} and {y.shape}.")
        if x.dtype != y.dtype:
            raise ValueError(f"Input dtypes must match, got {x.dtype} and {y.dtype}.")
        if rhs_transpose:
            if x.shape[1] != y.shape[1]:
                raise ValueError(
                    f"Incompatible shapes for matmul: contracting dimension mismatch: {x.shape} and {y.shape}."
                )
            return jnp.dot(x, y.T, preferred_element_type=jnp.float32).astype(x.dtype)
        if x.shape[1] != y.shape[0]:
            raise ValueError(f"Incompatible shapes for matmul: contracting dimension mismatch: {x.shape} and {y.shape}.")
        return jnp.dot(x, y, preferred_element_type=jnp.float32).astype(x.dtype)

    m_per_device, k = x.shape
    m = m_per_device * tp_size
    if rhs_transpose:
        n_per_device, _ = y.shape
    else:
        _, n_per_device = y.shape
    n = n_per_device * tp_size

    validate_inputs(x, y, tp_size, rhs_transpose)
    if bn is None:
        bn = n_per_device
    if bk is None:
        bk = k
    bn = int(bn)
    bk = int(bk)
    if bn < 1 or bk < 1:
        raise ValueError(f"bn and bk must be positive, got {bn=} and {bk=}.")
    if n_per_device % bn != 0:
        raise ValueError(f"n_per_device ({n_per_device}) must be divisible by bn ({bn}).")
    if k % bk != 0:
        raise ValueError(f"k ({k}) must be divisible by bk ({bk}).")

    grid_n = n_per_device // bn
    grid_k = k // bk
    acc_shape = (m_per_device, bn)
    if grid_k == 1:
        acc_shape = (8, 128)
    acc_bytes = (
        acc_shape[0]
        * acc_shape[1]
        * (dtypes.bit_width(jnp.float32) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(jnp.float32))
        // 8
    )
    y_vmem_shape = (n_per_device, k) if rhs_transpose else (k, n_per_device)
    estimated_vmem_bytes = get_vmem_estimate_bytes(
        m,
        n,
        k,
        bn,
        acc_bytes,
        tp_size,
        x.dtype,
        y.dtype,
        x.dtype,
    )
    out_shape = [
        jax.ShapeDtypeStruct((m, n_per_device), x.dtype),
        jax.ShapeDtypeStruct((tp_size - 1, m_per_device, k), x.dtype),
    ]
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        out_specs=[
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM),
        ],
        scratch_shapes=(
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((2, tp_size - 1)),
            pltpu.SemaphoreType.DMA(
                (
                    2,
                    tp_size - 1,
                )
            ),
            pltpu.VMEM((2, m_per_device, k), x.dtype),
            pltpu.VMEM(y_vmem_shape, y.dtype),
            pltpu.VMEM((2, m_per_device, bn), x.dtype),
            pltpu.VMEM(acc_shape, jnp.float32),
        ),
        grid=(tp_size + 2, grid_n, grid_k),
    )
    flops = 2 * m * k * n_per_device
    bytes_accessed = x.dtype.itemsize * (m * k + k * n_per_device + m * n_per_device)
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)

    @functools.partial(jax.jit, static_argnames=["bn", "bk", "rhs_transpose"])
    def _all_gather_matmul_call(x, y, bn, bk, rhs_transpose):
        return pl.pallas_call(
            functools.partial(
                _all_gather_kernel,
                bn=bn,
                bk=bk,
                axis_name=axis_name,
                rhs_transpose=rhs_transpose,
            ),
            out_shape=out_shape,
            grid_spec=grid_spec,
            compiler_params=pltpu.CompilerParams(
                collective_id=collective_id,
                vmem_limit_bytes=estimated_vmem_bytes + 8 * 1024 * 1024,
            ),
            cost_estimate=cost_estimate,
            name=f"all_gather_matmul_kernel_bn_{bn}_bk_{bk}_rhs_transpose_{rhs_transpose}",
        )(x, y)[0]

    return _all_gather_matmul_call(x, y, bn, bk, rhs_transpose)
