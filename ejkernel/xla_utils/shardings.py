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
from jax import numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec


def get_corrected_named_sharding(
    shape: tuple[int, ...],
    partition_spec: PartitionSpec,
    mesh: Mesh,
) -> NamedSharding:
    """
    Calculates the corrected PartitionSpec based on shape and mesh, returns NamedSharding.

    This function takes an array shape and a desired PartitionSpec.
    It determines the effective PartitionSpec by correcting the input based on:
      - Axis names present in the current mesh.
      - Divisibility of array dimensions by the product of corresponding mesh axis sizes.

    It does NOT correct based on mesh axes having size 1, allowing such axes
    to persist in the spec if explicitly provided and divisibility holds.

    Args:
        shape: The shape of the target JAX array.
        partition_spec: The desired PartitionSpec.
        raise_mesh_error: If True, raises an error if no mesh is active.
                          If False, returns a replicated NamedSharding on an
                          empty mesh if no mesh is found.

    Returns:
        A NamedSharding object containing the current mesh and the corrected
        PartitionSpec.

    Raises:
        AssertionError: If no mesh is active and raise_mesh_error is True.
    """

    ndim = len(shape)
    original_spec = partition_spec

    if len(original_spec) == 0:
        return NamedSharding(mesh, PartitionSpec())

    spec_tuple = tuple(original_spec)
    if len(spec_tuple) < ndim:
        spec_tuple += (None,) * (ndim - len(spec_tuple))
    elif len(spec_tuple) > ndim:
        spec_tuple = spec_tuple[:ndim]

    corrected_spec_list = list(spec_tuple)
    mesh_axis_names = set(mesh.axis_names)

    for i, axis_spec in enumerate(spec_tuple):
        if axis_spec is None:
            continue

        current_axis_names = []
        if isinstance(axis_spec, str):
            current_axis_names.append(axis_spec)
        elif isinstance(axis_spec, tuple):
            current_axis_names.extend(axis_spec)
        else:
            corrected_spec_list[i] = None
            continue

        valid_axis = True
        total_mesh_size_for_dim = 1
        for axis_name in current_axis_names:
            if axis_name not in mesh_axis_names:
                valid_axis = False
                break
            total_mesh_size_for_dim *= mesh.shape[axis_name]
        if not valid_axis:
            corrected_spec_list[i] = None
            continue

        if total_mesh_size_for_dim > 0 and shape[i] % total_mesh_size_for_dim != 0:
            corrected_spec_list[i] = None
            continue
        elif total_mesh_size_for_dim == 0:
            corrected_spec_list[i] = None
            continue

    corrected_spec = PartitionSpec(*corrected_spec_list)
    if not any(axis is not None for axis in corrected_spec):
        final_spec_to_apply = PartitionSpec()
    else:
        final_spec_to_apply = corrected_spec

    return NamedSharding(mesh, final_spec_to_apply)


@partial(jax.jit, static_argnames=("cp_size", "seq_dim", "to_contiguous"))
def reorder_sequence(tensor, cp_size: int, seq_dim: int = 1, to_contiguous: bool = False):
    if tensor is None:
        return tensor

    seq_len = tensor.shape[seq_dim]
    group_size = seq_len // (2 * cp_size)

    if cp_size % 2 != 0:
        raise ValueError(f"{cp_size=} must be a multiple of 2.")

    if seq_len % (cp_size * 2) != 0:
        raise ValueError(f"{tensor.shape=} is not a multiple of {cp_size*2=}")

    ori_tensor_shape = tensor.shape
    reshaped = tensor.reshape(
        *ori_tensor_shape[:seq_dim],
        2 * cp_size,
        group_size,
        *ori_tensor_shape[seq_dim + 1 :],
    )

    if not to_contiguous:
        first_half = jnp.arange(cp_size)
        second_half = jnp.arange(2 * cp_size - 1, cp_size - 1, -1)
        src_indices = jnp.stack([first_half, second_half], axis=1).reshape(-1)

    else:
        half = cp_size // 2
        first_pair = [4 * r for r in range(half)]
        second_pair = [4 * r + 2 for r in range(half)]
        third_pair = [2 * cp_size - 1 - 4 * r for r in range(half)]
        fourth_pair = [i - 2 for i in third_pair]
        first_block = first_pair + third_pair
        second_block = second_pair + fourth_pair
        src_indices = jnp.stack([jnp.array(first_block), jnp.array(second_block)], axis=1).reshape(-1)

    reordered = jnp.take(reshaped, src_indices, axis=seq_dim)

    return reordered.reshape(ori_tensor_shape)
