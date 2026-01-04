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


"""Data carrier classes for kernel configuration parameters.

This module provides dataclasses that encapsulate forward and backward pass
parameters for various kernel operations, particularly attention mechanisms.
These parameter carriers enable consistent configuration across different
kernel implementations and facilitate autotuning by providing hashable
parameter sets.

Classes:
    FwdParams: Forward pass parameters for kernel configuration
    BwdParams: Backward pass parameters for kernel configuration

The parameter carriers support:
    - Block size configuration for tiling strategies
    - Warp and pipeline stage configuration for GPU kernels
    - Consistent hashing for configuration caching
    - Optional parameters that can be None for auto-selection
"""

import hashlib
from dataclasses import dataclass


def get_safe_hash_int(text, algorithm="md5"):
    """Generate an integer hash of text using the specified algorithm.

    Converts any input to a string and computes a hash using the specified
    algorithm from the hashlib module. The hash digest is then converted
    to a big-endian integer.

    Args:
        text: Input to hash (will be converted to string)
        algorithm: Hash algorithm name from hashlib (default: "md5")

    Returns:
        Integer representation of the hash digest

    Raises:
        ValueError: If the specified algorithm is not supported by hashlib
        Exception: If any other error occurs during hash generation

    Example:
        >>> get_safe_hash_int("test_string")
        123456789012345678901234567890
        >>> get_safe_hash_int("test", algorithm="sha256")
        987654321098765432109876543210
    """
    try:
        text_str = str(text)
        hash_object = getattr(hashlib, algorithm)(text_str.encode())
        return int.from_bytes(hash_object.digest(), byteorder="big")
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e
    except Exception as e:
        raise Exception(f"Error generating hash: {e!s}") from e


def hash_fn(self) -> int:
    """Generate a hash for an object based on its dictionary values.

    Creates a deterministic hash by concatenating string representations
    of all hashable-type values in the object's __dict__ attribute.
    Only includes values that are float, int, bool, dict, or list types.

    This function is designed to be used as a __hash__ method replacement
    for dataclasses that need consistent hashing for configuration caching.

    Args:
        self: Object instance with __dict__ attribute containing configuration

    Returns:
        Integer hash value derived from the object's attribute values

    Note:
        Only primitive types (float, int, bool) and collections (dict, list)
        are included in the hash. Other types (e.g., None, str, objects) are
        excluded to ensure stable hashing across different configurations.
    """
    shu = "".join(str(cu) for cu in self.__dict__.values() if isinstance(cu, float | int | bool | dict | list))
    return get_safe_hash_int(shu)


@dataclass
class FwdParams:
    """Forward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for forward pass kernels,
    particularly for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention
        kv_blocksize: Block size for key/value dimension in attention
        blocksize_heads: Block size for head dimension in multi-head attention
        blocksize_keys: Block size for key sequence length
        num_key_splits: Number of splits for key computation
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        All parameters are optional (None) to allow automatic selection
        during kernel execution or autotuning.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None

    blocksize_heads: int | None = None
    blocksize_keys: int | None = None
    num_key_splits: int | None = None

    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn


@dataclass
class BwdParams:
    """Backward pass parameters for kernel configuration.

    Encapsulates block sizes and execution parameters for backward pass kernels,
    used in gradient computation for attention and matrix multiplication operations.

    Attributes:
        blocksize_m: Block size for M dimension (rows of output matrix)
        blocksize_k: Block size for K dimension (reduction dimension)
        blocksize_n: Block size for N dimension (columns of output matrix)
        q_blocksize: Block size for query dimension in attention gradients
        kv_blocksize: Block size for key/value dimension in attention gradients
        num_warps: Number of GPU warps for thread block execution
        num_stages: Number of pipeline stages for memory optimization

    Note:
        Parameters are typically smaller than forward pass due to different
        memory access patterns in gradient computation.
    """

    blocksize_m: int | None = None
    blocksize_k: int | None = None
    blocksize_n: int | None = None
    q_blocksize: int | None = None
    kv_blocksize: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None

    __hash__ = hash_fn
