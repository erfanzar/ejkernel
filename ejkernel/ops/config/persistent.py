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


"""Persistent disk-based configuration cache for kernel optimization.

This module provides persistent storage for optimal kernel configurations,
allowing optimization results to be preserved across program runs. The cache
uses JSON files for storage with automatic serialization/deserialization.

Key Features:
    - Atomic file operations to prevent corruption
    - Automatic serialization for dataclasses and Pydantic models
    - Custom loader/dumper support for complex types
    - Device and operation-specific configuration storage
    - Thread-safe atomic updates using temporary files

The persistent cache complements the in-memory cache by providing:
    - Long-term storage of optimization results
    - Sharing of configurations across program runs
    - Reduced autotuning overhead for repeated operations
    - Backup storage for critical optimization data

Example Usage:
    >>>
    >>> cache = PersistentCache('/path/to/cache.json')
    >>> cache.put('gpu:0', 'matmul_v1', 'key123', my_config)
    >>> config = cache.get('gpu:0', 'matmul_v1', 'key123')
    >>>
    >>>
    >>> def custom_dumper(cfg): return cfg.to_dict()
    >>> def custom_loader(data): return MyConfig.from_dict(data)
    >>> cache = PersistentCache('/path/to/cache.json', custom_loader, custom_dumper)
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from argparse import Namespace
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any, Generic, TypeVar

Cfg = TypeVar("Cfg")


class PersistentCache(Generic[Cfg]):
    """Persistent disk-based cache for kernel configurations.

    Provides thread-safe storage and retrieval of optimization configurations
    using JSON files. Supports automatic serialization for common types and
    custom serialization via loader/dumper functions.

    The cache uses atomic file operations to ensure data consistency and
    prevent corruption during concurrent access.

    Type Parameters:
        Cfg: Configuration type to be cached

    Attributes:
        path: File path for the JSON cache
        loader: Optional function to deserialize configurations
        dumper: Optional function to serialize configurations
    """

    def __init__(
        self,
        opname: str,
        path: str | None = None,
        loader: Callable[[Any], Cfg] | None = None,
        dumper: Callable[[Cfg], Any] | None = None,
        cfg_type: type[Cfg] | None = None,
    ):
        """Initialize persistent cache with file path and optional serializers.

        Args:
            path: File path for JSON storage
            loader: Optional function to deserialize stored data to Cfg type
            dumper: Optional function to serialize Cfg type for storage

        Note:
            If loader/dumper are not provided, automatic serialization is attempted
            for dataclasses and Pydantic models. Raw values are stored as-is.
        """
        if path is None:
            path = str(pathlib.Path().home().expanduser() / "ejkernel-presistent-cache" / f"{opname}.json")

        pathlib.Path(os.path.dirname(os.path.abspath(path)) or ".").mkdir(exist_ok=True, parents=True)

        self.path = path
        self.loader = loader
        self.dumper = dumper
        self.cfg_type = cfg_type
        try:
            with open(path, "r") as f:
                self._data = json.load(f)
        except FileNotFoundError:
            self._data = {}

    def _key(self, device: str, op_id: str, call_key: str) -> str:
        """Generate internal storage key from cache coordinates.

        Args:
            device: Device fingerprint (e.g., 'gpu:0', 'tpu:v4')
            op_id: Operation identifier with version (e.g., 'matmul@v1')
            call_key: Function call signature hash

        Returns:
            String key for internal storage dictionary
        """
        return "|".join((device, op_id, call_key))

    def get(self, device: str, op_id: str, call_key: str) -> Cfg | None:
        """Retrieve cached configuration for the given coordinates.

        Args:
            device: Device fingerprint for the configuration
            op_id: Operation identifier
            call_key: Function call signature hash

        Returns:
            Cached configuration if found, None otherwise

        Note:
            If a custom loader was provided, it will be used to deserialize
            the stored data. Otherwise, the raw JSON data is returned.
        """
        raw = self._data.get(self._key(device, op_id, call_key))
        out = None if raw is None else (self.loader(raw) if self.loader else raw)
        if out is not None and isinstance(out, dict):
            if self.cfg_type is None:
                out = Namespace(**out)
            else:
                out = self.cfg_type(**out)
        return out

    def put(self, device: str, op_id: str, call_key: str, cfg: Cfg):
        """Store configuration in the cache with atomic file update.

        Args:
            device: Device fingerprint for the configuration
            op_id: Operation identifier
            call_key: Function call signature hash
            cfg: Configuration to store

        Note:
            Uses atomic file operations (write to temporary file, then replace)
            to ensure data consistency. If a custom dumper was provided, it will
            be used for serialization. Otherwise, automatic serialization is
            attempted for dataclasses and Pydantic models.

        Serialization Priority:
            1. Custom dumper function (if provided)
            2. Dataclass.asdict() for dataclass objects
            3. model_dump() for Pydantic v2 models
            4. Raw value (must be JSON-serializable)
        """
        if self.dumper is not None:
            val = self.dumper(cfg)
        else:
            if is_dataclass(cfg):
                val = asdict(cfg)
            elif hasattr(cfg, "model_dump"):
                val = cfg.model_dump()
            else:
                val = cfg

        self._data[self._key(device, op_id, call_key)] = val

        dir_name = os.path.dirname(os.path.abspath(self.path)) or "."
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
            json.dump(self._data, tmp)
            tmp_path = tmp.name
        os.replace(tmp_path, self.path)
