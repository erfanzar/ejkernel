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

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("EJKERNEL_AUTOTUNE_POLICY", "heuristics")

_TRITON_ONLY_TEST_BASENAMES = {
    # Cross-backend comparisons that require Triton.
    "test_flash_attention_xla_triton.py",
    "test_native_sparse_attention_xla_triton.py",
    "test_ragged_page_attention_v3_reference.py",
}


def _has_triton() -> bool:
    return importlib.util.find_spec("triton") is not None


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool:
    if _has_triton():
        return False

    if collection_path.name in _TRITON_ONLY_TEST_BASENAMES:
        return True

    # Also ignore the entire Triton test directory when Triton isn't installed.
    parts = set(collection_path.parts)
    return "test" in parts and "kernels" in parts and "_triton" in parts
