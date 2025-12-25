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

import importlib.util

import pytest


def test_import_works_without_triton_installed():
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("Triton is installed; this test only validates the optional-dependency path.")

    import ejkernel

    assert ejkernel.kernels.triton is None
    assert ejkernel.utils.triton is None


def test_triton_call_errors_cleanly_when_triton_missing():
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("Triton is installed; this test only validates the optional-dependency path.")

    import ejkernel

    with pytest.raises(ValueError, match=r"triton.*installed"):
        ejkernel.callib.triton_call((), kernel=None, out_shape=(), grid=1)

