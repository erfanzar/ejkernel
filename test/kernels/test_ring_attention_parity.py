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


"""Test feature parity between XLA and Pallas ring attention implementations."""

import inspect

import pytest

from ejkernel.kernels._pallas.tpu.ring_attention import _interface as pallas_interface
from ejkernel.kernels._xla.ring_attention import _interface as xla_interface


class TestFeatureParity:
    """Test that XLA and Pallas implementations have feature parity."""

    def test_function_exists_in_both(self):
        """Test that ring_attention function exists in both implementations."""
        assert hasattr(xla_interface, "ring_attention")
        assert hasattr(pallas_interface, "ring_attention")

    def test_default_values_consistency(self):
        """Test that default values are consistent where applicable."""
        xla_sig = inspect.signature(xla_interface.ring_attention)
        pallas_sig = inspect.signature(pallas_interface.ring_attention)

        assert xla_sig.parameters["float32_logits"].default
        assert pallas_sig.parameters["float32_logits"].default

        assert xla_sig.parameters["deterministic"].default
        assert pallas_sig.parameters["deterministic"].default

        assert xla_sig.parameters["pdrop"].default == 0.0
        assert pallas_sig.parameters["pdrop"].default == 0.0

        assert xla_sig.parameters["attention_sink_size"].default == 0
        assert pallas_sig.parameters["attention_sink_size"].default == 0

        assert xla_sig.parameters["prevent_cse"].default
        assert pallas_sig.parameters["prevent_cse"].default


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
