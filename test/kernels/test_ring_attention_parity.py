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

    def test_parameter_parity(self):
        """Test that both implementations support the same parameters."""
        xla_sig = inspect.signature(xla_interface.ring_attention)
        pallas_sig = inspect.signature(pallas_interface.ring_attention)

        xla_params = set(xla_sig.parameters.keys())
        pallas_params = set(pallas_sig.parameters.keys())

        # Core parameters that should exist in both

        # Check XLA has all core features (with name variations)
        xla_core = {
            "query",
            "key",
            "value",
            "bias",
            "q_segment_ids",
            "kv_segment_ids",
            "softmax_aux",
            "axis_name",
            "float32_logits",
            "query_chunk_size",
            "key_chunk_size",
            "causal_block_size",
            "sliding_window",
            "logit_soft_cap",
            "attention_sink_size",
            "deterministic",
            "dropout_rng",
            "pdrop",
            "policy",
            "prevent_cse",
        }

        # Check that XLA has all required parameters
        missing_in_xla = xla_core - xla_params
        assert not missing_in_xla, f"XLA missing parameters: {missing_in_xla}"

        # Pallas uses slightly different names but has equivalent functionality
        pallas_core = {
            "q",
            "k",
            "v",
            "bias",
            "q_segment_ids",
            "kv_segment_ids",
            "softmax_aux",
            "axis_name",
            "float32_logits",
            "query_chunk_size",
            "key_chunk_size",
            "causal_block_size",
            "sliding_window",
            "logit_soft_cap",
            "attention_sink_size",
            "deterministic",
            "dropout_rng",
            "pdrop",
            "policy",
            "prevent_cse",
        }

        # Check that Pallas has all required parameters
        missing_in_pallas = pallas_core - pallas_params
        assert not missing_in_pallas, f"Pallas missing parameters: {missing_in_pallas}"

    def test_feature_support(self):
        """Test that both implementations advertise the same features."""
        # Features that should be supported by both

        # Both implementations should have these features
        xla_sig = inspect.signature(xla_interface.ring_attention)
        pallas_sig = inspect.signature(pallas_interface.ring_attention)

        # Check sliding window
        assert "sliding_window" in xla_sig.parameters
        assert "sliding_window" in pallas_sig.parameters

        # Check logit soft cap
        assert "logit_soft_cap" in xla_sig.parameters
        assert "logit_soft_cap" in pallas_sig.parameters

        # Check attention sink
        assert "attention_sink_size" in xla_sig.parameters
        assert "attention_sink_size" in pallas_sig.parameters

        # Check separate segment IDs
        assert "q_segment_ids" in xla_sig.parameters
        assert "kv_segment_ids" in xla_sig.parameters
        assert "q_segment_ids" in pallas_sig.parameters
        assert "kv_segment_ids" in pallas_sig.parameters

        # Check dropout
        assert "deterministic" in xla_sig.parameters
        assert "dropout_rng" in xla_sig.parameters
        assert "pdrop" in xla_sig.parameters
        assert "deterministic" in pallas_sig.parameters
        assert "dropout_rng" in pallas_sig.parameters
        assert "pdrop" in pallas_sig.parameters

        # Check checkpoint policy
        assert "policy" in xla_sig.parameters
        assert "policy" in pallas_sig.parameters

        # Check prevent CSE
        assert "prevent_cse" in xla_sig.parameters
        assert "prevent_cse" in pallas_sig.parameters

    def test_default_values_consistency(self):
        """Test that default values are consistent where applicable."""
        xla_sig = inspect.signature(xla_interface.ring_attention)
        pallas_sig = inspect.signature(pallas_interface.ring_attention)

        # Check some key defaults
        # float32_logits should default to True in both
        assert xla_sig.parameters["float32_logits"].default
        assert pallas_sig.parameters["float32_logits"].default

        # deterministic should default to True in both
        assert xla_sig.parameters["deterministic"].default
        assert pallas_sig.parameters["deterministic"].default

        # pdrop should default to 0.0 in both
        assert xla_sig.parameters["pdrop"].default == 0.0
        assert pallas_sig.parameters["pdrop"].default == 0.0

        # attention_sink_size should default to 0 in both
        assert xla_sig.parameters["attention_sink_size"].default == 0
        assert pallas_sig.parameters["attention_sink_size"].default == 0

        # prevent_cse should default to True in both
        assert xla_sig.parameters["prevent_cse"].default
        assert pallas_sig.parameters["prevent_cse"].default


class TestDocumentation:
    """Test that both implementations have proper documentation."""

    def test_xla_has_docstring(self):
        """Test that XLA implementation has documentation."""
        assert xla_interface.ring_attention.__doc__ is not None
        assert len(xla_interface.ring_attention.__doc__) > 100

    def test_pallas_has_docstring(self):
        """Test that Pallas implementation has documentation."""
        assert pallas_interface.ring_attention.__doc__ is not None
        assert len(pallas_interface.ring_attention.__doc__) > 100

    def test_docstrings_mention_key_features(self):
        """Test that docstrings mention the key features."""
        xla_doc = xla_interface.ring_attention.__doc__.lower()
        pallas_doc = pallas_interface.ring_attention.__doc__.lower()

        key_features = [
            "sliding window",
            "attention sink",
            "soft cap",
            "segment",
            "dropout",
        ]

        for feature in key_features:
            # At least one implementation should mention it
            assert feature in xla_doc or feature in pallas_doc, f"Neither docstring mentions '{feature}'"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
