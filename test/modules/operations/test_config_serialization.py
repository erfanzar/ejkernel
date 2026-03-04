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

"""Tests for config serialization (to_dict, from_dict, to_json, from_json)."""

from __future__ import annotations

import json

from ejkernel.modules.operations.configs import (
    BaseOperationConfig,
    FlashAttentionConfig,
    GatedDeltaRuleConfig,
    KernelDeltaAttentionConfig,
    MultiLatentRaggedPageAttentionConfig,
)
from ejkernel.ops import FwdParams


def test_base_config_to_dict():
    cfg = BaseOperationConfig(platform="xla", backend="tpu")
    d = cfg.to_dict()
    assert d == {"platform": "xla", "backend": "tpu"}


def test_base_config_roundtrip_dict():
    cfg = BaseOperationConfig(platform="triton", backend="gpu")
    cfg2 = BaseOperationConfig.from_dict(cfg.to_dict())
    assert cfg2.platform == cfg.platform
    assert cfg2.backend == cfg.backend


def test_base_config_roundtrip_json():
    cfg = BaseOperationConfig(platform="pallas", backend="tpu")
    j = cfg.to_json()
    parsed = json.loads(j)
    assert parsed["platform"] == "pallas"
    cfg2 = BaseOperationConfig.from_json(j)
    assert cfg2.platform == cfg.platform
    assert cfg2.backend == cfg.backend


def test_gdr_config_roundtrip():
    cfg = GatedDeltaRuleConfig(chunk_size=128, platform="xla")
    d = cfg.to_dict()
    assert d["chunk_size"] == 128
    cfg2 = GatedDeltaRuleConfig.from_dict(d)
    assert cfg2.chunk_size == 128

    cfg3 = GatedDeltaRuleConfig.from_json(cfg.to_json())
    assert cfg3.chunk_size == 128


def test_kda_config_roundtrip():
    cfg = KernelDeltaAttentionConfig(platform="xla")
    d = cfg.to_dict()
    assert d["platform"] == "xla"
    cfg2 = KernelDeltaAttentionConfig.from_dict(d)
    assert cfg2.platform == "xla"


def test_mlrpa_config_roundtrip():
    cfg = MultiLatentRaggedPageAttentionConfig(
        chunk_prefill_size=512,
        num_kv_pages_per_block=8,
        num_queries_per_block=32,
        platform="pallas",
    )
    d = cfg.to_dict()
    assert d["chunk_prefill_size"] == 512
    assert d["num_kv_pages_per_block"] == 8

    cfg2 = MultiLatentRaggedPageAttentionConfig.from_dict(d)
    assert cfg2.chunk_prefill_size == 512
    assert cfg2.num_queries_per_block == 32


def test_flash_attention_config_with_fwd_params():
    cfg = FlashAttentionConfig(
        fwd_params=FwdParams(q_blocksize=128, kv_blocksize=64),
        platform="triton",
    )
    d = cfg.to_dict()
    assert d["fwd_params"]["q_blocksize"] == 128
    assert d["fwd_params"]["kv_blocksize"] == 64

    # from_dict needs FwdParams reconstruction
    cfg2 = FlashAttentionConfig.from_dict(d)
    assert cfg2.fwd_params.q_blocksize == 128
    assert cfg2.fwd_params.kv_blocksize == 64


def test_config_hash_stability():
    cfg1 = GatedDeltaRuleConfig(chunk_size=64)
    cfg2 = GatedDeltaRuleConfig(chunk_size=64)
    cfg3 = GatedDeltaRuleConfig(chunk_size=128)
    assert hash(cfg1) == hash(cfg2)
    assert hash(cfg1) != hash(cfg3)
