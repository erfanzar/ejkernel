# Copyright 2023 The EASYDEL/EJGPU(EasyDeLJaxGPUUtilities) Author @erfanzar (Erfan Zare Chavoshi).
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
from .decode_attention import decode_attention
from .flash_attn import flash_attention
from .flash_attn_varlen import attention_varlen_vanilla, flash_attn_varlen, flash_attn_varlen_decode
from .gla import recurrent_gla
from .lightning_attn import lightning_attn
from .mean_pooling import mean_pooling
from .native_spare_attention import apply_native_spare_attention, native_spare_attention
from .recurrent import recurrent

__all__ = (
    "apply_native_spare_attention",
    "attention_varlen_vanilla",
    "decode_attention",
    "flash_attention",
    "flash_attn_varlen",
    "flash_attn_varlen_decode",
    "lightning_attn",
    "mean_pooling",
    "native_spare_attention",
    "recurrent",
    "recurrent_gla",
)
