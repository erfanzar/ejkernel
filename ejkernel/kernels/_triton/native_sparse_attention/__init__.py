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


from ._interface import (
    _bwd_call as native_sparse_attention_gpu_bwd,
)
from ._interface import (
    _fwd_call as native_sparse_attention_gpu_fwd,
)
from ._interface import (
    apply_native_sparse_attention,
    native_sparse_attention,
)

__all__ = (
    "apply_native_sparse_attention",
    "native_sparse_attention",
    "native_sparse_attention_gpu_bwd",
    "native_sparse_attention_gpu_fwd",
)
