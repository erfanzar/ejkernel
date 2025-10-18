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

import jax

from ..config.selection import ConfigSelectorChain
from ..core import _get_platform_method
from ..registry import get_invocations
from ..utils.fingerprint import device_fingerprint, get_device_platform
from ..utils.meta import LABEL_RE, extract_labels_from_hlo_text
from .tuning import AutotuningResult, Entry


def _labels_to_invocations(lowered) -> list[tuple[str, str]]:
    try:
        hlo_text = lowered.compiler_ir(dialect="hlo").as_text()
    except Exception:
        hlo_text = str(lowered)
    labels = extract_labels_from_hlo_text(hlo_text)
    pairs = []
    for lab in labels:
        m = LABEL_RE.search(lab)
        if not m:
            continue
        pairs.append((m.group("op"), m.group("key")))
    return pairs


def autotune_lowered(selector: ConfigSelectorChain, lowered) -> AutotuningResult:
    dev = device_fingerprint()
    invs = get_invocations(dev)
    targets = _labels_to_invocations(lowered)
    entries: list[Entry] = []
    platform = get_device_platform()

    for op_id_v, call_key in targets:
        if op_id_v not in invs or call_key not in invs[op_id_v]:
            continue
        kernel, args, kwargs = invs[op_id_v][call_key]
        inv_args, inv_kwargs = kernel.prepare(*args, **kwargs)
        static_fun_kwargs = {k: v for k, v in inv_kwargs.items() if callable(v)}
        dyn_kwargs = {k: v for k, v in inv_kwargs.items() if not callable(v)}
        tmp_inv = type(
            "Tmp",
            (),
            dict(op_id=kernel.op_id, args=inv_args, kwargs=dyn_kwargs, batch_axes=None, override_cfg=None, stamp=False),
        )()

        cand_method = _get_platform_method(kernel, "candidate_cfgs", platform, None) or kernel.candidate_cfgs
        candidates = tuple(cand_method(tmp_inv))
        run_method = _get_platform_method(kernel, "run", platform, None) or kernel.run

        def mk(c, _run=run_method, _static=static_fun_kwargs):
            def f(*a, **k):
                return _run(*a, cfg=c, **(k | _static))  # noqa: B023

            return f

        with jax.core.eval_context():
            best = selector.tuner.autotune(mk, inv_args, dyn_kwargs, candidates)
        selector.cache.put(dev, op_id_v, call_key, best)
        if selector.persistent is not None and selector.persist_autotune:
            selector.persistent.put(dev, op_id_v, call_key, best)
        entries.append(Entry(op_id_v, call_key, best))

    return AutotuningResult(dev, tuple(entries))
