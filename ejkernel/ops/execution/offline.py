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


"""Offline autotuning for pre-compiled JAX computations.

This module provides utilities for autotuning kernel configurations
based on lowered (compiled but not yet executed) JAX computations.
It enables optimization of kernels based on the actual operations
that will be executed, rather than relying on heuristics.

Key Functions:
    autotune_lowered: Autotune all ejkernel operations found in a lowered computation

The offline tuning workflow:
    1. Lower a JAX function to get compiled representation
    2. Extract ejkernel operation labels from the HLO code
    3. Match labels to recorded invocations in the registry
    4. Run autotuning for each matched operation
    5. Store optimal configurations in cache

Example:
    >>> lowered = jax.jit(my_function).lower(example_args)
    >>> result = autotune_lowered(selector, lowered)
    >>> with result:  # Apply optimized configurations
    ...     output = jax.jit(my_function)(real_args)
"""

from __future__ import annotations

import jax

from ..config.selection import ConfigSelectorChain
from ..core import _get_platform_method
from ..registry import get_invocations
from ..utils.fingerprint import device_fingerprint, get_device_platform
from ..utils.meta import LABEL_RE, extract_labels_from_hlo_text
from .tuning import AutotuningResult, Entry


def _labels_to_invocations(lowered) -> list[tuple[str, str]]:
    """Extract operation ID and call key pairs from a lowered computation.

    Parses the HLO representation of a lowered JAX computation to find
    all embedded ejkernel operation labels and extracts their identifiers.

    Args:
        lowered: JAX lowered computation object

    Returns:
        List of (operation_id, call_key) tuples found in the computation
    """
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
    """Autotune all ejkernel operations found in a lowered JAX computation.

    Analyzes a lowered JAX computation to identify all ejkernel operations,
    then runs autotuning for each operation using recorded invocations from
    the global registry.

    Args:
        selector: ConfigSelectorChain with tuner and cache for optimization
        lowered: JAX lowered computation containing ejkernel operations

    Returns:
        AutotuningResult containing optimal configurations for all tuned operations.
        Can be used as a context manager to temporarily apply the configurations.

    Example:
        >>> lowered = jax.jit(my_model).lower(example_input)
        >>> result = autotune_lowered(selector, lowered)
        >>> with result:
        ...     # Runs with optimized configurations
        ...     output = jax.jit(my_model)(real_input)

    Note:
        Only operations that have been previously recorded in the invocation
        registry will be tuned. Use EJKERNEL_OPS_RECORD=1 environment variable
        to enable invocation recording during initial runs.
    """
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
