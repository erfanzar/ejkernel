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


"""Metadata extraction and label processing for compiled JAX programs.

This module provides utilities for working with compilation metadata and
labels embedded in JAX compiled programs. It enables extraction of operation
identifiers and configuration mappings from compiled HLO code.

Key Functions:
    label: Generate standardized labels for operations
    extract_labels_from_hlo_text: Find all ejkernel labels in HLO text
    find_labels_in_lowered: Extract labels from lowered JAX computations
    labels_to_configs: Map found labels back to their configurations

Label Format:
    Labels follow the pattern: 'ejkernel_ops#operation@version:hash'
    Example: 'ejkernel_ops#matmul@v1:1a2b3c4d5e6f7g8h'

These utilities enable:
    - Tracking which operations were compiled with which configurations
    - Post-compilation analysis of optimization choices
    - Debugging and profiling of specific operation instances
    - Configuration recovery from compiled programs

Example Usage:
    >>>
    >>> op_label = label('matmul@v1', '1a2b3c4d5e6f7g8h')
    >>> print(op_label)
    >>>
    >>>
    >>> lowered = jax.jit(my_function).lower(args)
    >>> labels = find_labels_in_lowered(lowered)
    >>>
    >>>
    >>> configs = labels_to_configs(lowered, selector)
"""

from __future__ import annotations

import os
import re

from .fingerprint import device_fingerprint

LABEL_PREFIXES = ("ejkernel_ops#", "ejkernel_ops#")
LABEL_RE = re.compile(r"(?:ejkernel_ops|ejkernel_ops)#(?P<op>[^:]+@v[0-9A-Za-z_.-]+):(?P<key>[0-9a-f]{16})")


def labels_to_configs(lowered, selector):
    """Extract labels from lowered computation and map them to configurations.

    Finds all ejkernel operation labels in the compiled code and retrieves
    their corresponding configurations from the cache system.

    Args:
        lowered: JAX lowered computation object
        selector: ConfigSelectorChain for cache lookups

    Returns:
        List of (label, config) tuples for all found operations

    Note:
        Configurations are looked up first in memory cache, then in
        persistent cache if available. Operations without cached
        configurations will have None as their config value.
    """
    dev = device_fingerprint()
    labels = find_labels_in_lowered(lowered)
    out = []
    for lab in labels:
        m = LABEL_RE.search(lab)
        if not m:
            continue
        op_id, call_key = m.group("op"), m.group("key")
        cfg = selector.cache.get(dev, op_id, call_key)
        if cfg is None and selector.persistent is not None:
            cfg = selector.persistent.get(dev, op_id, call_key)
        out.append((lab, cfg))
    return out


def label(op_id: str, call_hash: str, prefix: str | None = None) -> str:
    """Generate a standardized label for an operation.

    Creates a label string that uniquely identifies an operation instance
    for embedding in compiled code and later retrieval.

    Args:
        op_id: Operation identifier with version (e.g., 'matmul@v1')
        call_hash: 16-character hash of the call signature

    Returns:
        Formatted label string following ejkernel convention

    Examples:
        >>> label('matmul@v1', '1a2b3c4d5e6f7g8h')
        'ejkernel_ops#matmul@v1:1a2b3c4d5e6f7g8h'
    """
    chosen_prefix = prefix or os.getenv("EJKERNEL_OPS_PREFIX") or os.getenv("EJKERNEL_OPS_PREFIX") or "ejkernel_ops#"
    return f"{chosen_prefix}{op_id}:{call_hash}"


def extract_labels_from_hlo_text(hlo_text: str) -> list[str]:
    """Find all ejkernel operation labels in HLO text.

    Searches through HLO (High Level Operations) text to find all
    embedded ejkernel operation labels using regex pattern matching.

    Args:
        hlo_text: String containing HLO representation of compiled code

    Returns:
        List of found label strings

    Note:
        The regex pattern matches the standard ejkernel label format:
        'ejkernel_ops#' + operation_name + ':' + 16-char hex hash
    """
    pat = re.compile(r"(?:ejkernel_ops|ejkernel_ops)#[A-Za-z0-9_.@-]+:[0-9a-f]{16}")
    return pat.findall(hlo_text)


def find_labels_in_lowered(lowered) -> list[str]:
    """Extract operation labels from a JAX lowered computation.

    Converts the lowered computation to HLO text representation and
    extracts all embedded ejkernel operation labels.

    Args:
        lowered: JAX lowered computation object

    Returns:
        List of operation labels found in the compiled code

    Note:
        First attempts to get HLO representation, falls back to
        string representation if HLO extraction fails.
    """
    try:
        t = lowered.compiler_ir(dialect="hlo").as_text()
    except Exception:
        t = str(lowered)
    return extract_labels_from_hlo_text(t)
