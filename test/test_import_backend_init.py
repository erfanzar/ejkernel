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

import sys

# When executed as a script (python test/test_*.py), Python puts the test
# directory at sys.path[0], which can shadow stdlib modules (e.g. test/types).
# Drop that entry and prepend repo root to keep stdlib resolution correct.
if sys.path:
    p0 = sys.path[0].replace("\\", "/")
    if p0.endswith("/test"):
        sys.path.pop(0)

file_path = __file__.replace("\\", "/")
if "/test/" in file_path:
    repo_root = file_path.rsplit("/test/", 1)[0]
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)

import importlib.util  # noqa
import os  # noqa
import subprocess  # noqa
import textwrap  # noqa

import pytest  # noqa


def test_importing_ejkernel_tree_does_not_trigger_jax_backend_init():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed in this environment.")

    script = textwrap.dedent(
        """
        import importlib
        import pkgutil

        import jax
        import jax._src.xla_bridge as xb

        def _backend_init_guard(*args, **kwargs):
            raise RuntimeError("backend-init-guard-triggered")

        xb.get_backend = _backend_init_guard
        xb.backends = _backend_init_guard
        jax.default_backend = _backend_init_guard
        jax.devices = _backend_init_guard

        import ejkernel

        # Top-level package imports/exports.
        importlib.import_module("ejkernel.errors")
        importlib.import_module("ejkernel.kernels")
        importlib.import_module("ejkernel.modules")
        importlib.import_module("ejkernel.quantization")
        importlib.import_module("ejkernel.ops")
        importlib.import_module("ejkernel.types")
        importlib.import_module("ejkernel.utils")
        importlib.import_module("ejkernel.xla_utils")
        from ejkernel import *  # noqa: F403,F401

        optional_missing_roots = {"triton", "cutlass", "cuda"}
        failures = []
        for mod in pkgutil.walk_packages(ejkernel.__path__, ejkernel.__name__ + "."):
            name = mod.name
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as err:
                missing = (err.name or "").split(".", 1)[0]
                if missing in optional_missing_roots:
                    continue
                failures.append((name, f"ModuleNotFoundError: {err!r}"))
            except Exception as err:
                if "backend-init-guard-triggered" in str(err):
                    raise
                failures.append((name, repr(err)))

        if failures:
            raise RuntimeError(f"Unexpected import failures: {failures[:20]}")
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    assert proc.returncode == 0, f"stdout:\\n{proc.stdout}\\nstderr:\\n{proc.stderr}"


if __name__ == "__main__":
    pytest.main([__file__])
