# -*- coding: utf-8 -*-
"""The public code must import without the private-corpus dependency.

The fifth review found that ``experiment.py`` imported ``corpus`` at module
level, which imports ``jieba`` - so even the public regression tests failed in
an environment that (as the README says) only needs jieba for the private
corpus.  This check simulates such an environment.

    python check_public_imports.py
"""

from __future__ import annotations

import builtins
import importlib
import sys

MODULES = ["config", "metrics", "models", "baselines", "new_baselines",
           "experiment", "benchmark_public", "fair_readout",
           "occlusion_robustness", "run_fcntm_train", "rescore_clean_test",
           "run_public_missing_field", "compare_occlusion"]


def main() -> int:
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "jieba" or name.startswith("jieba."):
            raise ModuleNotFoundError("No module named 'jieba' (simulated)")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    try:
        for name in MODULES:
            importlib.import_module(name)
            print(f"  ok  {name}")
    except ModuleNotFoundError as exc:
        if "jieba" in str(exc):
            print(f"  FAIL  a public module needs jieba: {exc}")
            return 1
        raise
    finally:
        builtins.__import__ = real_import
    print("all public modules import without jieba")
    return 0


if __name__ == "__main__":
    sys.exit(main())
