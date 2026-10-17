"""Run model-only tests without importing dataset/evaluation dependencies.

The namespace packages below load the real model modules and their helpers;
only package __init__ imports of datasets and evaluation engines are skipped.
This does not test the training/evaluation CLI or replace a dataset evaluation.
"""
import sys
import types
import unittest
from pathlib import Path

import torch


def main():
    code_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(code_dir))
    for name in ("cls_libs", "det_libs"):
        package = types.ModuleType(name)
        package.__path__ = [str(code_dir / name)]
        sys.modules[name] = package
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.discover(str(code_dir / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
