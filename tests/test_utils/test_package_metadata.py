from __future__ import annotations

import importlib


def test_root_package_exposes_metadata():
    pkg = importlib.import_module("src")
    assert pkg.__version__ == "0.1.0"
    assert "Johannes" in pkg.__author__
