#!/usr/bin/env python3
"""Run every experiment script sequentially."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT_DIR / "scripts" / "experiments"


def _has_main(path: Path) -> bool:
    try:
        source = path.read_text()
    except OSError as exc:
        print(f"[skip] Could not read {path}: {exc}", file=sys.stderr)
        return False
    return "def main" in source


def _experiment_scripts() -> Iterable[Path]:
    if not EXPERIMENTS_DIR.exists():
        return []
    candidates = sorted(EXPERIMENTS_DIR.rglob("exp_*.py"))
    return [
        path
        for path in candidates
        if not path.name.endswith("_wip.py") and _has_main(path)
    ]


def _run_script(script: Path) -> None:
    print(f"\n=== Running {script.relative_to(ROOT_DIR)} ===")
    result = subprocess.run([sys.executable, str(script)], cwd=ROOT_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Experiment failed: {script}")


def main() -> None:
    scripts = list(_experiment_scripts())
    if not scripts:
        print("No experiment scripts found.")
        return

    for script in scripts:
        _run_script(script)

    print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
