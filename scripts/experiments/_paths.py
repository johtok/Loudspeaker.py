from __future__ import annotations

import sys
from pathlib import Path

REPO_NAME = "Loudspeaker.py"


def _locate_repo_root(marker: str = REPO_NAME) -> Path:
    """Walk upward from this file to locate the repository root directory."""

    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.name == marker:
            return candidate
    raise RuntimeError(
        f"Unable to determine repository root named '{marker}' starting from {here}"
    )


REPO_ROOT = _locate_repo_root()
SRC_DIR = REPO_ROOT / "src"


def script_dir(file: str | Path) -> Path:
    """Return the directory containing the provided script file."""

    return Path(file).resolve().parent


def ensure_sys_path(*additional: Path | str) -> None:
    """Insert required directories onto sys.path for experiment scripts."""

    entries = (REPO_ROOT, SRC_DIR, *additional)
    for entry in entries:
        entry_str = str(entry)
        if entry_str not in sys.path:
            sys.path.append(entry_str)


__all__ = [
    "REPO_NAME",
    "REPO_ROOT",
    "SRC_DIR",
    "ensure_sys_path",
    "script_dir",
]
