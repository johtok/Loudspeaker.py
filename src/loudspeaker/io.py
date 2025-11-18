from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_npz_bundle(path: str | Path, **arrays: Any) -> Path:
    """Persist a collection of arrays (NumPy/JAX) to a .npz archive.

    Parameters
    ----------
    path:
        Output file path.
    arrays:
        Mapping from name to array-like objects. Entries with ``None`` values are skipped.
        Entries are converted via ``np.asarray`` so JAX DeviceArrays are supported.
        At minimum, ``ts`` and ``states`` should be provided for timeseries dumps.
    """

    if "ts" not in arrays or "states" not in arrays:
        raise ValueError("save_npz_bundle requires at least 'ts' and 'states' entries.")

    payload: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if value is None:
            continue
        payload[name] = np.asarray(value)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **payload)
    return output
