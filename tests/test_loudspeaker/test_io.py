from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from loudspeaker.io import save_npz_bundle


def test_save_npz_bundle_requires_ts_and_states(tmp_path):
    output = tmp_path / "bundle.npz"
    with pytest.raises(ValueError):
        save_npz_bundle(output, states=np.zeros(2))
    with pytest.raises(ValueError):
        save_npz_bundle(output, ts=np.zeros(2))


def test_save_npz_bundle_converts_arrays_and_skips_none(tmp_path):
    output = tmp_path / "bundle.npz"
    ts = np.linspace(0.0, 1.0, 4)
    states = np.zeros((4, 2))
    saved = save_npz_bundle(
        output,
        ts=ts,
        states=states,
        optional=None,
        scalar=3.14,
    )
    assert saved == output
    assert Path(saved).exists()
    with np.load(saved) as data:
        assert set(data.files) == {"ts", "states", "scalar"}
        np.testing.assert_allclose(data["scalar"], 3.14)
