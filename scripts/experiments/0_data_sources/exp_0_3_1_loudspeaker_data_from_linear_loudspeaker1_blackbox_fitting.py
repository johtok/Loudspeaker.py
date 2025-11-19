#!/usr/bin/env python3
"""Dataset export for linear loudspeaker blackbox fitting (taxonomy 0.3.1)."""

# %%
import csv
import sys
from pathlib import Path

import jax.random as jr
import numpy as np

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.append(str(_EXPERIMENTS_ROOT))

if __package__ in (None, ""):
    from _paths import REPO_ROOT, ensure_sys_path, script_dir
else:
    from ._paths import REPO_ROOT, ensure_sys_path, script_dir

SCRIPT_DIR = script_dir(__file__)
ensure_sys_path(SCRIPT_DIR)
OUT_DIR = (
    REPO_ROOT
    / "out"
    / "0_data_sources"
    / "exp_0_3_1_loudspeaker_data_from_linear_loudspeaker1_blackbox_fitting"
)

from loudspeaker import (
    LabelSpec,
    normalized_labels,
    plot_timeseries_bundle,
    save_figure,
    save_npz_bundle,
)
from loudspeaker.data import build_loudspeaker_dataset
from loudspeaker.loudspeaker_sim import LoudspeakerConfig


def _write_summary(forcing: np.ndarray, states: np.ndarray, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    stats = {
        "forcing_mean": float(np.mean(forcing)),
        "forcing_std": float(np.std(forcing)),
        "displacement_std": float(np.std(states[..., 0])),
        "velocity_std": float(np.std(states[..., 1])),
        "current_std": float(np.std(states[..., 2])),
    }
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in stats.items():
            writer.writerow([key, value])


# %%
def main(dataset_size: int = 128) -> None:
    config = LoudspeakerConfig(num_samples=512, sample_rate=48000.0)
    key = jr.PRNGKey(202)
    ts, forcing, states = build_loudspeaker_dataset(
        config,
        dataset_size=dataset_size,
        key=key,
        band=(20.0, 1500.0),
    )
    dataset_path = save_npz_bundle(
        OUT_DIR / "linear_loudspeaker_dataset.npz",
        ts=ts,
        forcing=forcing,
        states=states,
    )
    summary_filename = "dataset_summary.csv"
    _write_summary(np.asarray(forcing), np.asarray(states), summary_filename)
    summary_path = OUT_DIR / summary_filename

    preview_ax = plot_timeseries_bundle(
        ts,
        states[0],
        labels=STATE_LABELS_NORMALIZED,
        normalized=True,
        title="Preview Trajectory (sample 0)",
        styles=("solid", "solid", "solid"),
    )
    save_figure(preview_ax, OUT_DIR / "preview_sample.png")

    print(f"Saved dataset to {dataset_path}")
    print(f"Summary written to {summary_path}")


# %%
if __name__ == "__main__":
    print("Exporting loudspeaker dataset for downstream blackbox fitting...")
    main()

# %%
CONE_DISPLACEMENT = LabelSpec("Cone displacement", "m", "x")
CONE_VELOCITY = LabelSpec("Cone velocity", "m/s", "v")
COIL_CURRENT = LabelSpec("Coil current", "A", "i")
STATE_LABELS_NORMALIZED = normalized_labels(
    CONE_DISPLACEMENT, CONE_VELOCITY, COIL_CURRENT
)
