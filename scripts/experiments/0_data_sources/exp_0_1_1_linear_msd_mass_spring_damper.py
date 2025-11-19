#!/usr/bin/env python3
"""Linear MSD (mass-spring-damper) data source (taxonomy 0.1.1)."""

# %%
import sys
from pathlib import Path

import jax.random as jr

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
    REPO_ROOT / "out" / "0_data_sources" / "exp_0_1_1_linear_msd_mass_spring_damper"
)

from loudspeaker import (
    LabelSpec,
    MSDConfig,
    complex_tone_control,
    normalized_labels,
    pink_noise_control,
    plot_phase_fan,
    plot_timeseries_bundle,
    raw_labels,
    save_figure,
    save_npz_bundle,
    simulate_msd_system,
)

POSITION = LabelSpec("Position", "m", "x")
VELOCITY = LabelSpec("Velocity", "m/s", "v")
STATE_LABELS = raw_labels(POSITION, VELOCITY)
STATE_LABELS_NORMALIZED = normalized_labels(POSITION, VELOCITY)


def _save_fig(ax, filename: str) -> None:
    save_figure(ax, OUT_DIR / filename)


def _render_suite(name: str, result, forcing_values) -> None:
    slug = name.lower().replace(" ", "_")
    phase_ax = plot_phase_fan(
        result.states,
        labels=STATE_LABELS,
        normalized=False,
        title=f"{name} Phase",
    )
    _save_fig(phase_ax, f"{slug}_phase_raw.png")
    phase_norm_ax = plot_phase_fan(
        result.states,
        labels=STATE_LABELS_NORMALIZED,
        normalized=True,
        title=f"{name} Phase (Normalized)",
    )
    _save_fig(phase_norm_ax, f"{slug}_phase_normalized.png")
    ts_ax = plot_timeseries_bundle(
        result.ts,
        result.states,
        labels=STATE_LABELS,
        normalized=False,
        title=f"{name} Timeseries",
        styles=("solid", "solid"),
    )
    _save_fig(ts_ax, f"{slug}_timeseries_raw.png")
    ts_norm_ax = plot_timeseries_bundle(
        result.ts,
        result.states,
        labels=STATE_LABELS_NORMALIZED,
        normalized=True,
        title=f"{name} Timeseries (Normalized)",
        styles=("solid", "solid"),
    )
    _save_fig(ts_norm_ax, f"{slug}_timeseries_normalized.png")
    save_npz_bundle(
        OUT_DIR / f"{slug}_trajectory.npz",
        ts=result.ts,
        forcing=forcing_values,
        states=result.states,
        acceleration=result.acceleration,
    )


# %%
def main() -> None:
    config = MSDConfig()
    key = jr.PRNGKey(0)

    pink_control = pink_noise_control(
        num_samples=config.num_samples,
        dt=config.dt,
        key=key,
        band=(1.0, 100.0),
    )
    complex_control = complex_tone_control(
        num_samples=config.num_samples,
        dt=config.dt,
    )

    pink_result = simulate_msd_system(
        config,
        pink_control,
        capture_details=True,
    )
    complex_result = simulate_msd_system(
        config,
        complex_control,
        capture_details=True,
    )
    _render_suite("Pink Noise", pink_result, pink_control.values)
    _render_suite("Complex Tone", complex_result, complex_control.values)


# %%
if __name__ == "__main__":
    main()

# %%
