#!/usr/bin/env python3
"""Linear loudspeaker data source (taxonomy 0.1.2)."""

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
OUT_DIR = REPO_ROOT / "out" / "0_data_sources" / "exp_0_1_2_linear_loudspeaker"

from loudspeaker import (
    LabelSpec,
    LoudspeakerConfig,
    normalized_labels,
    plot_phase_fan,
    plot_timeseries_bundle,
    raw_labels,
    save_figure,
    save_npz_bundle,
    simulate_loudspeaker_system,
)
from loudspeaker.testsignals import complex_tone_control, pink_noise_control

CONE_DISPLACEMENT = LabelSpec("Cone displacement", "m", "x")
CONE_VELOCITY = LabelSpec("Cone velocity", "m/s", "v")
COIL_CURRENT = LabelSpec("Coil current", "A", "i")
STATE_LABELS = raw_labels(CONE_DISPLACEMENT, CONE_VELOCITY, COIL_CURRENT)
STATE_LABELS_NORMALIZED = normalized_labels(
    CONE_DISPLACEMENT, CONE_VELOCITY, COIL_CURRENT
)


def _save_fig(ax, filename: str) -> None:
    save_figure(ax, OUT_DIR / filename)


# %%
def main() -> None:
    config = LoudspeakerConfig(num_samples=1024, sample_rate=48000.0)
    key = jr.PRNGKey(84)
    pink_key, complex_key = jr.split(key)

    pink_control = pink_noise_control(
        num_samples=config.num_samples,
        dt=config.dt,
        key=pink_key,
        band=(10.0, 800.0),
    )
    complex_control = complex_tone_control(
        num_samples=config.num_samples,
        dt=config.dt,
        frequencies=(30.0, 55.0, 90.0, 150.0),
    )

    pink_result = simulate_loudspeaker_system(
        config,
        pink_control,
        capture_details=True,
    )
    complex_result = simulate_loudspeaker_system(
        config,
        complex_control,
        capture_details=True,
    )
    ts = pink_result.ts

    def _render_suite(result, name: str, forcing_values) -> None:
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
            ts,
            result.states,
            labels=STATE_LABELS,
            normalized=False,
            title=f"{name} Timeseries",
            styles=("solid", "solid", "solid"),
        )
        _save_fig(ts_ax, f"{slug}_timeseries_raw.png")
        ts_norm_ax = plot_timeseries_bundle(
            ts,
            result.states,
            labels=STATE_LABELS_NORMALIZED,
            normalized=True,
            title=f"{name} Timeseries (Normalized)",
            styles=("solid", "solid", "solid"),
        )
        _save_fig(ts_norm_ax, f"{slug}_timeseries_normalized.png")
        save_npz_bundle(
            OUT_DIR / f"{slug}_trajectory.npz",
            ts=ts,
            forcing=forcing_values,
            states=result.states,
            voltages=result.voltages,
            coil_force=result.coil_force,
        )

    _render_suite(pink_result, "Pink Noise", pink_control.values)
    _render_suite(complex_result, "Complex Tone", complex_control.values)


# %%
if __name__ == "__main__":
    print("Running linear loudspeaker data generator...")
    main()

# %%
