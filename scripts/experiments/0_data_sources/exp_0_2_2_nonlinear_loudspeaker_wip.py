#!/usr/bin/env python3
"""Nonlinear loudspeaker data source (taxonomy 0.2.2)."""

# %%
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
OUT_DIR = REPO_ROOT / "out" / "0_data_sources" / "exp_0_2_2_nonlinear_loudspeaker"

from loudspeaker import (
    LabelSpec,
    normalized_labels,
    plot_phase_fan,
    plot_timeseries_bundle,
    raw_labels,
    save_figure,
    save_npz_bundle,
)
from loudspeaker.loudspeaker_sim import (
    NonlinearLoudspeakerConfig,
    simulate_nonlinear_loudspeaker_system,
)
from loudspeaker.testsignals import complex_tone_control, pink_noise_control

CONE_DISPLACEMENT = LabelSpec("Cone displacement", "m", "x")
CONE_VELOCITY = LabelSpec("Cone velocity", "m/s", "v")
COIL_CURRENT = LabelSpec("Coil current", "A", "i")
COIL_FORCE = LabelSpec("Coil force", "N", "F_c")
COIL_VOLTAGE = LabelSpec("Voice-coil voltage", "V", "V_c")
STATE_LABELS = raw_labels(CONE_DISPLACEMENT, CONE_VELOCITY, COIL_CURRENT)
STATE_LABELS_NORMALIZED = normalized_labels(
    CONE_DISPLACEMENT, CONE_VELOCITY, COIL_CURRENT
)
FORCE_VOLTAGE_LABELS = normalized_labels(COIL_FORCE, COIL_VOLTAGE)


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
        styles=("solid", "solid", "solid"),
    )
    _save_fig(ts_ax, f"{slug}_timeseries_raw.png")

    ts_norm_ax = plot_timeseries_bundle(
        result.ts,
        result.states,
        labels=STATE_LABELS_NORMALIZED,
        normalized=True,
        title=f"{name} Timeseries (Normalized)",
        styles=("solid", "solid", "solid"),
    )
    _save_fig(ts_norm_ax, f"{slug}_timeseries_normalized.png")

    if result.coil_force is not None:
        mech_ax = plot_timeseries_bundle(
            result.ts,
            np.stack(
                (
                    np.asarray(result.coil_force),
                    np.asarray(forcing_values),
                ),
                axis=1,
            ),
            labels=FORCE_VOLTAGE_LABELS,
            normalized=True,
            title=f"{name} Electrical vs Mechanical Drive",
            styles=("solid", "dashed"),
        )
        _save_fig(mech_ax, f"{slug}_coil_force_vs_voltage.png")

    save_npz_bundle(
        OUT_DIR / f"{slug}_trajectory.npz",
        ts=result.ts,
        forcing=forcing_values,
        states=result.states,
        voltages=result.voltages,
        coil_force=result.coil_force,
    )


# %%
def main() -> None:
    config = NonlinearLoudspeakerConfig(
        num_samples=2048,
        sample_rate=48000.0,
        suspension_cubic=0.9,
        force_factor_sag=0.25,
    )
    key = jr.PRNGKey(111)
    pink_key, _ = jr.split(key)

    pink_control = pink_noise_control(
        num_samples=config.num_samples,
        dt=config.dt,
        key=pink_key,
        band=(20.0, 2000.0),
    )
    complex_control = complex_tone_control(
        num_samples=config.num_samples,
        dt=config.dt,
        frequencies=(35.0, 60.0, 120.0, 220.0, 320.0),
        amplitudes=(1.0, 0.6, 0.4, 0.3, 0.2),
    )

    pink_result = simulate_nonlinear_loudspeaker_system(
        config,
        pink_control,
        capture_details=True,
    )
    complex_result = simulate_nonlinear_loudspeaker_system(
        config,
        complex_control,
        capture_details=True,
    )

    _render_suite("Pink Noise", pink_result, pink_control.values)
    _render_suite("Complex Tone", complex_result, complex_control.values)


# %%
if __name__ == "__main__":
    print("Generating nonlinear loudspeaker trajectories...")
    main()

# %%
