#!/usr/bin/env python3
"""Nonlinear MSD (Duffing) data source (taxonomy 0.2.1)."""

#%%
import os
import sys

import jax.random as jr
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUT_DIR = os.path.join(
    ROOT_DIR,
    "out",
    "0_data_sources",
    "exp_0_2_1_nonlinear_msd",
)
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker import (
    LabelSpec,
    normalized_labels,
    plot_phase_fan,
    plot_timeseries_bundle,
    raw_labels,
    save_figure,
    save_npz_bundle,
)
from loudspeaker.nonlinear_msd import NonlinearMSDSimConfig, simulate_nonlinear_msd_system
from loudspeaker.testsignals import complex_tone_control, pink_noise_control

POSITION = LabelSpec("Position", "m", "x")
VELOCITY = LabelSpec("Velocity", "m/s", "v")
ACCELERATION = LabelSpec("Acceleration", "m/s^2", "a")
FORCING = LabelSpec("Forcing force", "N", "F")
STATE_LABELS = raw_labels(POSITION, VELOCITY)
STATE_LABELS_NORMALIZED = normalized_labels(POSITION, VELOCITY)
ACC_FORCE_LABELS = normalized_labels(ACCELERATION, FORCING)

def _save_fig(ax, filename: str) -> None:
    save_figure(ax, os.path.join(OUT_DIR, filename))


def _render_suite(
    name: str,
    result,
    force_values,
) -> None:
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
    if result.acceleration is not None:
        acc_ax = plot_timeseries_bundle(
            result.ts,
            np.stack(
                (
                    np.asarray(result.acceleration),
                    np.asarray(force_values),
                ),
                axis=1,
            ),
            labels=ACC_FORCE_LABELS,
            normalized=True,
            title=f"{name} Acceleration vs Forcing",
            styles=("solid", "dotted"),
        )
        _save_fig(acc_ax, f"{slug}_acc_vs_force.png")
    save_npz_bundle(
        os.path.join(OUT_DIR, f"{slug}_trajectory.npz"),
        ts=result.ts,
        forcing=force_values,
        states=result.states,
        acceleration=result.acceleration,
    )


#%%
def main() -> None:
    config = NonlinearMSDSimConfig(
        num_samples=2048,
        sample_rate=400.0,
        cubic=12.0,
    )
    key = jr.PRNGKey(21)
    pink_key, complex_key = jr.split(key)

    pink_control = pink_noise_control(
        num_samples=config.num_samples,
        dt=config.dt,
        key=pink_key,
        band=(1.0, 80.0),
    )
    complex_control = complex_tone_control(
        num_samples=config.num_samples,
        dt=config.dt,
        frequencies=(5.0, 12.0, 21.0, 34.0),
        amplitudes=(1.0, 0.8, 0.6, 0.4),
    )

    pink_result = simulate_nonlinear_msd_system(
        config,
        pink_control,
        capture_details=True,
    )
    complex_result = simulate_nonlinear_msd_system(
        config,
        complex_control,
        capture_details=True,
    )

    _render_suite("Pink Noise", pink_result, pink_control.values)
    _render_suite("Complex Tone", complex_result, complex_control.values)


#%%
if __name__ == "__main__":
    print("Generating nonlinear MSD trajectories...")
    main()

# %%
