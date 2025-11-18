#!/usr/bin/env python3
"""Exp3 MSD simulation demo using shared libs and forcing signals."""

#%%
import os
import sys

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker import (
    MSDConfig,
    complex_tone_control,
    pink_noise_control,
    plot_phase_fan,
    plot_timeseries_bundle,
    simulate_msd_system,
)


def _save_fig(ax, filename: str) -> None:
    if isinstance(ax, np.ndarray):
        fig = ax.ravel()[0].figure
    else:
        fig = ax.figure
    path = os.path.join(SCRIPT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


#%%
def main():
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
    ts = pink_result.ts
    pink_states = pink_result.states
    complex_states = complex_result.states
    pink_acc = pink_result.acceleration
    complex_acc = complex_result.acceleration

    labels = ("position", "velocity")
    for idx, (name, states) in enumerate(
        ("Pink Noise", pink_states),
        ("Complex Tone", complex_states),
    ):
        slug = name.lower().replace(" ", "_")
        phase_ax = plot_phase_fan(
            states,
            labels=labels,
            normalized=False,
            title=f"{name} Phase",
        )
        _save_fig(phase_ax, f"{slug}_phase_raw.png")
        phase_norm_ax = plot_phase_fan(
            states,
            labels=labels,
            normalized=True,
            title=f"{name} Phase",
        )
        _save_fig(phase_norm_ax, f"{slug}_phase_normalized.png")
        ts_ax = plot_timeseries_bundle(
            ts,
            states,
            labels=labels,
            normalized=False,
            title=f"{name} Timeseries",
            styles=("solid", "solid"),
        )
        _save_fig(ts_ax, f"{slug}_timeseries_raw.png")
        ts_norm_ax = plot_timeseries_bundle(
            ts,
            states,
            labels=labels,
            normalized=True,
            title=f"{name} Timeseries",
            styles=("solid", "solid"),
        )
        _save_fig(ts_norm_ax, f"{slug}_timeseries_normalized.png")


#%%
if __name__ == "__main__":
    main()

# %%
