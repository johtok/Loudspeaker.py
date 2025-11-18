#!/usr/bin/env python3
"""Exp4 Loudspeaker simulation using shared libs and forcing signals."""

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
    LoudspeakerConfig,
    plot_phase_fan,
    plot_timeseries_bundle,
    simulate_loudspeaker_system,
)
from loudspeaker.testsignals import complex_tone_control, pink_noise_control


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

    state_labels = ("cone displacement", "cone velocity", "coil current")

    def _render_suite(result, name):
        slug = name.lower().replace(" ", "_")
        phase_ax = plot_phase_fan(
            result.states,
            labels=state_labels,
            normalized=False,
            title=f"Exp4 {name} Phase",
        )
        _save_fig(phase_ax, f"{slug}_phase_raw.png")
        phase_norm_ax = plot_phase_fan(
            result.states,
            labels=state_labels,
            normalized=True,
            title=f"Exp4 {name} Phase",
        )
        _save_fig(phase_norm_ax, f"{slug}_phase_normalized.png")
        ts_ax = plot_timeseries_bundle(
            ts,
            result.states,
            labels=state_labels,
            normalized=False,
            title=f"Exp4 {name} Timeseries",
            styles=("solid", "solid", "solid"),
        )
        _save_fig(ts_ax, f"{slug}_timeseries_raw.png")
        ts_norm_ax = plot_timeseries_bundle(
            ts,
            result.states,
            labels=state_labels,
            normalized=True,
            title=f"Exp4 {name} Timeseries",
            styles=("solid", "solid", "solid"),
        )
        _save_fig(ts_norm_ax, f"{slug}_timeseries_normalized.png")

    _render_suite(pink_result, "Pink Noise")
    _render_suite(complex_result, "Complex Tone")


#%%
if __name__ == "__main__":
    print("Running Exp4 loudspeaker simulation...")
    main()

# %%
