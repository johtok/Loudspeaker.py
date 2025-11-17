#!/usr/bin/env python3
"""Exp4 Loudspeaker simulation using shared libs and forcing signals."""

#%%
import os
import sys

import jax.random as jr

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
        plot_phase_fan(
            result.states,
            labels=state_labels,
            normalized=False,
            title=f"Exp4 {name} Phase",
        )
        plot_phase_fan(
            result.states,
            labels=state_labels,
            normalized=True,
            title=f"Exp4 {name} Phase",
        )
        plot_timeseries_bundle(
            ts,
            result.states,
            labels=state_labels,
            normalized=False,
            title=f"Exp4 {name} Timeseries",
        )
        plot_timeseries_bundle(
            ts,
            result.states,
            labels=state_labels,
            normalized=True,
            title=f"Exp4 {name} Timeseries",
        )

    _render_suite(pink_result, "Pink Noise")
    _render_suite(complex_result, "Complex Tone")


#%%
if __name__ == "__main__":
    print("Running Exp4 loudspeaker simulation...")
    main()

# %%
