#!/usr/bin/env python3
"""Exp3 MSD simulation demo using shared libs and forcing signals."""

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
    MSDConfig,
    complex_tone_control,
    pink_noise_control,
    plot_normalized_phase_suite,
    plot_phase,
    plot_trajectory,
    simulate_msd_system,
)


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

    plot_trajectory(ts, pink_states, title="Pink Noise Driven MSD")
    plot_trajectory(ts, complex_states, title="Complex Tone Driven MSD")
    plot_phase(pink_states, title="Pink Noise Phase Portrait")
    plot_phase(complex_states, title="Complex Tone Phase Portrait")
    plot_normalized_phase_suite(
        ts,
        pink_states[:, 0],
        pink_states[:, 1],
        pink_acc,
        title_prefix="Pink Noise Normalized Phase Plots",
    )
    plot_normalized_phase_suite(
        ts,
        complex_states[:, 0],
        complex_states[:, 1],
        complex_acc,
        title_prefix="Complex Tone Normalized Phase Plots",
    )


#%%
if __name__ == "__main__":
    main()

# %%
