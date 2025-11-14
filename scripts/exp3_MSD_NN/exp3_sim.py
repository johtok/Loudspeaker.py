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

from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.plotting import (
    plot_normalized_phase_suite,
    plot_phase,
    plot_trajectory,
)
from loudspeaker.testsignals import (
    complex_tone_control,
    pink_noise_control,
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

    ts, pink_states, _, pink_acc = simulate_msd_system(
        config, pink_control, return_details=True
    )
    _, complex_states, _, complex_acc = simulate_msd_system(
        config, complex_control, return_details=True
    )

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
