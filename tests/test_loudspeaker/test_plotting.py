from __future__ import annotations

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loudspeaker import plotting


def _sample_states():
    ts = np.linspace(0.0, 1.0, 5)
    position = np.sin(2 * np.pi * ts)
    velocity = np.cos(2 * np.pi * ts)
    states = np.stack([position, velocity], axis=1)
    return ts, states


def test_plot_trajectory_returns_axis():
    ts, states = _sample_states()
    ax = plotting.plot_trajectory(ts, states)
    assert ax is not None
    assert len(ax.lines) == states.shape[1]


def test_plot_phase_draws_phase_portrait():
    _, states = _sample_states()
    ax = plotting.plot_phase(states)
    assert ax is not None
    assert len(ax.lines) == 1


def test_plot_residuals_creates_series():
    ts, states = _sample_states()
    zeros = np.zeros_like(states)
    ax = plotting.plot_residuals(ts, states, zeros)
    assert ax is not None
    assert len(ax.lines) == states.shape[1]


def test_plot_loss_plots_series():
    losses = [1.0, 0.5, 0.25]
    ax = plotting.plot_loss(losses)
    assert ax is not None
    assert len(ax.lines) == 1


def test_plot_normalized_phase_suite_returns_axes_array():
    ts, states = _sample_states()
    axes = plotting.plot_normalized_phase_suite(
        ts,
        states[:, 0],
        states[:, 1],
        np.gradient(states[:, 1], ts),
    )
    assert axes.shape == (2, 2)
