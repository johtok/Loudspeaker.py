from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

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


def test_plot_phase_fan_supports_multiple_states():
    _, states = _sample_states()
    extended = np.column_stack([states, states[:, 0]])
    ax = plotting.plot_phase_fan(extended)
    assert ax is not None


def test_plot_timeseries_bundle_can_normalize():
    ts, states = _sample_states()
    ax = plotting.plot_timeseries_bundle(
        ts, states, normalized=True, styles=("--", "-.")
    )
    assert ax is not None


def test_plot_phase_fan_requires_two_dimensions():
    with pytest.raises(ValueError):
        plotting.plot_phase_fan(np.ones((4, 1)))


def test_plot_phase_fan_normalized_uses_custom_colors():
    _, states = _sample_states()
    extended = np.column_stack([states, states[:, 0]])
    ax = plotting.plot_phase_fan(extended, normalized=True, colors=["k", "r", "g"])
    assert "Normalized" in ax.get_title()


def test_plot_timeseries_bundle_rejects_mismatched_ts():
    ts = np.linspace(0.0, 1.0, 5)
    states = np.ones((4, 2))
    with pytest.raises(ValueError):
        plotting.plot_timeseries_bundle(ts, states)


def test_plot_timeseries_bundle_requires_two_dimensional_states():
    ts = np.linspace(0.0, 1.0, 5)
    states = np.ones(5)
    with pytest.raises(ValueError):
        plotting.plot_timeseries_bundle(ts, states)


def test_plot_normalized_phase_suite_supports_title_prefix():
    ts, states = _sample_states()
    axes = plotting.plot_normalized_phase_suite(
        ts,
        states[:, 0],
        states[:, 1],
        np.gradient(states[:, 1], ts),
        title_prefix="Test Suite",
    )
    assert axes[0, 0].figure._suptitle.get_text() == "Test Suite"


def test_save_figure_supports_array_input(tmp_path):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot([0, 1], [0, 1])
    out = tmp_path / "figure.png"
    plotting.save_figure(axes, out)
    assert out.exists()


def test_save_figure_accepts_figures_and_validates_type(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = tmp_path / "figure2.png"
    plotting.save_figure(fig, out)
    assert out.exists()
    out_axis = tmp_path / "figure3.png"
    plotting.save_figure(ax, out_axis)
    assert out_axis.exists()
    with pytest.raises(TypeError):
        plotting.save_figure("not an axes", tmp_path / "bad.png")
