from __future__ import annotations

from typing import Iterable, Sequence

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def _get_ax(ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_trajectory(
    ts: Sequence[float] | npt.ArrayLike,
    states: npt.ArrayLike,
    labels: Iterable[str] = ("position", "velocity"),
    ax: Axes | None = None,
    title: str | None = "State Trajectories",
) -> Axes:
    ax = _get_ax(ax)
    states_np = np.asarray(states)
    for dim, label in enumerate(labels):
        ax.plot(ts, states_np[:, dim], label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State")
    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def plot_phase(
    states: npt.ArrayLike,
    ax: Axes | None = None,
    title: str = "Phase Portrait",
) -> Axes:
    ax = _get_ax(ax)
    states_np = np.asarray(states)
    ax.plot(states_np[:, 0], states_np[:, 1], marker="o")
    ax.set_xlabel("Position")
    ax.set_ylabel("Velocity")
    ax.set_title(title)
    return ax


def plot_residuals(
    ts: Sequence[float] | npt.ArrayLike,
    target: npt.ArrayLike,
    prediction: npt.ArrayLike,
    labels: Iterable[str] = ("position", "velocity"),
    ax: Axes | None = None,
    title: str = "Residuals",
) -> Axes:
    ax = _get_ax(ax)
    target_np = np.asarray(target)
    prediction_np = np.asarray(prediction)
    residuals = prediction_np - target_np
    for dim, label in enumerate(labels):
        ax.plot(ts, residuals[:, dim], label=f"{label} residual")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Prediction - Target")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_loss(
    losses: Sequence[float],
    ax: Axes | None = None,
    title: str = "Training Loss",
) -> Axes:
    ax = _get_ax(ax)
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    return ax


def plot_normalized_phase_suite(
    ts: Sequence[float] | npt.ArrayLike,
    position: npt.ArrayLike,
    velocity: npt.ArrayLike,
    acceleration: npt.ArrayLike,
    title_prefix: str = "",
) -> npt.NDArray[np.object_]:
    """Replicates the normalized phase plots used in the full MSD visualizer."""

    pos = np.asarray(position)
    vel = np.asarray(velocity)
    acc = np.asarray(acceleration)
    stacked = np.column_stack([pos, vel, acc])
    mean = stacked.mean(axis=0, keepdims=True)
    std = stacked.std(axis=0, keepdims=True) + 1e-8
    norm_values = (stacked - mean) / std

    pos_norm, vel_norm, acc_norm = norm_values.T
    ts_np = np.asarray(ts)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(pos_norm, vel_norm, "b-", linewidth=1, alpha=0.8)
    axes[0, 0].set_xlabel("Normalized Position")
    axes[0, 0].set_ylabel("Normalized Velocity")
    axes[0, 0].set_title("Position-Velocity Phase Plot")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(pos_norm, acc_norm, "r-", linewidth=1, alpha=0.8)
    axes[0, 1].set_xlabel("Normalized Position")
    axes[0, 1].set_ylabel("Normalized Acceleration")
    axes[0, 1].set_title("Position-Acceleration Phase Plot")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(vel_norm, acc_norm, "g-", linewidth=1, alpha=0.8)
    axes[1, 0].set_xlabel("Normalized Velocity")
    axes[1, 0].set_ylabel("Normalized Acceleration")
    axes[1, 0].set_title("Velocity-Acceleration Phase Plot")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(ts_np, pos_norm, label="Position", alpha=0.7)
    axes[1, 1].plot(ts_np, vel_norm, label="Velocity", alpha=0.7)
    axes[1, 1].plot(ts_np, acc_norm, label="Acceleration", alpha=0.7)
    axes[1, 1].set_xlabel("Time [s]")
    axes[1, 1].set_ylabel("Normalized Value")
    axes[1, 1].set_title("Normalized Time Series")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    if title_prefix:
        fig.suptitle(title_prefix)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    return axes
