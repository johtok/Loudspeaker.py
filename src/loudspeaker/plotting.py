from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure


def _normalize_data(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True) + 1e-8
    return (values - mean) / std, mean, std


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
    styles: Iterable[str] | None = None,
) -> Axes:
    ax = _get_ax(ax)
    states_np = np.asarray(states)
    style_list = list(styles) if styles is not None else None
    for dim, label in enumerate(labels):
        if style_list is not None and dim < len(style_list):
            ax.plot(ts, states_np[:, dim], label=label, linestyle=style_list[dim])
        else:
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


def plot_phase_fan(
    states: npt.ArrayLike,
    *,
    normalized: bool = False,
    labels: Iterable[str] | None = None,
    title: str = "Phase Fan Plot",
    colors: Iterable[str] | None = None,
) -> Axes:
    states_np = np.asarray(states)
    if states_np.ndim != 2 or states_np.shape[1] < 2:
        raise ValueError(
            "states must be (num_samples, num_dims>=2) for phase fan plots."
        )
    plot_values = states_np
    if normalized:
        plot_values, _, _ = _normalize_data(states_np)
    label_list = (
        list(labels)
        if labels is not None
        else [f"state_{i}" for i in range(states_np.shape[1])]
    )
    color_list = list(colors) if colors is not None else ["C0", "C1", "C2", "C3"]
    _, ax = plt.subplots(figsize=(6, 4))
    base = plot_values[:, 0]
    for idx in range(1, plot_values.shape[1]):
        ax.plot(
            base,
            plot_values[:, idx],
            label=f"{label_list[0]} vs {label_list[idx]}",
            color=color_list[idx % len(color_list)],
        )
    mode = "Normalized" if normalized else "Raw"
    ax.set_xlabel(label_list[0])
    ax.set_ylabel("State value")
    ax.set_title(f"{title} ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def plot_timeseries_bundle(
    ts: npt.ArrayLike,
    states: npt.ArrayLike,
    *,
    normalized: bool = False,
    labels: Iterable[str] | None = None,
    title: str = "Timeseries Bundle",
    colors: Iterable[str] | None = None,
    styles: Iterable[str] | None = None,
) -> Axes:
    ts_np = np.asarray(ts)
    states_np = np.asarray(states)
    if states_np.ndim != 2:
        raise ValueError(
            "states must be (num_samples, num_dims) for time-series bundles."
        )
    if ts_np.shape[0] != states_np.shape[0]:
        raise ValueError("ts must match number of samples.")
    plot_values = states_np
    if normalized:
        plot_values, _, _ = _normalize_data(states_np)
    label_list = (
        list(labels)
        if labels is not None
        else [f"state_{i}" for i in range(states_np.shape[1])]
    )
    color_list = list(colors) if colors is not None else ["C0", "C1", "C2", "C3"]
    _, ax = plt.subplots(figsize=(8, 4))
    style_list = list(styles) if styles is not None else None
    for idx in range(plot_values.shape[1]):
        kwargs: dict[str, Any] = {
            "label": label_list[idx],
            "color": color_list[idx % len(color_list)],
        }
        if style_list is not None and idx < len(style_list):
            kwargs["linestyle"] = style_list[idx]
        ax.plot(ts_np, plot_values[:, idx], **kwargs)
    mode = "Normalized" if normalized else "Raw"
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State value")
    ax.set_title(f"{title} ({mode})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def save_figure(
    ax: Axes | np.ndarray | Figure | SubFigure,
    path: str | Path,
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = True,
) -> None:
    """Persist matplotlib content regardless of axes layout."""

    fig_like: Figure | SubFigure
    if isinstance(ax, np.ndarray):
        first = ax.ravel()[0]
        fig_like = cast(Figure, first.figure)
    elif isinstance(ax, SubFigure):
        fig_like = ax
    elif isinstance(ax, Figure):
        fig_like = ax
    elif isinstance(ax, Axes):
        fig_like = cast(Figure, ax.figure)
    else:
        raise TypeError(f"Unsupported object type for save_figure: {type(ax)}")

    fig = fig_like.figure if isinstance(fig_like, SubFigure) else fig_like
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    if close:
        plt.close(fig)
