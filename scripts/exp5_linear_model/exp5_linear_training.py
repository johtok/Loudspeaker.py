#!/usr/bin/env python3
"""Exp5: Direct linear model fitting using least squares on simulated data."""

#%%
import csv
import os
import sys
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(ROOT_DIR, "out", "exp5_linear_model")
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.data import StaticTrainingStrategy, build_msd_dataset, msd_dataloader
from loudspeaker.metrics import mse
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.models import LinearMSDModel
from loudspeaker.neuralode import (
    NeuralODE,
    build_loss_fn,
    plot_neural_ode_predictions,
    predict_neural_ode,
    train_neural_ode,
)
from loudspeaker.plotting import plot_timeseries_bundle


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jax.Array, jax.Array]


def _save_fig(ax, folder: str, filename: str) -> None:
    if isinstance(ax, np.ndarray):
        fig = ax.ravel()[0].figure
    else:
        fig = ax.figure
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _evaluation_batches(forcing: jnp.ndarray, reference: jnp.ndarray) -> Iterable[Batch]:
    for i in range(forcing.shape[0]):
        yield forcing[i : i + 1], reference[i : i + 1]

# %%

def main(
    optimizer_factory=optax.adam,
    num_samples: int = 64,
    dataset_size: int = 512,
    batch_size: int = 32,
    num_steps: int = 500,
    train_fraction: float = 0.8,
):
    if dataset_size < 2:
        raise ValueError("dataset_size must be at least 2 for train/test splits.")

    config = MSDConfig(num_samples=num_samples)
    key = jr.PRNGKey(2025)
    data_key, model_key, loader_key = jr.split(key, 3)

    dataset = build_msd_dataset(
        config=config,
        dataset_size=dataset_size,
        key=data_key,
        band=(1.0, 80.0),
    )

    train_size = max(1, int(train_fraction * dataset_size))
    if train_size >= dataset_size:
        train_size = dataset_size - 1
    test_size = dataset_size - train_size

    train_forcing, test_forcing = dataset.forcing[:train_size], dataset.forcing[train_size:]
    train_reference, test_reference = dataset.reference[:train_size], dataset.reference[train_size:]

    strategy = StaticTrainingStrategy(steps=num_steps)
    data_loader = msd_dataloader(
        train_forcing,
        train_reference,
        batch_size=batch_size,
        key=loader_key,
        strategy=strategy,
    )

    model = LinearMSDModel(config=config, key=model_key)
    optimizer = optimizer_factory(learning_rate=1e-2)
    loss_fn = build_loss_fn(
        ts=dataset.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )

    neural_ode = NeuralODE(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ts=dataset.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=strategy.total_steps,
    )

    trained = train_neural_ode(neural_ode, data_loader)
    plot_dir = os.path.join(
        OUT_DIR,
        f"exp5_{optimizer_factory.__name__}_samples_{num_samples}_ds_{dataset_size}_bs_{batch_size}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    test_mse = None
    if test_size > 0:
        test_predictions, test_targets = predict_neural_ode(trained, _evaluation_batches(test_forcing, test_reference))
        test_mse = float(mse(test_predictions, test_targets))
        print("Exp5 Test MSE:", test_mse)
        eval_ts = neural_ode.ts[: test_predictions.shape[1]]
        first_ref = test_targets[0]
        first_pred = test_predictions[0]
        state_bundle = jnp.column_stack(
            [
                first_ref[:, 0],
                first_pred[:, 0],
                first_ref[:, 1],
                first_pred[:, 1],
            ]
        )
        ts_ax = plot_timeseries_bundle(
            eval_ts,
            state_bundle,
            labels=("pos target", "pos pred", "vel target", "vel pred"),
            styles=("solid", "--", "solid", "--"),
            title="Exp5 Test State Trajectories",
        )
        _save_fig(ts_ax, plot_dir, "test_states.png")
        traj_ax, resid_ax = plot_neural_ode_predictions(
            trained,
            _evaluation_batches(test_forcing, test_reference),
            max_batches=1,
            title="Exp5 Test Predictions",
            target_labels=("position", "velocity"),
            prediction_labels=("pred position", "pred velocity"),
        )
        _save_fig(traj_ax, plot_dir, "test_predictions.png")
        _save_fig(resid_ax, plot_dir, "test_residuals.png")

    csv_path = os.path.join(plot_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["final_mae", float(mae(eval_prediction, eval_reference))])
        writer.writerow(["final_mse", float(mse(eval_prediction, eval_reference))])
        if test_mse is not None:
            writer.writerow(["test_mse", test_mse])

    base_model = LinearMSDModel(config=config)
    param_mse = float(jnp.mean((trained.model.weight - base_model.weight) ** 2))
    print("Exp5 state matrix parameter MSE:", param_mse)
    print("Exp5 training history length:", len(trained.history))

# %%

if __name__ == "__main__":
    print("Running Exp5 linear model training...")
    main()

# %%
