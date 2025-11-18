#!/usr/bin/env python3
"""Exp4 loudspeaker neural ODE training using shared libs."""

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
OUT_DIR = os.path.join(ROOT_DIR, "out", "exp4_loudspeaker_sim")
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.data import StaticTrainingStrategy, TrainingStrategy
from loudspeaker.loudspeaker_sim import LoudspeakerConfig
from loudspeaker.metrics import mae, mse
from loudspeaker.models import LinearLoudspeakerModel
from loudspeaker.neuralode import (
    NeuralODE,
    build_loss_fn,
    plot_neural_ode_loss,
    plot_neural_ode_predictions,
    predict_neural_ode,
    solve_with_model,
    train_neural_ode,
)
from loudspeaker.testsignals import pink_noise_control


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray]


def _save_fig(ax, folder: str, filename: str) -> None:
    if isinstance(ax, np.ndarray):
        fig = ax.ravel()[0].figure
    else:
        fig = ax.figure
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _evaluation_batches(forcing: jnp.ndarray, reference: jnp.ndarray) -> list[Batch]:
    return [(forcing[i : i + 1], reference[i : i + 1]) for i in range(forcing.shape[0])]


def _single_batch_loader(batch: Batch) -> Iterable[Batch]:
    yield batch


def _build_loudspeaker_dataset(
    config: LoudspeakerConfig,
    dataset_size: int,
    key: jr.PRNGKey,
    band: Tuple[float, float] | None = (20.0, 1000.0),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing_values = []
    reference_states = []
    rng = key
    for _ in range(dataset_size):
        rng, forcing_key = jr.split(rng)
        control = pink_noise_control(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            band=band,
        )
        forcing_values.append(control.values)
        sim = solve_with_model(
            LinearLoudspeakerModel(config),
            ts,
            control,
            config.initial_state,
            config.dt,
        )
        reference_states.append(sim)
    return ts, jnp.stack(forcing_values), jnp.stack(reference_states)


#%%
def main(
    optimizer_factory=optax.sgd,
    loss: str = "mse",
    num_samples: int = 512,
    dataset_size: int = 64,
    batch_size: int = 4,
    num_steps: int = 200,
    strategy: TrainingStrategy | None = None,
    train_fraction: float = 0.8,
):
    config = LoudspeakerConfig(num_samples=num_samples)
    if dataset_size < 2:
        raise ValueError("dataset_size must be at least 2 to support train/test split.")
    key = jr.PRNGKey(8)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, forcing_values, reference_states = _build_loudspeaker_dataset(
        config=config,
        dataset_size=dataset_size,
        key=data_key,
    )

    train_size = max(1, int(train_fraction * dataset_size))
    if train_size == dataset_size:
        train_size -= 1
    test_size = dataset_size - train_size
    train_forcing, test_forcing = forcing_values[:train_size], forcing_values[train_size:]
    train_reference, test_reference = reference_states[:train_size], reference_states[train_size:]

    if strategy is None:
        strategy = StaticTrainingStrategy(steps=num_steps)
    total_steps = strategy.total_steps

    def dataloader():
        dataset_size = train_forcing.shape[0]
        perm = jnp.arange(dataset_size)
        rng = loader_key
        while True:
            rng, perm_key = jr.split(rng)
            perm = jr.permutation(perm_key, perm)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                if end <= dataset_size:
                    idx = perm[start:end]
                    yield train_forcing[idx], train_reference[idx]

    model = LinearLoudspeakerModel(config=config, key=model_key)
    optimizer = optimizer_factory(learning_rate=1e-3)
    loss_fn = build_loss_fn(
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        loss_type=loss,
    )

    neural_ode = NeuralODE(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=total_steps,
    )

    trained = train_neural_ode(neural_ode, dataloader())
    plot_dir = os.path.join(
        OUT_DIR,
        f"exp4_{optimizer_factory.__name__}_loss_{loss}_samples_{num_samples}_ds_{dataset_size}_bs_{batch_size}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    eval_batch = (test_forcing[:1], test_reference[:1])
    predictions, targets = predict_neural_ode(trained, _single_batch_loader(eval_batch), max_batches=1)
    eval_prediction = predictions[0]
    eval_reference = targets[0]

    final_mae = float(mae(eval_prediction, eval_reference))
    final_mse = float(mse(eval_prediction, eval_reference))
    print("Final MAE:", final_mae)
    print("Final MSE:", final_mse)

    test_mse = None
    if test_size > 0:
        test_loader = iter(_evaluation_batches(test_forcing, test_reference))
        test_predictions, test_targets = predict_neural_ode(trained, test_loader)
        test_mse = float(mse(test_predictions, test_targets))
        print("Test set MSE:", test_mse)

    base_model = LinearLoudspeakerModel(config=config)
    param_mse = float(jnp.mean((trained.model.weight - base_model.weight) ** 2))
    print("State matrix parameter MSE:", param_mse)

    traj_ax, resid_ax = plot_neural_ode_predictions(
        trained,
        _single_batch_loader(eval_batch),
        max_batches=1,
        title="Loudspeaker Reference vs Predicted",
        target_labels=(
            "reference displacement",
            "reference velocity",
            "reference coil current",
        ),
        prediction_labels=(
            "predicted displacement",
            "predicted velocity",
            "predicted coil current",
        ),
        residual_labels=(
            "displacement residual",
            "velocity residual",
            "coil current residual",
        ),
    )
    _save_fig(traj_ax, plot_dir, "training_predictions.png")
    _save_fig(resid_ax, plot_dir, "training_residuals.png")
    loss_ax = plot_neural_ode_loss(trained)
    _save_fig(loss_ax, plot_dir, "training_loss.png")

    csv_path = os.path.join(plot_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["final_mae", final_mae])
        writer.writerow(["final_mse", final_mse])
        if test_mse is not None:
            writer.writerow(["test_mse", test_mse])

# %%

if __name__ == "__main__":
    print("Training Exp4 loudspeaker Neural ODE (SGD)...")
    main()
# %%

if __name__ == "__main__":
    print("Training Exp4 loudspeaker Neural ODE (Adam)...")
    main(optimizer_factory=optax.adam)

# %%
