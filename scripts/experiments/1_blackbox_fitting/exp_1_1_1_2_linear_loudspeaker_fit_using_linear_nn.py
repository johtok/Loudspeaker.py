#!/usr/bin/env python3
"""Linear loudspeaker fit using linear NN (taxonomy 1.1.1.2)."""

#%%
import csv
import os
import sys
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUT_DIR = os.path.join(
    ROOT_DIR,
    "out",
    "1_blackbox_fitting",
    "exp_1_1_1_2_linear_loudspeaker_fit_using_linear_nn",
)
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.data import (
    StaticTrainingStrategy,
    TrainingStrategy,
    build_loudspeaker_dataset,
)
from loudspeaker.loudspeaker_sim import LoudspeakerConfig
from loudspeaker import LabelSpec
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
from loudspeaker.plotting import save_figure
from loudspeaker.io import save_npz_bundle


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray]

CONE_DISPLACEMENT = LabelSpec("Cone displacement", "m", "x")
CONE_VELOCITY = LabelSpec("Cone velocity", "m/s", "v")
COIL_CURRENT = LabelSpec("Coil current", "A", "i")
STATE_LABELS = (
    CONE_DISPLACEMENT.raw(),
    CONE_VELOCITY.raw(),
    COIL_CURRENT.raw(),
)
TARGET_LABELS = tuple(f"Reference {label}" for label in STATE_LABELS)
PREDICTION_LABELS = tuple(f"Predicted {label}" for label in STATE_LABELS)
RESIDUAL_LABELS = tuple(f"{label} residual" for label in STATE_LABELS)


def _save_fig(ax, folder: str, filename: str) -> None:
    save_figure(ax, os.path.join(folder, filename))


def _evaluation_batches(forcing: jnp.ndarray, reference: jnp.ndarray) -> list[Batch]:
    return [(forcing[i : i + 1], reference[i : i + 1]) for i in range(forcing.shape[0])]


def _single_batch_loader(batch: Batch) -> Iterable[Batch]:
    yield batch


#%%
def main(
    optimizer_factory=optax.sgd,
    loss: str = "norm_mse",
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

    ts, forcing_values, reference_states = build_loudspeaker_dataset(
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
    eval_forcing = eval_batch[0][0]

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
        target_labels=TARGET_LABELS,
        prediction_labels=PREDICTION_LABELS,
        residual_labels=RESIDUAL_LABELS,
    )
    _save_fig(traj_ax, plot_dir, "training_predictions.png")
    _save_fig(resid_ax, plot_dir, "training_residuals.png")
    loss_ax = plot_neural_ode_loss(trained)
    _save_fig(loss_ax, plot_dir, "training_loss.png")
    save_npz_bundle(
        os.path.join(plot_dir, "evaluation_results.npz"),
        ts=ts,
        forcing=eval_forcing,
        states=eval_reference,
        prediction=eval_prediction,
    )

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
