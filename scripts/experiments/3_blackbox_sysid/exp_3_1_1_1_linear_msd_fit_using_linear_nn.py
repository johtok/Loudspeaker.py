#!/usr/bin/env python3
"""Linear MSD fit using linear NN (taxonomy 3.1.1.1)."""

# %%
import csv
from pathlib import Path
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

if __package__ in (None, ""):
    from _paths import REPO_ROOT, ensure_sys_path, script_dir
else:
    from ._paths import REPO_ROOT, ensure_sys_path, script_dir

SCRIPT_DIR = script_dir(__file__)
ensure_sys_path(SCRIPT_DIR)
OUT_DIR = (
    REPO_ROOT
    / "out"
    / "3_blackbox_sysid"
    / "exp_3_1_1_1_linear_msd_fit_using_linear_nn"
)

from loudspeaker import LabelSpec
from loudspeaker.data import StaticTrainingStrategy, build_msd_dataset, msd_dataloader
from loudspeaker.io import save_npz_bundle
from loudspeaker.metrics import mae, mse
from loudspeaker.models import LinearMSDModel
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.neuralode import (
    NeuralODE,
    TensorBoardCallback,
    build_loss_fn,
    plot_neural_ode_predictions,
    predict_neural_ode,
    train_neural_ode,
)
from loudspeaker.plotting import plot_timeseries_bundle, save_figure

jax.config.update("jax_enable_x64", True)


Batch = Tuple[jax.Array, jax.Array]

POSITION = LabelSpec("Position", "m", "x")
VELOCITY = LabelSpec("Velocity", "m/s", "v")
POSITION_LABEL = POSITION.raw()
VELOCITY_LABEL = VELOCITY.raw()
TARGET_LABELS = (
    f"Reference {POSITION_LABEL}",
    f"Reference {VELOCITY_LABEL}",
)
PREDICTION_LABELS = (
    f"Predicted {POSITION_LABEL}",
    f"Predicted {VELOCITY_LABEL}",
)
RESIDUAL_LABELS = (
    f"{POSITION_LABEL} residual",
    f"{VELOCITY_LABEL} residual",
)


def _save_fig(ax, folder: Path, filename: str) -> None:
    save_figure(ax, folder / filename)


def _evaluation_batches(
    forcing: jnp.ndarray, reference: jnp.ndarray
) -> Iterable[Batch]:
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

    train_forcing, test_forcing = (
        dataset.forcing[:train_size],
        dataset.forcing[train_size:],
    )
    train_reference, test_reference = (
        dataset.reference[:train_size],
        dataset.reference[train_size:],
    )

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

    run_name = f"exp5_{optimizer_factory.__name__}_samples_{num_samples}_ds_{dataset_size}_bs_{batch_size}"
    tensorboard_dir = REPO_ROOT / "out" / "tensorboard" / "3_blackbox_sysid" / run_name

    neural_ode = NeuralODE(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ts=dataset.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=strategy.total_steps,
        tensorboard_callback=TensorBoardCallback(str(tensorboard_dir)),
    )

    trained = train_neural_ode(neural_ode, data_loader)
    plot_dir = OUT_DIR / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    eval_predictions, eval_targets = predict_neural_ode(
        trained,
        _evaluation_batches(train_forcing[:1], train_reference[:1]),
    )
    eval_prediction = eval_predictions[0]
    eval_reference = eval_targets[0]
    eval_forcing = train_forcing[0]

    test_mse = None
    if test_size > 0:
        test_predictions, test_targets = predict_neural_ode(
            trained, _evaluation_batches(test_forcing, test_reference)
        )
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
        bundle_labels = (
            f"Target {POSITION_LABEL}",
            f"Predicted {POSITION_LABEL}",
            f"Target {VELOCITY_LABEL}",
            f"Predicted {VELOCITY_LABEL}",
        )
        ts_ax = plot_timeseries_bundle(
            eval_ts,
            state_bundle,
            labels=bundle_labels,
            styles=("solid", "--", "solid", "--"),
            title="Exp5 Test State Trajectories",
        )
        _save_fig(ts_ax, plot_dir, "test_states.png")
        traj_ax, resid_ax = plot_neural_ode_predictions(
            trained,
            _evaluation_batches(test_forcing, test_reference),
            max_batches=1,
            title="Exp5 Test Predictions",
            target_labels=TARGET_LABELS,
            prediction_labels=PREDICTION_LABELS,
            residual_labels=RESIDUAL_LABELS,
        )
        _save_fig(traj_ax, plot_dir, "test_predictions.png")
        _save_fig(resid_ax, plot_dir, "test_residuals.png")
    save_npz_bundle(
        plot_dir / "evaluation_results.npz",
        ts=dataset.ts,
        forcing=eval_forcing,
        states=eval_reference,
        prediction=eval_prediction,
    )

    base_model = LinearMSDModel(config=config)
    param_mse = float(jnp.mean((trained.model.weight - base_model.weight) ** 2))
    print("Exp5 state matrix parameter MSE:", param_mse)
    print("Exp5 training history length:", len(trained.history))

    csv_path = plot_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["final_mae", float(mae(eval_prediction, eval_reference))])
        writer.writerow(["final_mse", float(mse(eval_prediction, eval_reference))])
        if test_mse is not None:
            writer.writerow(["test_mse", test_mse])
        writer.writerow(["param_mse", param_mse])


# %%

if __name__ == "__main__":
    print("Running Exp5 linear model training...")
    main()

# %%
