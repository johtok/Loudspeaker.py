#!/usr/bin/env python3
"""Linear MSD fit using linear NN (taxonomy 1.1.1.1)."""

# %%
import csv
import functools
import sys
from pathlib import Path
from typing import Callable, Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

_EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(_EXPERIMENTS_ROOT) not in sys.path:
    sys.path.append(str(_EXPERIMENTS_ROOT))

if __package__ in (None, ""):
    from _paths import REPO_ROOT, ensure_sys_path, script_dir
else:
    from ._paths import REPO_ROOT, ensure_sys_path, script_dir

SCRIPT_DIR = script_dir(__file__)
ensure_sys_path(SCRIPT_DIR)
OUT_DIR = (
    REPO_ROOT
    / "out"
    / "1_blackbox_fitting"
    / "exp_1_1_1_1_linear_msd_fit_using_linear_nn"
)

from loudspeaker import LabelSpec
from loudspeaker.data import (
    StaticTrainingStrategy,
    TrainTestSplit,
    TrainingStrategy,
    build_msd_dataset,
    msd_dataloader,
)
from loudspeaker.io import save_npz_bundle
from loudspeaker.metrics import mae, mse
from loudspeaker.models import LinearMSDModel
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.neuralode import (
    NeuralODE,
    TensorBoardCallback,
    build_loss_fn,
    plot_neural_ode_loss,
    plot_neural_ode_predictions,
    predict_neural_ode,
    tensorboard_log_time_series,
    train_neural_ode,
)
from loudspeaker.plotting import normalize_state_pair, save_figure

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


def _evaluation_batches(forcing: jnp.ndarray, reference: jnp.ndarray) -> list[Batch]:
    return [(forcing[i : i + 1], reference[i : i + 1]) for i in range(forcing.shape[0])]


def _single_batch_loader(batch: Batch) -> Iterable[Batch]:
    yield batch


# %%
def _optimizer_name(factory: Callable[..., optax.GradientTransformation]) -> str:
    if hasattr(factory, "__name__"):
        return factory.__name__  # type: ignore[attr-defined]
    if isinstance(factory, functools.partial):
        return getattr(factory.func, "__name__", factory.__class__.__name__)
    return factory.__class__.__name__


def main(
    optimizer_factory=optax.sgd,
    loss: str = "norm_mse",
    num_samples: int = 20,
    dataset_size: int = 128,
    batch_size: int = 8,
    num_steps: int = 400,
    strategy: TrainingStrategy | None = None,
    train_fraction: float = 0.8,
):
    config = MSDConfig(num_samples=num_samples)
    if dataset_size < 2:
        raise ValueError("dataset_size must be at least 2 to support train/test split.")
    key = jr.PRNGKey(42)
    data_key, model_key, loader_key = jr.split(key, 3)

    dataset = build_msd_dataset(
        config=config,
        dataset_size=dataset_size,
        key=data_key,
        band=(1.0, 100.0),
    )
    ts = dataset.ts
    forcing_values = dataset.forcing
    reference_states = dataset.reference

    split = TrainTestSplit.from_dataset(
        forcing_values,
        reference_states,
        train_fraction=train_fraction,
    )
    train_forcing = split.train_forcing
    train_reference = split.train_reference

    if strategy is None:
        strategy = StaticTrainingStrategy(steps=num_steps)
    total_steps = strategy.total_steps

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
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        loss_type=loss,
    )

    optimizer_name = _optimizer_name(optimizer_factory)
    run_name = f"exp3_{optimizer_name}_loss_{loss}_samples_{num_samples}_ds_{dataset_size}_bs_{batch_size}"
    tensorboard_dir = (
        REPO_ROOT / "out" / "tensorboard" / "1_blackbox_fitting" / run_name
    )

    neural_ode = NeuralODE(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=total_steps,
        tensorboard_callback=TensorBoardCallback(str(tensorboard_dir)),
    )

    trained = train_neural_ode(neural_ode, data_loader)
    plot_dir = OUT_DIR / run_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    eval_batch = split.evaluation_batch()
    predictions, targets = predict_neural_ode(
        trained, _single_batch_loader(eval_batch), max_batches=1
    )
    eval_prediction = predictions[0]
    eval_reference = targets[0]
    eval_forcing = eval_batch[0][0]

    final_mae = float(mae(eval_prediction, eval_reference))
    final_mse = float(mse(eval_prediction, eval_reference))
    print("Final MAE:", final_mae)
    print("Final MSE:", final_mse)

    test_loader = iter(split.evaluation_batches())
    test_predictions, test_targets = predict_neural_ode(trained, test_loader)
    test_mse = float(mse(test_predictions, test_targets))
    print("Test set MSE:", test_mse)

    base_model = LinearMSDModel(config=config)
    param_mse = float(jnp.mean((trained.model.weight - base_model.weight) ** 2))
    print("State matrix parameter MSE:", param_mse)

    traj_ax, resid_ax = plot_neural_ode_predictions(
        trained,
        _single_batch_loader(eval_batch),
        max_batches=1,
        title="Reference vs Predicted Trajectory",
        target_labels=TARGET_LABELS,
        prediction_labels=PREDICTION_LABELS,
        residual_labels=RESIDUAL_LABELS,
    )
    norm_traj_ax, norm_resid_ax = plot_neural_ode_predictions(
        trained,
        _single_batch_loader(eval_batch),
        max_batches=1,
        title="Normalized Reference vs Predicted Trajectory",
        target_labels=TARGET_LABELS,
        prediction_labels=PREDICTION_LABELS,
        residual_labels=RESIDUAL_LABELS,
        normalize=True,
    )
    loss_ax = plot_neural_ode_loss(trained)
    tensorboard_cb = trained.tensorboard_callback
    if tensorboard_cb is not None:
        summary_step = int(total_steps)
        raw_residuals = eval_prediction - eval_reference
        norm_target, norm_prediction = normalize_state_pair(
            eval_reference, eval_prediction
        )
        norm_residuals = norm_prediction - norm_target
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/states/raw/reference",
            eval_reference,
            labels=TARGET_LABELS,
            step_offset=summary_step,
        )
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/states/raw/predicted",
            eval_prediction,
            labels=PREDICTION_LABELS,
            step_offset=summary_step,
        )
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/states/normalized/reference",
            norm_target,
            labels=TARGET_LABELS,
            step_offset=summary_step,
        )
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/states/normalized/predicted",
            norm_prediction,
            labels=PREDICTION_LABELS,
            step_offset=summary_step,
        )
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/residuals/raw",
            raw_residuals,
            labels=RESIDUAL_LABELS,
            step_offset=summary_step,
        )
        tensorboard_log_time_series(
            tensorboard_cb,
            "training/residuals/normalized",
            norm_residuals,
            labels=RESIDUAL_LABELS,
            step_offset=summary_step,
        )
        for idx, value in enumerate(trained.history):
            tensorboard_cb.log_scalar("training/loss/history", idx, value)
        tensorboard_cb.log_scalar("metrics/final_mae", summary_step, final_mae)
        tensorboard_cb.log_scalar("metrics/final_mse", summary_step, final_mse)
        tensorboard_cb.log_scalar("metrics/test_mse", summary_step, test_mse)
        tensorboard_cb.log_scalar("metrics/param_mse", summary_step, param_mse)
    _save_fig(traj_ax, plot_dir, "training_predictions.png")
    _save_fig(resid_ax, plot_dir, "training_residuals.png")
    _save_fig(norm_traj_ax, plot_dir, "training_predictions_normalized.png")
    _save_fig(norm_resid_ax, plot_dir, "training_residuals_normalized.png")
    _save_fig(loss_ax, plot_dir, "training_loss.png")
    save_npz_bundle(
        plot_dir / "evaluation_results.npz",
        ts=ts,
        forcing=eval_forcing,
        states=eval_reference,
        prediction=eval_prediction,
    )

    csv_path = plot_dir / "metrics.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["final_mae", final_mae])
        writer.writerow(["final_mse", final_mse])
        writer.writerow(["test_mse", test_mse])
        writer.writerow(["param_mse", param_mse])


# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader (SGD)...")
    main()

# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader (Adam)...")
    main(optimizer_factory=optax.adam)

# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader (AdamW)...")
    main(optimizer_factory=optax.adamw)
# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader (Nadam)...")
    main(optimizer_factory=optax.nadam)
# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader (Lion)...")
    main(optimizer_factory=optax.lion)

# %%
