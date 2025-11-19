#!/usr/bin/env python3
"""Nonlinear MSD fit using nonlinear NN (taxonomy 1.2.2.1)."""

# %%
import csv
import sys
from pathlib import Path
from typing import Iterator, Tuple

import equinox as eqx
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
    / "exp_1_2_2_1_nonlinear_msd_fit_using_nonlinear_nn"
)

from loudspeaker import LabelSpec
from loudspeaker.io import save_npz_bundle
from loudspeaker.metrics import mse, nrmse
from loudspeaker.nonlinear_msd import (
    NonlinearMSDConfig,
    build_nonlinear_msd_training_data,
)
from loudspeaker.plotting import plot_loss, plot_timeseries_bundle, save_figure

jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

VELOCITY = LabelSpec("Velocity", "m/s", "v")
ACCELERATION = LabelSpec("Acceleration", "m/s^2", "a")
VELOCITY_LABEL = VELOCITY.raw()
ACCELERATION_LABEL = ACCELERATION.raw()
VELOCITY_COMPARISON_LABELS = (
    f"Target {VELOCITY_LABEL}",
    f"Predicted {VELOCITY_LABEL}",
)
ACCELERATION_COMPARISON_LABELS = (
    f"Target {ACCELERATION_LABEL}",
    f"Predicted {ACCELERATION_LABEL}",
)


def _save_fig(ax, folder: Path, filename: str) -> None:
    save_figure(ax, folder / filename)


class SingleNN(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.layer1 = eqx.nn.Linear(3, 50, key=k1)
        self.layer2 = eqx.nn.Linear(50, 50, key=k2)
        self.layer3 = eqx.nn.Linear(50, 2, key=k3)

    def __call__(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([state, control], axis=-1)
        x = jnp.tanh(self.layer1(inputs))
        x = jnp.tanh(self.layer2(x))
        return self.layer3(x)


def make_dataloader(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    derivatives: jnp.ndarray,
    *,
    batch_size: int,
    key: jr.PRNGKey,
) -> Iterator[Batch]:
    num_samples = states.shape[0]
    if batch_size > num_samples:
        raise ValueError("batch_size cannot exceed dataset size.")
    rng = key
    while True:
        rng, perm_key = jr.split(rng)
        perm = jr.permutation(perm_key, num_samples)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            if end <= num_samples:
                idx = perm[start:end]
                yield states[idx], controls[idx], derivatives[idx]


def main(
    optimizer_factory=optax.adam,
    config: NonlinearMSDConfig = NonlinearMSDConfig(),
    batch_size: int = 128,
    num_steps: int = 1500,
    train_fraction: float = 0.8,
):
    key = jr.PRNGKey(7)
    states, controls, derivatives = build_nonlinear_msd_training_data(config, key)

    train_size = max(1, int(train_fraction * config.dataset_size))
    if train_size >= config.dataset_size:
        train_size = config.dataset_size - 1
    test_size = config.dataset_size - train_size

    train_states, test_states = states[:train_size], states[train_size:]
    train_controls, test_controls = controls[:train_size], controls[train_size:]
    train_derivs, test_derivs = derivatives[:train_size], derivatives[train_size:]

    loader = make_dataloader(
        train_states,
        train_controls,
        train_derivs,
        batch_size=batch_size,
        key=jr.PRNGKey(101),
    )

    model = SingleNN(key=jr.PRNGKey(55))
    optimizer = optimizer_factory(learning_rate=1e-3)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    def loss_fn(current_model: SingleNN, batch: Batch) -> jnp.ndarray:
        batch_states, batch_controls, batch_derivs = batch
        preds = jax.vmap(current_model)(batch_states, batch_controls)
        return mse(preds, batch_derivs)

    loss_grad = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(
        current_model: SingleNN,
        current_opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[SingleNN, optax.OptState, jnp.ndarray]:
        loss_val, grads = loss_grad(current_model, batch)
        updates, new_state = optimizer.update(
            grads, current_opt_state, eqx.filter(current_model, eqx.is_array)
        )
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_state, loss_val

    history = []
    for _ in range(num_steps):
        batch = next(loader)
        model, opt_state, loss_value = step(model, opt_state, batch)
        history.append(float(loss_value))

    plot_dir = (
        OUT_DIR / f"exp7_{optimizer_factory.__name__}_bs_{batch_size}_steps_{num_steps}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    if history:
        loss_ax = plot_loss(history, title="Exp7 Training Loss")
        _save_fig(loss_ax, plot_dir, "training_loss.png")

    if test_size > 0:
        predictions = jax.vmap(model)(test_states, test_controls)
        std = jnp.std(test_derivs, axis=0, keepdims=True) + 1e-8
        component_nrmse_pct = 100.0 * jnp.sqrt(
            jnp.mean(((predictions - test_derivs) / std) ** 2, axis=0)
        )
        vel_nrmse_pct = float(component_nrmse_pct[0])
        acc_nrmse_pct = float(component_nrmse_pct[1])
        test_nrmse_pct = float(100.0 * nrmse(predictions, test_derivs, normalizer=std))
        print(f"Exp7 test derivative NRMSE: {test_nrmse_pct:.2f}%")
        ts = jnp.arange(predictions.shape[0], dtype=jnp.float32)
        vel_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 0], predictions[:, 0]], axis=1),
            labels=VELOCITY_COMPARISON_LABELS,
            title=f"Exp7 Velocity Predictions (NRMSE={vel_nrmse_pct:.2f}%)",
            styles=("solid", "--"),
        )
        _save_fig(vel_ax, plot_dir, "velocity_predictions.png")
        acc_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 1], predictions[:, 1]], axis=1),
            labels=ACCELERATION_COMPARISON_LABELS,
            title=f"Exp7 Acceleration Predictions (NRMSE={acc_nrmse_pct:.2f}%)",
            styles=("solid", "--"),
        )
        _save_fig(acc_ax, plot_dir, "acceleration_predictions.png")
        save_npz_bundle(
            plot_dir / "test_predictions.npz",
            ts=ts,
            states=test_derivs,
            forcing=test_controls,
            prediction=predictions,
        )

        csv_path = plot_dir / "metrics.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["test_nrmse_percent", test_nrmse_pct])

    print("Exp7 training steps:", len(history))


# %%
if __name__ == "__main__":
    print("Training Exp7 single-network nonlinear MSD...")
    main()
