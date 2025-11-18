#!/usr/bin/env python3
"""Nonlinear MSD sysid using ensemble NN (taxonomy 3.2.2.1)."""

# %%
import csv
import os
import sys
from typing import Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
OUT_DIR = os.path.join(
    ROOT_DIR,
    "out",
    "3_blackbox_sysid",
    "exp_3_2_2_1_nonlinear_msd_sysid_using_ensemble_nn",
)
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

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


def _save_fig(ax, folder: str, filename: str) -> None:
    save_figure(ax, os.path.join(folder, filename))


class _MiniMLP(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, in_size: int, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.layer1 = eqx.nn.Linear(in_size, 10, key=k1)
        self.layer2 = eqx.nn.Linear(10, 10, key=k2)
        self.layer3 = eqx.nn.Linear(10, 1, key=k3)

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.tanh(self.layer1(inputs))
        x = jnp.tanh(self.layer2(x))
        return self.layer3(x)


class EnsembleMSD(eqx.Module):
    nets: tuple[_MiniMLP, ...]
    rows: int
    cols: int

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        outputs = jnp.stack([net(inputs).squeeze(-1) for net in self.nets])
        return outputs.reshape(self.rows, self.cols)


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
    num_steps: int = 1000,
    train_fraction: float = 0.8,
):
    key = jr.PRNGKey(6)
    states, controls, derivatives, matrices = build_nonlinear_msd_training_data(
        config,
        key,
        include_matrices=True,
    )

    train_size = max(1, int(train_fraction * config.dataset_size))
    if train_size >= config.dataset_size:
        train_size = config.dataset_size - 1
    test_size = config.dataset_size - train_size

    train_states, test_states = states[:train_size], states[train_size:]
    train_controls, test_controls = controls[:train_size], controls[train_size:]
    train_derivs, test_derivs = derivatives[:train_size], derivatives[train_size:]
    test_matrices = matrices[train_size:]

    loader = make_dataloader(
        train_states,
        train_controls,
        train_derivs,
        batch_size=batch_size,
        key=jr.PRNGKey(99),
    )

    train_mean = jnp.mean(train_derivs, axis=0, keepdims=True)
    train_std = jnp.std(train_derivs, axis=0, keepdims=True) + 1e-8

    net_keys = jr.split(jr.PRNGKey(123), 6)
    nets = tuple(_MiniMLP(3, key=k) for k in net_keys)
    model = EnsembleMSD(nets=nets, rows=2, cols=3)

    optimizer = optimizer_factory(learning_rate=3e-3)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    def loss_fn(current_model: EnsembleMSD, batch: Batch) -> jnp.ndarray:
        batch_states, batch_controls, batch_derivs = batch
        inputs = jnp.concatenate([batch_states, batch_controls], axis=1)
        matrices_pred = jax.vmap(current_model)(inputs)
        preds = jnp.einsum("bij,bj->bi", matrices_pred, inputs)
        preds_norm = (preds - train_mean) / train_std
        targets_norm = (batch_derivs - train_mean) / train_std
        return mse(preds_norm, targets_norm)

    loss_grad = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(
        current_model: EnsembleMSD,
        current_opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[EnsembleMSD, optax.OptState, jnp.ndarray]:
        (loss_val, grads) = loss_grad(current_model, batch)
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

    def evaluate(current_model: EnsembleMSD):
        inputs = jnp.concatenate([test_states, test_controls], axis=1)
        matrices_pred = jax.vmap(current_model)(inputs)
        preds = jnp.einsum("bij,bj->bi", matrices_pred, inputs)
        component_scaled = (preds - test_derivs) / train_std
        component_nrmse_pct = 100.0 * jnp.sqrt(jnp.mean(component_scaled**2, axis=0))
        state_nrmse_pct = float(100.0 * nrmse(preds, test_derivs, normalizer=train_std))
        param_mse = float(jnp.mean((matrices_pred - test_matrices) ** 2))
        return state_nrmse_pct, param_mse, preds, matrices_pred, component_nrmse_pct

    plot_dir = os.path.join(
        OUT_DIR,
        f"exp6_{optimizer_factory.__name__}_bs_{batch_size}_steps_{num_steps}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    if history:
        loss_ax = plot_loss(history, title="Exp6 Training Loss")
        _save_fig(loss_ax, plot_dir, "training_loss.png")

    if test_size > 0:
        test_state_nrmse_pct, param_mse, preds, matrices_pred, component_nrmse_pct = (
            evaluate(model)
        )
        vel_nrmse_pct = float(component_nrmse_pct[0])
        acc_nrmse_pct = float(component_nrmse_pct[1])
        print(f"Exp6 test derivative NRMSE: {test_state_nrmse_pct:.2f}%")
        print("Exp6 parameter matrix MSE:", param_mse)
        ts = jnp.arange(test_derivs.shape[0], dtype=jnp.float32)
        vel_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 0], preds[:, 0]], axis=1),
            labels=VELOCITY_COMPARISON_LABELS,
            title=f"Exp6 Velocity Predictions (NRMSE={vel_nrmse_pct:.2f}%)",
            styles=("solid", "--"),
        )
        _save_fig(vel_ax, plot_dir, "velocity_predictions.png")
        acc_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 1], preds[:, 1]], axis=1),
            labels=ACCELERATION_COMPARISON_LABELS,
            title=f"Exp6 Acceleration Predictions (NRMSE={acc_nrmse_pct:.2f}%)",
            styles=("solid", "--"),
        )
        _save_fig(acc_ax, plot_dir, "acceleration_predictions.png")
        save_npz_bundle(
            os.path.join(plot_dir, "test_predictions.npz"),
            ts=ts,
            states=test_derivs,
            forcing=test_controls,
            prediction=preds,
            matrices=matrices_pred,
        )

        csv_path = os.path.join(plot_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["test_state_nrmse_percent", test_state_nrmse_pct])
            writer.writerow(["param_mse", param_mse])
    print("Exp6 training steps:", len(history))


# %%
if __name__ == "__main__":
    print("Training Exp6 nonlinear MSD ensemble...")
    main()
