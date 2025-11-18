#!/usr/bin/env python3
"""Exp7: Nonlinear MSD identification using a single neural network vector field."""

#%%
import os
import sys
from dataclasses import dataclass
from typing import Iterator, Tuple

import csv
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUT_DIR = os.path.join(ROOT_DIR, "out", "exp7_nonlinear_msd_single")
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.metrics import mse
from loudspeaker.plotting import plot_timeseries_bundle, plot_loss


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _save_fig(ax, folder: str, filename: str) -> None:
    if isinstance(ax, np.ndarray):
        fig = ax.ravel()[0].figure
    else:
        fig = ax.figure
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class NonlinearMSDConfig:
    mass: float = 0.05
    stiffness: float = 100.0
    damping: float = 0.4
    cubic: float = 5.0
    state_scale: float = 1.0
    control_scale: float = 1.0
    dataset_size: int = 2048


def _true_derivative(config: NonlinearMSDConfig, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    pos, vel = state
    force = control[0]
    acc = (
        force
        - config.damping * vel
        - config.stiffness * pos
        - config.cubic * (pos**3)
    ) / config.mass
    return jnp.array([vel, acc])


def build_training_data(config: NonlinearMSDConfig, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if config.dataset_size < 2:
        raise ValueError("dataset_size must be at least 2.")
    key, state_key, control_key = jr.split(key, 3)
    states = config.state_scale * jr.normal(state_key, (config.dataset_size, 2))
    controls = config.control_scale * jr.normal(control_key, (config.dataset_size, 1))
    derivatives = jax.vmap(lambda s, u: _true_derivative(config, s, u))(states, controls)
    return states, controls, derivatives


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
    states, controls, derivatives = build_training_data(config, key)

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
        updates, new_state = optimizer.update(grads, current_opt_state, eqx.filter(current_model, eqx.is_array))
        new_model = eqx.apply_updates(current_model, updates)
        return new_model, new_state, loss_val

    history = []
    for _ in range(num_steps):
        batch = next(loader)
        model, opt_state, loss_value = step(model, opt_state, batch)
        history.append(float(loss_value))

    plot_dir = os.path.join(
        OUT_DIR,
        f"exp7_{optimizer_factory.__name__}_bs_{batch_size}_steps_{num_steps}",
    )
    os.makedirs(plot_dir, exist_ok=True)

    if history:
        loss_ax = plot_loss(history, title="Exp7 Training Loss")
        _save_fig(loss_ax, plot_dir, "training_loss.png")

    if test_size > 0:
        predictions = jax.vmap(model)(test_states, test_controls)
        mean = jnp.mean(test_derivs, axis=0, keepdims=True)
        std = jnp.std(test_derivs, axis=0, keepdims=True) + 1e-8
        preds_norm = (predictions - mean) / std
        targets_norm = (test_derivs - mean) / std
        test_mse = float(mse(preds_norm, targets_norm))
        print("Exp7 test derivative MSE:", test_mse)
        ts = jnp.arange(predictions.shape[0], dtype=jnp.float32)
        vel_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 0], predictions[:, 0]], axis=1),
            labels=("target velocity", "pred velocity"),
            title="Exp7 Velocity Predictions",
            styles=("solid", "--"),
        )
        _save_fig(vel_ax, plot_dir, "velocity_predictions.png")
        acc_ax = plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 1], predictions[:, 1]], axis=1),
            labels=("target acceleration", "pred acceleration"),
            title="Exp7 Acceleration Predictions",
            styles=("solid", "--"),
        )
        _save_fig(acc_ax, plot_dir, "acceleration_predictions.png")

        csv_path = os.path.join(plot_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["test_mse", test_mse])

    print("Exp7 training steps:", len(history))


#%%
if __name__ == "__main__":
    print("Training Exp7 single-network nonlinear MSD...")
    main()
