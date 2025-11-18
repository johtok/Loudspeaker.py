#!/usr/bin/env python3
"""Exp6: Nonlinear MSD identification using ensemble neural networks."""

#%%
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.metrics import mse
from loudspeaker.plotting import plot_phase_fan, plot_timeseries_bundle


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


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


def _true_matrix(config: NonlinearMSDConfig, state: jnp.ndarray) -> jnp.ndarray:
    pos, _ = state
    return jnp.array(
        [
            [0.0, 1.0, 0.0],
            [
                (-config.stiffness - config.cubic * pos**2) / config.mass,
                -config.damping / config.mass,
                1.0 / config.mass,
            ],
        ]
    )


def build_training_data(config: NonlinearMSDConfig, key: jr.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if config.dataset_size < 2:
        raise ValueError("dataset_size must be at least 2.")
    key, state_key, control_key = jr.split(key, 3)
    states = config.state_scale * jr.normal(state_key, (config.dataset_size, 2))
    controls = config.control_scale * jr.normal(control_key, (config.dataset_size, 1))
    derivatives = jax.vmap(lambda s, u: _true_derivative(config, s, u))(states, controls)
    matrices = jax.vmap(lambda s: _true_matrix(config, s))(states)
    return states, controls, derivatives, matrices


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
    states, controls, derivatives, matrices = build_training_data(config, key)

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
        return mse(preds, batch_derivs)

    loss_grad = eqx.filter_value_and_grad(loss_fn)

    @eqx.filter_jit
    def step(
        current_model: EnsembleMSD,
        current_opt_state: optax.OptState,
        batch: Batch,
    ) -> tuple[EnsembleMSD, optax.OptState, jnp.ndarray]:
        (loss_val, grads) = loss_grad(current_model, batch)
        updates, new_state = optimizer.update(grads, current_opt_state, eqx.filter(current_model, eqx.is_array))
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
        state_mse = float(mse(preds, test_derivs))
        param_mse = float(jnp.mean((matrices_pred - test_matrices) ** 2))
        return state_mse, param_mse, preds, matrices_pred

    if test_size > 0:
        test_state_mse, param_mse, preds, matrices_pred = evaluate(model)
        print("Exp6 test derivative MSE:", test_state_mse)
        print("Exp6 parameter matrix MSE:", param_mse)
        ts = jnp.arange(test_derivs.shape[0], dtype=jnp.float32)
        plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 0], preds[:, 0]], axis=1),
            labels=("target velocity", "pred velocity"),
            title="Exp6 Velocity Predictions",
        )
        plot_timeseries_bundle(
            ts,
            jnp.stack([test_derivs[:, 1], preds[:, 1]], axis=1),
            labels=("target acceleration", "pred acceleration"),
            title="Exp6 Acceleration Predictions",
        )
    print("Exp6 training steps:", len(history))


#%%
if __name__ == "__main__":
    print("Training Exp6 nonlinear MSD ensemble...")
    main()
