#!/usr/bin/env python3
"""Baseline Exp3 MSD neural ODE training with shared libs."""

#%%
import os
import sys

import jax.numpy as jnp
import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.metrics import mae, mse
from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.neuralode import (
    LinearMSDModel,
    build_loss_fn,
    solve_with_model,
    train_model,
)
from loudspeaker.plotting import plot_loss, plot_residuals, plot_trajectory
from loudspeaker.testsignals import build_control_signal, pink_noise_control


def build_msd_dataset(
    config: MSDConfig,
    dataset_size: int,
    key: jr.PRNGKey,
    band: tuple[float, float] = (1.0, 100.0),
):
    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    forcing_values = []
    reference_states = []
    ts = None
    current_key = key
    for _ in range(dataset_size):
        current_key, forcing_key = jr.split(current_key)
        forcing = pink_noise_control(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            band=band,
        )
        ts, reference = simulate_msd_system(config, forcing)
        forcing_values.append(forcing.values)
        reference_states.append(reference)
    if ts is None:
        raise RuntimeError("Failed to generate MSD dataset.")
    return ts, jnp.stack(forcing_values), jnp.stack(reference_states)


def msd_dataloader(
    forcing_values: jnp.ndarray,
    reference_states: jnp.ndarray,
    batch_size: int,
    *,
    key: jr.PRNGKey,
):
    dataset_size = forcing_values.shape[0]
    if batch_size > dataset_size:
        raise ValueError("batch_size cannot exceed dataset size")
    indices = jnp.arange(dataset_size)
    rng = key
    while True:
        rng, perm_key = jr.split(rng)
        perm = jr.permutation(perm_key, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_idx = perm[start:end]
            yield forcing_values[batch_idx], reference_states[batch_idx]
            start = end
            end = start + batch_size


#%%
def main(
    optimizer_factory=optax.sgd,
    loss="mse",
    num_samples=20,
    dataset_size=128,
    batch_size=8,
    num_steps=400,
):
    config = MSDConfig(num_samples=num_samples)
    key = jr.PRNGKey(42)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, forcing_values, reference_states = build_msd_dataset(
        config=config,
        dataset_size=dataset_size,
        key=data_key,
        band=(1.0, 100.0),
    )
    data_loader = msd_dataloader(
        forcing_values,
        reference_states,
        batch_size=batch_size,
        key=loader_key,
    )

    model = LinearMSDModel(config=config, key=model_key)
    optimizer = optimizer_factory(learning_rate=1e-2)
    loss_fn = build_loss_fn(
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        loss_type=loss,
    )

    trained_model, loss_history = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_steps=num_steps,
        dataloader=data_loader,
    )

    eval_forcing = build_control_signal(ts, forcing_values[0])
    eval_reference = reference_states[0]
    predictions = solve_with_model(
        trained_model,
        ts=ts,
        forcing=eval_forcing,
        initial_state=config.initial_state,
        dt=config.dt,
    )

    print("Final MAE:", float(mae(predictions, eval_reference)))
    print("Final MSE:", float(mse(predictions, eval_reference)))

    ax = plot_trajectory(
        ts,
        eval_reference,
        labels=("reference position", "reference velocity"),
        title="Reference vs Predicted Trajectory",
    )
    plot_trajectory(
        ts,
        predictions,
        labels=("predicted position", "predicted velocity"),
        ax=ax,
        title=None,
    )
    plot_residuals(ts, eval_reference, predictions)
    plot_loss(loss_history)


#%%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader...")
    main()

# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader...")
    main(optimizer_factory=optax.adam)

# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader...")
    main(optimizer_factory=optax.adamw)
# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader...")
    main(optimizer_factory=optax.nadam)
# %%
if __name__ == "__main__":
    print("Training Neural ODE with pink-noise forcing dataloader...")
    main(optimizer_factory=optax.lion)