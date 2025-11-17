#!/usr/bin/env python3
"""Baseline Exp3 MSD neural ODE training with shared libs."""

#%%
import os
import sys

import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.data import (
    StaticTrainingStrategy,
    TrainingStrategy,
    build_msd_dataset,
    msd_dataloader,
)
from loudspeaker.metrics import mae, mse
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.neuralode import (
    LinearMSDModel,
    build_loss_fn,
    solve_with_model,
    train_model,
)
from loudspeaker.plotting import plot_loss, plot_residuals, plot_trajectory
from loudspeaker.testsignals import build_control_signal


#%%
def main(
    optimizer_factory=optax.sgd,
    loss="mse",
    num_samples=20,
    dataset_size=128,
    batch_size=8,
    num_steps=400,
    strategy: TrainingStrategy | None = None,
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

    if strategy is None:
        strategy = StaticTrainingStrategy(steps=num_steps)
    total_steps = strategy.total_steps

    data_loader = msd_dataloader(
        forcing_values,
        reference_states,
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

    trained_model, loss_history = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_steps=total_steps,
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
