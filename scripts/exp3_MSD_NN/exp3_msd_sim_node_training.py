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

from loudspeaker.metrics import mae, mse
from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.neuralode import (
    LinearMSDModel,
    build_loss_fn,
    solve_with_model,
    train_model,
)
from loudspeaker.plotting import plot_loss, plot_residuals, plot_trajectory
from loudspeaker.testsignals import pink_noise_control


#%%
def main(opt=optax.sgd,loss="mse",num_samples=20):
    config = MSDConfig(num_samples=num_samples)
    key = jr.PRNGKey(42)

    forcing = pink_noise_control(
        num_samples=config.num_samples,
        dt=config.dt,
        key=key,
        band=(1.0, 100.0),
    )
    ts, reference_states = simulate_msd_system(config, forcing)

    model = LinearMSDModel(config=config, key=jr.PRNGKey(0))
    optimizer = opt(1e-2)
    loss_fn = build_loss_fn(
        ts=ts,
        forcing=forcing,
        reference=reference_states,
        initial_state=config.initial_state,
        dt=config.dt,
        loss_type=loss,
    )

    trained_model, loss_history = train_model(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_steps=400,
    )

    predictions = solve_with_model(
        trained_model,
        ts=ts,
        forcing=forcing,
        initial_state=config.initial_state,
        dt=config.dt,
    )

    print("Final MAE:", float(mae(predictions, reference_states)))
    print("Final MSE:", float(mse(predictions, reference_states)))

    ax = plot_trajectory(
        ts,
        reference_states,
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
    plot_residuals(ts, reference_states, predictions)
    plot_loss(loss_history)


#%%
if __name__ == "__main__":
   opt=optax.sgd
   loss="mse"
   num_samples=20
   print(f"Running {opt.__name__}, with loss={loss} on {num_samples} samples")
   main()

# %%
if __name__ == "__main__":
   opt=optax.sgd
   loss="norm_mse"
   num_samples=20
   print(f"Running {opt.__name__}, with loss={loss} on {num_samples} samples")
   main()

# %%
if __name__ == "__main__":
   opt=optax.adam
   loss="mse"
   num_samples=20
   print(f"Running {opt.__name__}, with loss={loss} on {num_samples} samples")
   main()
# %%
