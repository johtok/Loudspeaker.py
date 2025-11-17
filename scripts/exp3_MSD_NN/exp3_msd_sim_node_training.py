#!/usr/bin/env python3
"""Baseline Exp3 MSD neural ODE training with shared libs."""

#%%
import os
import sys
from typing import Iterable, Tuple

import jax
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
    NeuralODE,
    build_loss_fn,
    plot_neural_ode_loss,
    plot_neural_ode_predictions,
    predict_neural_ode,
    train_neural_ode,
)

jax.config.update("jax_enable_x64", True)

Batch = Tuple[jax.Array, jax.Array]


def _single_batch_loader(batch: Batch) -> Iterable[Batch]:
    yield batch


#%%
def main(
    optimizer_factory=optax.sgd,
    loss: str = "mse",
    num_samples: int = 20,
    dataset_size: int = 128,
    batch_size: int = 8,
    num_steps: int = 400,
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

    neural_ode = NeuralODE(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=total_steps,
    )

    trained = train_neural_ode(neural_ode, data_loader)

    eval_batch = (forcing_values[:1], reference_states[:1])
    predictions, targets = predict_neural_ode(trained, _single_batch_loader(eval_batch), max_batches=1)
    eval_prediction = predictions[0]
    eval_reference = targets[0]

    print("Final MAE:", float(mae(eval_prediction, eval_reference)))
    print("Final MSE:", float(mse(eval_prediction, eval_reference)))

    plot_neural_ode_predictions(
        trained,
        _single_batch_loader(eval_batch),
        max_batches=1,
        title="Reference vs Predicted Trajectory",
    )
    plot_neural_ode_loss(trained)


#%%
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
