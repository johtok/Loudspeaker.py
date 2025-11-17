#!/usr/bin/env python3
"""Exp4 loudspeaker neural ODE training using shared libs."""

#%%
import os
import sys
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
for path in (SCRIPT_DIR, ROOT_DIR):
    if path not in sys.path:
        sys.path.append(path)

from loudspeaker.data import StaticTrainingStrategy, TrainingStrategy
from loudspeaker.loudspeaker_sim import LoudspeakerConfig
from loudspeaker.metrics import mae, mse
from loudspeaker.models import LinearLoudspeakerModel
from loudspeaker.neuralode import (
    NeuralODE,
    build_loss_fn,
    plot_neural_ode_loss,
    plot_neural_ode_predictions,
    predict_neural_ode,
    solve_with_model,
    train_neural_ode,
)
from loudspeaker.testsignals import pink_noise_control


jax.config.update("jax_enable_x64", True)


Batch = Tuple[jnp.ndarray, jnp.ndarray]


def _single_batch_loader(batch: Batch) -> Iterable[Batch]:
    yield batch


def _build_loudspeaker_dataset(
    config: LoudspeakerConfig,
    dataset_size: int,
    key: jr.PRNGKey,
    band: Tuple[float, float] | None = (20.0, 1000.0),
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing_values = []
    reference_states = []
    rng = key
    for _ in range(dataset_size):
        rng, forcing_key = jr.split(rng)
        control = pink_noise_control(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            band=band,
        )
        forcing_values.append(control.values)
        sim = solve_with_model(
            LinearLoudspeakerModel(config),
            ts,
            control,
            config.initial_state,
            config.dt,
        )
        reference_states.append(sim)
    return ts, jnp.stack(forcing_values), jnp.stack(reference_states)


#%%
def main(
    optimizer_factory=optax.sgd,
    loss: str = "mse",
    num_samples: int = 512,
    dataset_size: int = 64,
    batch_size: int = 4,
    num_steps: int = 200,
    strategy: TrainingStrategy | None = None,
):
    config = LoudspeakerConfig(num_samples=num_samples)
    key = jr.PRNGKey(8)
    data_key, model_key = jr.split(key)

    ts, forcing_values, reference_states = _build_loudspeaker_dataset(
        config=config,
        dataset_size=dataset_size,
        key=data_key,
    )

    if strategy is None:
        strategy = StaticTrainingStrategy(steps=num_steps)
    total_steps = strategy.total_steps

    def dataloader():
        dataset_size = forcing_values.shape[0]
        perm = jnp.arange(dataset_size)
        while True:
            perm = jr.permutation(model_key, perm)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                if end <= dataset_size:
                    idx = perm[start:end]
                    yield forcing_values[idx], reference_states[idx]

    model = LinearLoudspeakerModel(config=config, key=model_key)
    optimizer = optimizer_factory(learning_rate=1e-3)
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

    trained = train_neural_ode(neural_ode, dataloader())

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
        title="Loudspeaker Reference vs Predicted",
        target_labels=(
            "reference displacement",
            "reference velocity",
            "reference coil current",
        ),
        prediction_labels=(
            "predicted displacement",
            "predicted velocity",
            "predicted coil current",
        ),
        residual_labels=(
            "displacement residual",
            "velocity residual",
            "coil current residual",
        ),
    )
    plot_neural_ode_loss(trained)

# %%

if __name__ == "__main__":
    print("Training Exp4 loudspeaker Neural ODE (SGD)...")
    main()
# %%

if __name__ == "__main__":
    print("Training Exp4 loudspeaker Neural ODE (Adam)...")
    main(optimizer_factory=optax.adam)

# %%
