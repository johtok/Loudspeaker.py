#!/usr/bin/env python3
"""Nonlinear loudspeaker fit using nonlinear NN (taxonomy 1.2.2.2)."""

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
    / "exp_1_2_2_2_nonlinear_loudspeaker_fit_using_nonlinear_nn"
)

from loudspeaker import LabelSpec
from loudspeaker.data import build_nonlinear_loudspeaker_training_data
from loudspeaker.io import save_npz_bundle
from loudspeaker.loudspeaker_sim import NonlinearLoudspeakerConfig
from loudspeaker.metrics import mse, nrmse
from loudspeaker.plotting import plot_loss, plot_timeseries_bundle, save_figure

jax.config.update("jax_enable_x64", True)

Batch = Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]

COIL_CURRENT = LabelSpec("Coil current", "A", "i")
EDDY_CURRENT = LabelSpec("Eddy current", "A", "i_2")
DISPLACEMENT = LabelSpec("Cone displacement", "m", "x")
VELOCITY = LabelSpec("Cone velocity", "m/s", "v")
STATE_LABELS = (
    COIL_CURRENT.raw(),
    EDDY_CURRENT.raw(),
    DISPLACEMENT.raw(),
    VELOCITY.raw(),
)


class SingleNN(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear

    def __init__(self, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.layer1 = eqx.nn.Linear(5, 64, key=k1)
        self.layer2 = eqx.nn.Linear(64, 64, key=k2)
        self.layer3 = eqx.nn.Linear(64, 4, key=k3)

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
    batch_size: int = 128,
    dataset_size: int = 16,
    num_samples: int = 512,
    num_steps: int = 1500,
    train_fraction: float = 0.8,
):
    config = NonlinearLoudspeakerConfig(
        num_samples=num_samples,
        sample_rate=48000.0,
        use_full_model=True,
    )
    key = jr.PRNGKey(7)
    data_key, model_key, loader_key = jr.split(key, 3)

    states, controls, derivatives = build_nonlinear_loudspeaker_training_data(
        config,
        dataset_size=dataset_size,
        key=data_key,
    )

    total_samples = states.shape[0]
    if total_samples < 2:
        raise ValueError("Training set must contain at least two samples.")
    train_size = max(1, int(train_fraction * total_samples))
    if train_size >= total_samples:
        train_size = total_samples - 1
    test_size = total_samples - train_size

    train_states, test_states = states[:train_size], states[train_size:]
    train_controls, test_controls = controls[:train_size], controls[train_size:]
    train_derivs, test_derivs = derivatives[:train_size], derivatives[train_size:]

    loader = make_dataloader(
        train_states,
        train_controls,
        train_derivs,
        batch_size=batch_size,
        key=loader_key,
    )

    model = SingleNN(key=model_key)
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
        OUT_DIR
        / f"exp8_{optimizer_factory.__name__}_bs_{batch_size}_steps_{num_steps}_ds_{dataset_size}"
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    if history:
        loss_ax = plot_loss(history, title="Exp8 Training Loss")
        save_figure(loss_ax, plot_dir / "training_loss.png")

    if test_size > 0:
        predictions = jax.vmap(model)(test_states, test_controls)
        std = jnp.std(test_derivs, axis=0, keepdims=True) + 1e-8
        component_nrmse_pct = 100.0 * jnp.sqrt(
            jnp.mean(((predictions - test_derivs) / std) ** 2, axis=0)
        )
        test_nrmse_pct = float(100.0 * nrmse(predictions, test_derivs, normalizer=std))
        print(f"Exp8 test derivative NRMSE: {test_nrmse_pct:.2f}%")

        ts = jnp.arange(predictions.shape[0], dtype=jnp.float32)
        primary_ax = plot_timeseries_bundle(
            ts,
            jnp.stack(
                [
                    test_derivs[:, 2],
                    predictions[:, 2],
                    test_derivs[:, 3],
                    predictions[:, 3],
                ],
                axis=1,
            ),
            labels=(
                f"Target {DISPLACEMENT.raw()}",
                f"Predicted {DISPLACEMENT.raw()}",
                f"Target {VELOCITY.raw()}",
                f"Predicted {VELOCITY.raw()}",
            ),
            styles=("solid", "--", "solid", "--"),
            title="Displacement & Velocity Derivatives",
        )
        save_figure(primary_ax, plot_dir / "motion_derivatives.png")

        electrical_ax = plot_timeseries_bundle(
            ts,
            jnp.stack(
                [
                    test_derivs[:, 0],
                    predictions[:, 0],
                    test_derivs[:, 1],
                    predictions[:, 1],
                ],
                axis=1,
            ),
            labels=(
                f"Target {COIL_CURRENT.raw()}",
                f"Predicted {COIL_CURRENT.raw()}",
                f"Target {EDDY_CURRENT.raw()}",
                f"Predicted {EDDY_CURRENT.raw()}",
            ),
            styles=("solid", "--", "solid", "--"),
            title="Coil & Eddy Current Derivatives",
        )
        save_figure(electrical_ax, plot_dir / "electrical_derivatives.png")

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
            for idx, label in enumerate(STATE_LABELS):
                writer.writerow(
                    [f"{label}_nrmse_percent", float(component_nrmse_pct[idx])]
                )

    print("Exp8 training steps:", len(history))


# %%
if __name__ == "__main__":
    print("Training Exp8 nonlinear loudspeaker network...")
    main()
