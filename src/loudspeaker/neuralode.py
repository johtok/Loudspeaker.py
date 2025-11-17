from __future__ import annotations

from typing import Callable, Iterable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .metrics import mse
from .msd_sim import MSDConfig
from .testsignals import ControlSignal, build_control_signal


class LinearMSDModel(eqx.Module):
    """Single dense layer without bias (2x3 parameters)."""

    weight: jax.Array

    def __init__(
        self,
        config: MSDConfig,
        perturbation: float = 0.01,
        key: jr.PRNGKey | None = None,
    ):
        base = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [-config.stiffness / config.mass, -config.damping / config.mass, 1.0 / config.mass],
            ],
            dtype=jnp.float64,
        )
        if key is not None:
            base = base + perturbation * jr.normal(key, base.shape)
        self.weight = base

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.weight @ inputs


def solve_with_model(
    model: LinearMSDModel,
    ts: jnp.ndarray,
    forcing: ControlSignal,
    initial_state: jnp.ndarray,
    dt: float,
    solver: Tsit5 | None = None,
) -> jnp.ndarray:
    """Integrate the neural ODE with the provided forcing."""

    solver = solver or Tsit5()

    def vf(t, y, args):
        force = forcing.evaluate(t)
        inputs = jnp.array([y[0], y[1], force])
        return model(inputs)

    sol = diffeqsolve(
        ODETerm(vf),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=initial_state,
        saveat=SaveAt(ts=ts),
    )
    return sol.ys


def norm_loss_fn(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    target_mean = jnp.mean(target, axis=0, keepdims=True)
    target_std = jnp.std(target, axis=0, keepdims=True) + eps
    pred_norm = (pred - target_mean) / target_std
    target_norm = (target - target_mean) / target_std
    return mse(pred_norm, target_norm) / pred.shape[0]


def build_loss_fn(
    ts: jnp.ndarray,
    initial_state: jnp.ndarray,
    dt: float,
    loss_type: str = "mse",
    forcing: ControlSignal | None = None,
    reference: jnp.ndarray | None = None,
) -> Callable[[LinearMSDModel, Tuple[jnp.ndarray, jnp.ndarray] | None], jnp.ndarray]:
    if (forcing is None) ^ (reference is None):
        raise ValueError(
            "If providing a default forcing/reference pair, both must be supplied."
        )

    ts = jnp.asarray(ts)
    default_data = None if forcing is None else (forcing, reference)

    def _loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        if loss_type in {"norm", "norm_mse"}:
            return norm_loss_fn(prediction, target)
        if loss_type == "mse":
            return mse(prediction, target)
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Expected 'mse' or 'norm'/'norm_mse'."
        )

    def _solve(model: LinearMSDModel, control: ControlSignal) -> jnp.ndarray:
        return solve_with_model(model, ts, control, initial_state, dt)

    def loss_fn(
        model: LinearMSDModel,
        batch: Tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if batch is None:
            if default_data is None:
                raise ValueError("loss_fn requires batch data when no defaults are set.")
            control, target = default_data
            prediction = _solve(model, control)
            return _loss(prediction, target)

        batch_forcing, batch_reference = batch
        if batch_forcing.ndim == 1:
            batch_forcing = batch_forcing[None, ...]
            batch_reference = batch_reference[None, ...]

        def sample_loss(forcing_values, target_values):
            control = build_control_signal(ts, forcing_values)
            prediction = _solve(model, control)
            return _loss(prediction, target_values)

        losses = jax.vmap(sample_loss)(batch_forcing, batch_reference)
        return jnp.mean(losses)

    return loss_fn


def train_model(
    model: LinearMSDModel,
    loss_fn: Callable[[LinearMSDModel, Tuple[jnp.ndarray, jnp.ndarray] | None], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    num_steps: int,
    dataloader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]] | None = None,
) -> Tuple[LinearMSDModel, list[float]]:
    history: list[float] = []

    if dataloader is None:
        loss_and_grad = eqx.filter_value_and_grad(lambda m: loss_fn(m, None))

        @eqx.filter_jit
        def step(model, opt_state):
            loss, grads = loss_and_grad(model)
            updates, opt_state = optimizer.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss

        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        for _ in range(num_steps):
            model, opt_state, loss = step(model, opt_state)
            history.append(float(loss))
        return model, history

    dataloader_iter = iter(dataloader)
    loss_and_grad = eqx.filter_value_and_grad(loss_fn)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = loss_and_grad(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(num_steps):
        batch = next(dataloader_iter)
        model, opt_state, loss = step(model, opt_state, batch)
        history.append(float(loss))
    return model, history
