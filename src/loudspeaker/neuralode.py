from __future__ import annotations

from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .metrics import mse
from .msd_sim import MSDConfig
from .testsignals import ControlSignal


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
    pred_std = jnp.std(pred, axis=0, keepdims=True) + eps
    target_std = jnp.std(target, axis=0, keepdims=True) + eps
    pred_norm = pred / pred_std
    target_norm = target / target_std
    return mse(pred_norm, target_norm)


def build_loss_fn(
    ts: jnp.ndarray,
    forcing: ControlSignal,
    reference: jnp.ndarray,
    initial_state: jnp.ndarray,
    dt: float,
    loss_type: str = "mse",
) -> Callable[[LinearMSDModel], jnp.ndarray]:
    def loss_fn(model: LinearMSDModel) -> jnp.ndarray:
        prediction = solve_with_model(model, ts, forcing, initial_state, dt)
        if loss_type in {"norm", "norm_mse"}:
            return norm_loss_fn(prediction, reference)
        if loss_type == "mse":
            return mse(prediction, reference)
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Expected 'mse' or 'norm'/'norm_mse'."
        )

    return loss_fn


def train_model(
    model: LinearMSDModel,
    loss_fn: Callable[[LinearMSDModel], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    num_steps: int,
) -> Tuple[LinearMSDModel, list[float]]:
    loss_and_grad = eqx.filter_value_and_grad(loss_fn)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state):
        loss, grads = loss_and_grad(model)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    history = []
    for _ in range(num_steps):
        model, opt_state, loss = step(model, opt_state)
        history.append(float(loss))
    return model, history
