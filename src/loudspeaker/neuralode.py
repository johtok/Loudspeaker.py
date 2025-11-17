from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from .metrics import mse
from .msd_sim import MSDConfig
from .testsignals import ControlSignal, build_control_signal


ControlBuilder = Callable[[jnp.ndarray, jnp.ndarray], ControlSignal]


class _LinearModel(eqx.Module):
    weight: jax.Array

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.weight @ inputs


class LinearMSDModel(_LinearModel):
    """Single dense layer without bias (2x3 parameters)."""

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
            dtype=jnp.float32,
        )
        if key is not None:
            base = base + perturbation * jr.normal(key, base.shape)
        super().__init__(weight=base)


@dataclass(frozen=True)
class LoudspeakerConfig:
    moving_mass: float = 0.02  # kg
    compliance: float = 1.8e-4  # m/N
    damping: float = 0.4  # N*s/m
    motor_force: float = 7.0  # N/A
    voice_coil_resistance: float = 6.0  # Ohm
    voice_coil_inductance: float = 0.5e-3  # H
    sample_rate: float = 48000.0
    num_samples: int = 512
    initial_state: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    )

    @property
    def stiffness(self) -> float:
        return 1.0 / self.compliance

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        return float(self.num_samples - 1) * self.dt


class LinearLoudspeakerModel(_LinearModel):
    """Three-state loudspeaker model capturing cone and coil dynamics."""

    def __init__(
        self,
        config: LoudspeakerConfig,
        perturbation: float = 0.01,
        key: jr.PRNGKey | None = None,
    ):
        inv_mass = 1.0 / config.moving_mass
        inv_inductance = 1.0 / config.voice_coil_inductance
        base = jnp.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [-config.stiffness * inv_mass, -config.damping * inv_mass, config.motor_force * inv_mass, 0.0],
                [0.0, -config.motor_force * inv_inductance, -config.voice_coil_resistance * inv_inductance, inv_inductance],
            ],
            dtype=jnp.float32,
        )
        if key is not None:
            base = base + perturbation * jr.normal(key, base.shape)
        super().__init__(weight=base)


class ReservoirMSDModel(_LinearModel):
    """Random linear reservoir that augments MSD dynamics with extra states."""

    def __init__(
        self,
        state_size: int,
        *,
        key: jr.PRNGKey,
        scale: float = 0.1,
    ):
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        weight_shape = (state_size, state_size + 1)
        self_key, _ = jr.split(key)
        base = scale * jr.normal(self_key, weight_shape, dtype=jnp.float32)
        super().__init__(weight=base)


def solve_with_model(
    model: _LinearModel,
    ts: jnp.ndarray,
    forcing: ControlSignal,
    initial_state: jnp.ndarray,
    dt: float,
    solver: Tsit5 | None = None,
    *,
    stepsize_controller: PIDController | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> jnp.ndarray:
    """Integrate the neural ODE with the provided forcing."""

    solver = solver or Tsit5()
    controller = stepsize_controller or PIDController(rtol=rtol, atol=atol)

    state_dim = model.weight.shape[0]
    input_dim = model.weight.shape[1]
    expected_input = state_dim + 1
    if input_dim != expected_input:
        raise ValueError(
            f"Model expects {input_dim} inputs but only supports state_dim + force ({expected_input})."
        )

    def vf(t, y, args):
        force = forcing.evaluate(t)
        inputs = jnp.concatenate([y, jnp.array([force], dtype=jnp.float32)])
        return model(inputs)

    sol = diffeqsolve(
        ODETerm(vf),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt,
        y0=initial_state,
        saveat=SaveAt(ts=ts),
        stepsize_controller=controller,
    )
    return sol.ys


@eqx.filter_jit
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
    *,
    control_builder: ControlBuilder = build_control_signal,
    solver: Tsit5 | None = None,
    stepsize_controller: PIDController | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> Callable[[_LinearModel, Tuple[jnp.ndarray, jnp.ndarray] | None], jnp.ndarray]:
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

    def _solve(
        model: _LinearModel,
        time_grid: jnp.ndarray,
        control: ControlSignal,
    ) -> jnp.ndarray:
        return solve_with_model(
            model,
            time_grid,
            control,
            initial_state,
            dt,
            solver=solver,
            stepsize_controller=stepsize_controller,
            rtol=rtol,
            atol=atol,
        )

    def loss_fn(
        model: LinearMSDModel,
        batch: Tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if batch is None:
            if default_data is None:
                raise ValueError("loss_fn requires batch data when no defaults are set.")
            control, target = default_data
            prediction = _solve(model, ts, control)
            return _loss(prediction, target)

        batch_forcing, batch_reference = batch
        if batch_forcing.ndim == 1:
            batch_forcing = batch_forcing[None, ...]
            batch_reference = batch_reference[None, ...]

        def sample_loss(forcing_values, target_values):
            length = forcing_values.shape[0]
            time_grid = ts[:length]
            control = control_builder(time_grid, forcing_values)
            prediction = _solve(model, time_grid, control)
            return _loss(prediction, target_values[:length])

        losses = jax.vmap(sample_loss)(batch_forcing, batch_reference)
        return jnp.mean(losses)

    return loss_fn


def _batch_iterator(dataloader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]] | None):
    if dataloader is None:
        while True:
            yield None

    iterator = iter(dataloader)
    while True:
        yield next(iterator)


def train_model(
    model: _LinearModel,
    loss_fn: Callable[[_LinearModel, Tuple[jnp.ndarray, jnp.ndarray] | None], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    num_steps: int,
    dataloader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]] | None = None,
) -> Tuple[LinearMSDModel, list[float]]:
    history: list[float] = [0.0]

    loss_and_grad = eqx.filter_value_and_grad(
        lambda current_model, batch: loss_fn(current_model, batch)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    batches = _batch_iterator(dataloader)

    @eqx.filter_jit
    def step(model, opt_state, batch):
        loss, grads = loss_and_grad(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(num_steps):
        batch = next(batches)
        model, opt_state, loss = step(model, opt_state, batch)
        history.append(float(loss))
    return model, history
