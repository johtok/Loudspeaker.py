from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Tuple

from matplotlib.axes import Axes

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve

from .loudspeaker_sim import LoudspeakerConfig
from .metrics import mse
from .models import (
    LinearLoudspeakerModel,
    LinearMSDModel,
    AugmentedMSDModel,
    _LinearModel,
)
from .plotting import plot_loss as plot_loss_curve
from .plotting import plot_residuals, plot_trajectory
from .testsignals import ControlSignal, build_control_signal


ControlBuilder = Callable[[jnp.ndarray, jnp.ndarray], ControlSignal]
Batch = Tuple[jnp.ndarray, jnp.ndarray]


@dataclass
class NeuralODE:
    """Container bundling a neural ODE model, loss, and solver metadata."""

    model: _LinearModel
    loss_fn: Callable[[_LinearModel, Batch | None], jnp.ndarray]
    optimizer: optax.GradientTransformation
    ts: jnp.ndarray
    initial_state: jnp.ndarray
    dt: float
    num_steps: int
    control_builder: ControlBuilder = build_control_signal
    solver: Tsit5 | None = None
    stepsize_controller: PIDController | None = None
    rtol: float = 1e-5
    atol: float = 1e-5
    history: list[float] = field(default_factory=list)

    def __post_init__(self: "NeuralODE") -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        ts_array = jnp.asarray(self.ts, dtype=jnp.float32)
        if ts_array.ndim != 1 or ts_array.size == 0:
            raise ValueError("ts must be a non-empty 1D array.")
        self.ts = ts_array
        self.initial_state = jnp.asarray(self.initial_state, dtype=jnp.float32)


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

    def vf(t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        force = forcing.evaluate(t)
        inputs = jnp.concatenate([y, jnp.array([force], dtype=jnp.float32)])
        result = model(inputs)
        return result.astype(y.dtype)

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

        def sample_loss(
            forcing_values: jnp.ndarray, target_values: jnp.ndarray
        ) -> jnp.ndarray:
            length = forcing_values.shape[0]
            time_grid = ts[:length]
            control = control_builder(time_grid, forcing_values)
            prediction = _solve(model, time_grid, control)
            return _loss(prediction, target_values[:length])

        losses = jax.vmap(sample_loss)(batch_forcing, batch_reference)
        return jnp.mean(losses)

    return loss_fn


def _batch_iterator(
    dataloader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]] | None,
) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray] | None]:
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
    def step(
        model: _LinearModel,
        opt_state: optax.OptState,
        batch: Tuple[jnp.ndarray, jnp.ndarray] | None,
    ) -> tuple[_LinearModel, optax.OptState, jnp.ndarray]:
        loss, grads = loss_and_grad(model, batch)
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for _ in range(num_steps):
        batch = next(batches)
        model, opt_state, loss = step(model, opt_state, batch)
        history.append(float(loss))
    return model, history


def train_neural_ode(
    neural_ode: NeuralODE,
    dataloader: Iterable[Batch] | None,
) -> NeuralODE:
    """Train the wrapped neural ODE model and store the loss history."""

    trained_model, history = train_model(
        neural_ode.model,
        neural_ode.loss_fn,
        neural_ode.optimizer,
        neural_ode.num_steps,
        dataloader,
    )
    neural_ode.model = trained_model
    neural_ode.history = history
    return neural_ode


def predict_neural_ode(
    neural_ode: NeuralODE,
    dataloader: Iterable[Batch],
    *,
    max_batches: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Run the trained neural ODE on samples from a dataloader."""

    if dataloader is None:
        raise ValueError("predict_neural_ode requires a dataloader.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")

    iterator = iter(dataloader)
    predictions: list[jnp.ndarray] = []
    targets: list[jnp.ndarray] = []
    batches_consumed = 0
    while max_batches is None or batches_consumed < max_batches:
        try:
            forcing_batch, reference_batch = next(iterator)
        except StopIteration:
            break

        if forcing_batch.ndim == 1:
            forcing_batch = forcing_batch[None, ...]
            reference_batch = reference_batch[None, ...]
        if forcing_batch.shape[0] != reference_batch.shape[0]:
            raise ValueError("Forcing/reference batch size mismatch.")

        for forcing_values, target_values in zip(forcing_batch, reference_batch):
            length = forcing_values.shape[0]
            if length > neural_ode.ts.shape[0]:
                raise ValueError("Sample length exceeds configured time grid.")

            time_grid = neural_ode.ts[:length]
            control = neural_ode.control_builder(time_grid, forcing_values)
            prediction = solve_with_model(
                neural_ode.model,
                time_grid,
                control,
                neural_ode.initial_state,
                neural_ode.dt,
                solver=neural_ode.solver,
                stepsize_controller=neural_ode.stepsize_controller,
                rtol=neural_ode.rtol,
                atol=neural_ode.atol,
            )
            predictions.append(prediction)
            targets.append(target_values[:length])

        batches_consumed += 1

    if not predictions:
        raise ValueError("predict_neural_ode requires dataloader to yield at least one batch.")

    return jnp.stack(predictions), jnp.stack(targets)


def plot_neural_ode_predictions(
    neural_ode: NeuralODE,
    dataloader: Iterable[Batch],
    *,
    sample_index: int = 0,
    max_batches: int | None = 1,
    target_labels: Iterable[str] = ("reference position", "reference velocity"),
    prediction_labels: Iterable[str] = ("predicted position", "predicted velocity"),
    residual_labels: Iterable[str] | None = None,
    title: str | None = "Neural ODE Prediction",
) -> tuple[Axes, Axes]:
    """Plot reference vs. predicted trajectories for a sample from the dataloader."""

    predictions, targets = predict_neural_ode(
        neural_ode,
        dataloader,
        max_batches=max_batches,
    )
    if sample_index < 0 or sample_index >= predictions.shape[0]:
        raise IndexError("sample_index is out of bounds for the collected predictions.")

    prediction = predictions[sample_index]
    target = targets[sample_index]
    ts = neural_ode.ts[: prediction.shape[0]]

    trajectory_ax = plot_trajectory(
        ts,
        target,
        labels=target_labels,
        title=title,
    )
    plot_trajectory(
        ts,
        prediction,
        labels=prediction_labels,
        ax=trajectory_ax,
        title=None,
    )
    labels_for_residuals = residual_labels or target_labels
    residual_ax = plot_residuals(ts, target, prediction, labels=labels_for_residuals)
    return trajectory_ax, residual_ax


def plot_neural_ode_loss(
    neural_ode: NeuralODE,
    *,
    ax: Axes | None = None,
    title: str = "Neural ODE Training Loss",
) -> Axes:
    """Visualize the stored training loss history for a NeuralODE instance."""

    if not neural_ode.history:
        raise ValueError("NeuralODE has no recorded loss history. Train before plotting.")
    return plot_loss_curve(neural_ode.history, ax=ax, title=title)
