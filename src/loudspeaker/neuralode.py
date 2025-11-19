from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Tuple,
    TypeAlias,
    TypeVar,
    cast,
)

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from diffrax import ODETerm, PIDController, SaveAt, Tsit5, diffeqsolve
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:  # Prefer JAX-aware progress bars when available.
    import jax_tqdm as _jax_tqdm_module
except ImportError:  # pragma: no cover - fallback to standard tqdm
    _jax_tqdm = None
else:  # pragma: no cover - fallback unused when jax_tqdm present
    _jax_tqdm = getattr(_jax_tqdm_module, "tqdm", None)
from tqdm import tqdm as _standard_tqdm

try:
    from tensorboard.compat.proto.event_pb2 import Event as _TBEvent
    from tensorboard.compat.proto.summary_pb2 import Summary as _TBSummary
    from tensorboard.summary.writer.event_file_writer import (
        EventFileWriter as _TBEventWriter,
    )
except ImportError:  # pragma: no cover - tensorboard optional
    _TBEvent = None
    _TBSummary = None
    _TBEventWriter = None

from .metrics import mse, norm_mse
from .models import _LinearModel
from .plotting import normalize_state_pair, plot_residuals, plot_trajectory
from .plotting import plot_loss as plot_loss_curve
from .testsignals import ControlSignal, build_control_signal

ControlBuilder = Callable[[jnp.ndarray, jnp.ndarray], ControlSignal]
Batch = Tuple[jnp.ndarray, jnp.ndarray]
ModelT = TypeVar("ModelT", bound=_LinearModel)
ScalarLike: TypeAlias = bool | int | float | jax.Array | np.ndarray


@dataclass(frozen=True)
class LossFunction(Generic[ModelT]):
    """Callable loss wrapper exposing reusable value-and-grad computations."""

    fn: Callable[[ModelT, Batch | None], jnp.ndarray]
    value_and_grad_fn: Callable[[ModelT, Batch | None], tuple[jnp.ndarray, Any]]

    def __call__(
        self, model: ModelT, batch: Batch | None
    ) -> jnp.ndarray:  # pragma: no cover - trivial forwarding
        return self.fn(model, batch)

    def value_and_grad(
        self, model: ModelT, batch: Batch | None
    ) -> tuple[jnp.ndarray, Any]:
        return self.value_and_grad_fn(model, batch)


class TensorBoardCallback:
    """TensorBoard scalar/image logger triggered from within JIT via filter_pure_callback."""

    def __init__(self, logdir: str | Path, tag: str = "training/loss") -> None:
        log_path = Path(logdir)
        log_path.mkdir(parents=True, exist_ok=True)
        if _TBEventWriter is None or _TBEvent is None or _TBSummary is None:
            warnings.warn(
                "TensorBoard is not available; TensorBoardCallback will no-op.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._writer = None
        else:
            self._writer = _TBEventWriter(str(log_path))
        self._default_tag = tag

    def _write_scalar(self, tag: str, step: int, value: float) -> None:
        if self._writer is None or _TBEvent is None or _TBSummary is None:
            return
        summary = cast(Any, _TBSummary())
        summary_value = summary.value.add()
        summary_value.tag = tag
        summary_value.simple_value = float(value)
        event = cast(Any, _TBEvent())
        event.wall_time = time.time()
        event.step = step
        event.summary.CopyFrom(summary)
        self._writer.add_event(event)
        self._writer.flush()

    def _write_image(
        self,
        tag: str,
        step: int,
        encoded_image: bytes,
        height: int,
        width: int,
    ) -> None:
        if self._writer is None or _TBEvent is None or _TBSummary is None:
            return
        summary = cast(Any, _TBSummary())
        summary_value = summary.value.add()
        summary_value.tag = tag
        image_summary = summary_value.image
        image_summary.height = height
        image_summary.width = width
        image_summary.colorspace = 3
        image_summary.encoded_image_string = encoded_image
        event = cast(Any, _TBEvent())
        event.wall_time = time.time()
        event.step = step
        event.summary.CopyFrom(summary)
        self._writer.add_event(event)
        self._writer.flush()

    def log_scalar(self, tag: str, step: int | float, value: float) -> None:
        if self._writer is None:
            return
        self._write_scalar(tag, int(step), float(value))

    def log_figure(self, tag: str, step: int | float, figure: Figure) -> None:
        if self._writer is None:
            return
        figure.canvas.draw()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        width, height = figure.canvas.get_width_height()
        self._write_image(tag, int(step), buffer.getvalue(), height, width)
        buffer.close()

    def __call__(self, step: jnp.ndarray, loss: jnp.ndarray) -> None:
        if self._writer is None:
            return

        def _log(step_value, loss_value):
            self._write_scalar(
                self._default_tag,
                int(np.asarray(step_value)),
                float(np.asarray(loss_value)),
            )
            return ()

        eqx.filter_pure_callback(
            _log,
            step,
            loss,
            result_shape_dtypes=(),
        )


class CheckpointManager:
    """Thin wrapper over orbax checkpointing for neural ODE models."""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_to_keep: int = 5,
    ) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        options = ocp.CheckpointManagerOptions(max_to_keep=max_to_keep)
        checkpointer = ocp.PyTreeCheckpointer()
        self._manager = ocp.CheckpointManager(
            str(path),
            checkpointer,
            options=options,
        )

    def save(
        self,
        step: int,
        model: _LinearModel,
        opt_state: optax.OptState,
    ) -> None:
        self._manager.save(step, {"model": model, "optimizer": opt_state})


@dataclass
class NeuralODE:
    """Container bundling a neural ODE model, loss, and solver metadata."""

    model: _LinearModel
    loss_fn: (
        LossFunction[_LinearModel] | Callable[[_LinearModel, Batch | None], jnp.ndarray]
    )
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
    tensorboard_callback: TensorBoardCallback | None = None
    checkpoint_manager: CheckpointManager | None = None
    checkpoint_every: int = 0

    def __post_init__(self: "NeuralODE") -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        ts_array = jnp.asarray(self.ts, dtype=jnp.float32)
        if ts_array.ndim != 1 or ts_array.size == 0:
            raise ValueError("ts must be a non-empty 1D array.")
        self.ts = ts_array
        self.initial_state = jnp.asarray(self.initial_state, dtype=jnp.float32)


@eqx.filter_jit
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

    def vf(t: ScalarLike, y: jnp.ndarray, _args: Any) -> jnp.ndarray:
        force_input = cast(float | jnp.ndarray, t)
        force = forcing.evaluate(force_input)
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
    if sol.ys is None:
        raise RuntimeError("Solver returned no trajectory.")
    return sol.ys


@eqx.filter_jit
def norm_loss_fn(
    pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8
) -> jnp.ndarray:
    """Alias retained for backward compatibility; uses metric.norm_mse."""

    return norm_mse(pred, target, eps=eps)


def build_loss_fn(
    ts: jnp.ndarray,
    initial_state: jnp.ndarray,
    dt: float,
    loss_type: str = "norm_mse",
    forcing: ControlSignal | None = None,
    reference: jnp.ndarray | None = None,
    *,
    control_builder: ControlBuilder = build_control_signal,
    solver: Tsit5 | None = None,
    stepsize_controller: PIDController | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> LossFunction[_LinearModel]:
    if (forcing is None) ^ (reference is None):
        raise ValueError(
            "If providing a default forcing/reference pair, both must be supplied."
        )

    ts = jnp.asarray(ts)
    default_data: tuple[ControlSignal, jnp.ndarray] | None = None
    if forcing is not None:
        if reference is None:
            raise ValueError(
                "If providing a default forcing/reference pair, both must be supplied."
            )
        default_data = (forcing, reference)

    def _loss(prediction: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        if loss_type in {"norm", "norm_mse"}:
            return norm_loss_fn(prediction, target)
        if loss_type == "mse":
            return mse(prediction, target)
        raise ValueError(
            f"Unknown loss_type '{loss_type}'. Expected 'mse' or 'norm'/'norm_mse'."
        )

    def loss_fn(
        model: _LinearModel,
        batch: Tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> jnp.ndarray:
        if batch is None:
            if default_data is None:
                raise ValueError(
                    "loss_fn requires batch data when no defaults are set."
                )
            control, target = default_data
            prediction = solve_with_model(
                model,
                ts,
                control,
                initial_state,
                dt,
                solver=solver,
                stepsize_controller=stepsize_controller,
                rtol=rtol,
                atol=atol,
            )
            return _loss(prediction, target)

        batch_forcing, batch_reference = batch
        if batch_forcing.ndim == 1:
            batch_forcing = batch_forcing[None, ...]
            batch_reference = batch_reference[None, ...]

        def body(
            running_loss: jnp.ndarray,
            inputs: tuple[jnp.ndarray, jnp.ndarray],
        ) -> tuple[jnp.ndarray, None]:
            forcing_values, target_values = inputs
            length = forcing_values.shape[0]
            time_grid = ts[:length]
            control = control_builder(time_grid, forcing_values)
            prediction = solve_with_model(
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
            sample_loss = _loss(prediction, target_values[:length])
            return running_loss + sample_loss, None

        total_loss, _ = jax.lax.scan(
            body,
            jnp.array(0.0, dtype=jnp.float32),
            (batch_forcing, batch_reference),
        )
        batch_size = batch_forcing.shape[0]
        return total_loss / batch_size

    jitted_loss = eqx.filter_jit(loss_fn)
    loss_and_grad = eqx.filter_value_and_grad(jitted_loss)
    return LossFunction(jitted_loss, loss_and_grad)


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
    model: ModelT,
    loss_fn: LossFunction[ModelT]
    | Callable[[ModelT, Tuple[jnp.ndarray, jnp.ndarray] | None], jnp.ndarray],
    optimizer: optax.GradientTransformation,
    num_steps: int,
    dataloader: Iterable[Tuple[jnp.ndarray, jnp.ndarray]] | None = None,
    *,
    tensorboard_callback: TensorBoardCallback | None = None,
    checkpoint_manager: CheckpointManager | None = None,
    checkpoint_every: int = 0,
) -> Tuple[ModelT, list[float]]:
    history: list[float] = []

    loss_and_grad: Callable[
        [ModelT, Tuple[jnp.ndarray, jnp.ndarray] | None], tuple[jnp.ndarray, Any]
    ]
    if isinstance(loss_fn, LossFunction):
        loss_and_grad = loss_fn.value_and_grad
    else:
        loss_and_grad = cast(
            Callable[
                [ModelT, Tuple[jnp.ndarray, jnp.ndarray] | None],
                tuple[jnp.ndarray, Any],
            ],
            eqx.filter_value_and_grad(loss_fn),
        )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    batches = _batch_iterator(dataloader)

    @eqx.filter_jit
    def step(
        model: ModelT,
        opt_state: optax.OptState,
        batch: Tuple[jnp.ndarray, jnp.ndarray] | None,
        step_index: jnp.ndarray,
    ) -> tuple[ModelT, optax.OptState, jnp.ndarray]:
        loss, grads = loss_and_grad(model, batch)
        if tensorboard_callback is not None:
            tensorboard_callback(step_index, loss)
        params = eqx.filter(model, eqx.is_array)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    iterator_source = range(num_steps)
    progress_bar = _jax_tqdm if _jax_tqdm is not None else _standard_tqdm
    iterator = progress_bar(iterator_source, desc="NeuralODE training")
    for step_idx in iterator:
        batch = next(batches)
        step_value = jnp.asarray(step_idx, dtype=jnp.int32)
        model, opt_state, loss = step(model, opt_state, batch, step_value)
        history.append(float(loss))
        if (
            checkpoint_manager is not None
            and checkpoint_every > 0
            and (step_idx + 1) % checkpoint_every == 0
        ):
            checkpoint_manager.save(step_idx + 1, model, opt_state)
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
        tensorboard_callback=neural_ode.tensorboard_callback,
        checkpoint_manager=neural_ode.checkpoint_manager,
        checkpoint_every=neural_ode.checkpoint_every,
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
        raise ValueError(
            "predict_neural_ode requires dataloader to yield at least one batch."
        )

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
    normalize: bool = False,
) -> tuple[Axes, Axes]:
    """Plot reference vs. predicted trajectories for a sample from the dataloader, optionally normalizing both series."""

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

    plot_target = target
    plot_prediction = prediction
    if normalize:
        norm_target, norm_prediction = normalize_state_pair(
            plot_target, plot_prediction
        )
        plot_target = jnp.asarray(norm_target)
        plot_prediction = jnp.asarray(norm_prediction)

    trajectory_ylabel = "Normalized State" if normalize else None
    trajectory_ax = plot_trajectory(
        ts,
        plot_target,
        labels=target_labels,
        title=title,
        ylabel=trajectory_ylabel,
    )
    plot_trajectory(
        ts,
        plot_prediction,
        labels=prediction_labels,
        ax=trajectory_ax,
        title=None,
        styles=tuple(["--"] * prediction.shape[1]),
    )
    labels_for_residuals = residual_labels or target_labels
    residual_ylabel = "Normalized Residual" if normalize else None
    residual_ax = plot_residuals(
        ts,
        plot_target,
        plot_prediction,
        labels=labels_for_residuals,
        ylabel=residual_ylabel,
    )
    return trajectory_ax, residual_ax


def plot_neural_ode_loss(
    neural_ode: NeuralODE,
    *,
    ax: Axes | None = None,
    title: str = "Neural ODE Training Loss",
) -> Axes:
    """Visualize the stored training loss history for a NeuralODE instance."""

    if not neural_ode.history:
        raise ValueError(
            "NeuralODE has no recorded loss history. Train before plotting."
        )
    return plot_loss_curve(neural_ode.history, ax=ax, title=title)
