from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from .loudspeaker_sim import LoudspeakerConfig, simulate_loudspeaker_system
from .msd_sim import MSDConfig, simulate_msd_system
from .testsignals import ControlSignal, build_control_signal, pink_noise_control

ForcingFactory = Callable[..., ControlSignal]
Batch = tuple[jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class MSDDataset:
    ts: jnp.ndarray
    forcing: jnp.ndarray
    reference: jnp.ndarray

    def __iter__(
        self: MSDDataset,
    ) -> Iterator[jnp.ndarray]:
        """Allow unpacking as (ts, forcing, reference)."""

        return iter((self.ts, self.forcing, self.reference))


@dataclass(frozen=True)
class TrainTestSplit:
    """Validation-focused train/test partition for MSD datasets."""

    train_forcing: jnp.ndarray
    train_reference: jnp.ndarray
    test_forcing: jnp.ndarray
    test_reference: jnp.ndarray

    @property
    def train_size(self) -> int:
        return self.train_forcing.shape[0]

    @property
    def test_size(self) -> int:
        return self.test_forcing.shape[0]

    def evaluation_batch(self) -> Batch:
        return (self.test_forcing[:1], self.test_reference[:1])

    def evaluation_batches(self) -> list[Batch]:
        if self.test_size == 0:
            return []
        seq_len = self.test_forcing.shape[1]
        ref_dim = self.test_reference.shape[2]
        indices = jnp.arange(self.test_size, dtype=jnp.int32)

        def _slice(idx: jax.Array) -> Batch:
            return (
                jax.lax.dynamic_slice(self.test_forcing, (idx, 0), (1, seq_len)),
                jax.lax.dynamic_slice(
                    self.test_reference,
                    (idx, 0, 0),
                    (1, seq_len, ref_dim),
                ),
            )

        forcing_batches, reference_batches = jax.vmap(_slice)(indices)
        return list(zip(forcing_batches, reference_batches))

    @classmethod
    def from_dataset(
        cls, forcing: jnp.ndarray, reference: jnp.ndarray, train_fraction: float
    ) -> "TrainTestSplit":
        if forcing.shape[0] != reference.shape[0]:
            raise ValueError(
                "Forcing and reference must share the same number of samples."
            )
        dataset_size = forcing.shape[0]
        if dataset_size < 2:
            raise ValueError("Need at least two samples to create a train/test split.")
        if not (0.0 <= train_fraction <= 1.0):
            raise ValueError("train_fraction must be between 0 and 1.")

        train_size = int(
            _train_size_expression(
                jnp.asarray(dataset_size, dtype=jnp.float32),
                jnp.asarray(train_fraction, dtype=jnp.float32),
            ).item()
        )
        test_size = dataset_size - train_size
        if test_size < 1:
            raise ValueError(
                "Train/test split must leave at least one sample for evaluation."
            )

        return cls(
            forcing[:train_size],
            reference[:train_size],
            forcing[train_size:],
            reference[train_size:],
        )


def _build_time_series_dataset(
    config: MSDConfig | LoudspeakerConfig,
    dataset_size: int,
    key: jax.Array,
    *,
    band: Tuple[float, float] | None,
    forcing_fn: ForcingFactory | None,
    forcing_kwargs: Mapping[str, Any] | None,
    simulate_fn: Callable[..., Any],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    forcing_fn = forcing_fn or pink_noise_control
    kwargs = dict(forcing_kwargs or {})
    if band is not None:
        kwargs.setdefault("band", band)

    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    keys = jr.split(key, dataset_size)

    forcing_values = jnp.stack(
        [
            forcing_fn(
                num_samples=config.num_samples,
                dt=config.dt,
                key=subkey,
                **kwargs,
            ).values
            for subkey in keys
        ]
    )
    reference_states = _simulate_states_batch(
        config,
        ts,
        forcing_values,
        simulate_fn=simulate_fn,
    )
    return ts, forcing_values, reference_states


def build_msd_dataset(
    config: MSDConfig,
    dataset_size: int,
    key: jax.Array,
    band: Tuple[float, float] | None = (1.0, 100.0),
    *,
    forcing_fn: ForcingFactory | None = None,
    forcing_kwargs: Mapping[str, Any] | None = None,
) -> MSDDataset:
    """Simulate random forcing trajectories for MSD training."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    ts, forcing_values, reference_states = _build_time_series_dataset(
        config,
        dataset_size,
        key,
        band=band,
        forcing_fn=forcing_fn,
        forcing_kwargs=forcing_kwargs,
        simulate_fn=simulate_msd_system,
    )
    return MSDDataset(ts=ts, forcing=forcing_values, reference=reference_states)


def build_loudspeaker_dataset(
    config: LoudspeakerConfig,
    dataset_size: int,
    key: jax.Array,
    band: Tuple[float, float] | None = (20.0, 1000.0),
    *,
    forcing_fn: ForcingFactory | None = None,
    forcing_kwargs: Mapping[str, Any] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Generate supervised data for loudspeaker identification experiments."""

    if dataset_size < 1:
        raise ValueError("dataset_size must be positive.")

    return _build_time_series_dataset(
        config,
        dataset_size,
        key,
        band=band,
        forcing_fn=forcing_fn,
        forcing_kwargs=forcing_kwargs,
        simulate_fn=simulate_loudspeaker_system,
    )


@dataclass(frozen=True)
class StrategyPhase:
    steps: int
    length_fraction: float

    def __post_init__(self):
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if not 0.0 < self.length_fraction <= 1.0:
            raise ValueError("length_fraction must be within (0, 1].")


class TrainingStrategy:
    """Iterable container describing curriculum phases."""

    def __init__(self, phases: Sequence[StrategyPhase]):
        if not phases:
            raise ValueError("TrainingStrategy requires at least one phase.")
        self._phases = tuple(phases)

    def __iter__(self: TrainingStrategy) -> Iterator[StrategyPhase]:
        return iter(self._phases)

    @property
    def phases(self: TrainingStrategy) -> tuple[StrategyPhase, ...]:
        return self._phases

    @property
    def total_steps(self: TrainingStrategy) -> int:
        return sum(phase.steps for phase in self._phases)


class StaticTrainingStrategy(TrainingStrategy):
    """Single-phase strategy matching the original training loop."""

    def __init__(self, steps: int, length_fraction: float = 1.0):
        super().__init__((StrategyPhase(steps, length_fraction),))


@jax.jit
def _phase_length_expression(
    num_samples: jnp.ndarray, length_fraction: jnp.ndarray
) -> jnp.ndarray:
    length = jnp.floor(num_samples * length_fraction)
    clipped = jnp.maximum(length, 2.0)
    return jnp.minimum(clipped, num_samples)


@jax.jit
def _train_size_expression(
    dataset_size: jnp.ndarray, train_fraction: jnp.ndarray
) -> jnp.ndarray:
    clipped_fraction = jnp.clip(train_fraction, 0.0, 1.0)
    raw_size = jnp.ceil(clipped_fraction * dataset_size)
    return jnp.clip(raw_size, 1.0, dataset_size - 1.0)


def _phase_length(num_samples: int, fraction: float) -> int:
    length = _phase_length_expression(
        jnp.asarray(num_samples, dtype=jnp.float32),
        jnp.asarray(fraction, dtype=jnp.float32),
    )
    return int(length.item())


def _simulate_states_batch(
    config: MSDConfig | LoudspeakerConfig,
    ts: jnp.ndarray,
    forcing_values: jnp.ndarray,
    *,
    simulate_fn: Callable[..., Any],
) -> jnp.ndarray:
    def _simulate(values: jax.Array) -> jnp.ndarray:
        control = build_control_signal(ts, values)
        return simulate_fn(config, control, ts=ts).states

    return jax.vmap(_simulate)(forcing_values)


def _permuted_batch_indices(
    dataset_size: int, batch_size: int, key: jax.Array
) -> Iterator[jnp.ndarray]:
    rng = key
    while True:
        rng, perm_key = jr.split(rng)
        perm = jr.permutation(perm_key, dataset_size)
        usable = dataset_size - dataset_size % batch_size
        if usable == 0:
            continue
        trimmed = perm[:usable]
        batch_count = usable // batch_size
        batches = tuple(jnp.split(trimmed, batch_count))
        yield from batches


def msd_dataloader(
    forcing_values: jnp.ndarray,
    reference_states: jnp.ndarray,
    batch_size: int,
    *,
    key: jax.Array,
    strategy: TrainingStrategy | None = None,
) -> Iterator[Batch]:
    """Iterate over MSD samples with optional curriculum strategy."""

    dataset_size = forcing_values.shape[0]
    if batch_size > dataset_size:
        raise ValueError("batch_size cannot exceed dataset size")
    if reference_states.shape[0] != dataset_size:
        raise ValueError("forcing/reference dataset mismatch")

    num_samples = forcing_values.shape[1]
    batch_indices = _permuted_batch_indices(dataset_size, batch_size, key)

    if strategy is None:
        while True:
            batch_idx = next(batch_indices)
            yield forcing_values[batch_idx], reference_states[batch_idx]
        return

    phase_specs = tuple(
        (phase.steps, _phase_length(num_samples, phase.length_fraction))
        for phase in strategy.phases
    )
    while True:
        for steps, phase_length in phase_specs:
            for _ in range(steps):
                batch_idx = next(batch_indices)
                yield (
                    forcing_values[batch_idx, :phase_length],
                    reference_states[batch_idx, :phase_length],
                )
