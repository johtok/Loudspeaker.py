from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from .loudspeaker_sim import LoudspeakerConfig, simulate_loudspeaker_system
from .msd_sim import MSDConfig, SimulationResult, simulate_msd_system
from .testsignals import ControlSignal, pink_noise_control

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

    forcing_fn = forcing_fn or pink_noise_control
    kwargs = dict(forcing_kwargs or {})
    if band is not None and "band" not in kwargs:
        kwargs["band"] = band

    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing_values: list[jnp.ndarray] = []
    reference_states: list[jnp.ndarray] = []
    current_key = key
    for _ in range(dataset_size):
        current_key, forcing_key = jr.split(current_key)
        forcing = forcing_fn(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            **kwargs,
        )
        sim_result: SimulationResult = simulate_msd_system(config, forcing, ts=ts)
        forcing_values.append(forcing.values)
        reference_states.append(sim_result.states)

    return MSDDataset(
        ts=ts,
        forcing=jnp.stack(forcing_values),
        reference=jnp.stack(reference_states),
    )


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

    forcing_fn = forcing_fn or pink_noise_control
    kwargs = dict(forcing_kwargs or {})
    if band is not None and "band" not in kwargs:
        kwargs["band"] = band

    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing_values: list[jnp.ndarray] = []
    reference_states: list[jnp.ndarray] = []
    rng = key
    for _ in range(dataset_size):
        rng, forcing_key = jr.split(rng)
        control = forcing_fn(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            **kwargs,
        )
        sim_result = simulate_loudspeaker_system(config, control)
        forcing_values.append(control.values)
        reference_states.append(sim_result.states)

    return ts, jnp.stack(forcing_values), jnp.stack(reference_states)


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


def _phase_length(num_samples: int, fraction: float) -> int:
    length = int(num_samples * fraction)
    if length < 2:
        length = 2
    return min(length, num_samples)


def _permuted_batch_indices(
    dataset_size: int, batch_size: int, key: jax.Array
) -> Iterator[jnp.ndarray]:
    rng = key
    while True:
        rng, perm_key = jr.split(rng)
        perm = jr.permutation(perm_key, dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            if end <= dataset_size:
                yield perm[start:end]


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

    phases = strategy.phases
    while True:
        for phase in phases:
            phase_length = _phase_length(num_samples, phase.length_fraction)
            for _ in range(phase.steps):
                batch_idx = next(batch_indices)
                yield (
                    forcing_values[batch_idx, :phase_length],
                    reference_states[batch_idx, :phase_length],
                )
