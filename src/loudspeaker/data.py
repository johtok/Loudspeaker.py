from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

import jax.numpy as jnp
import jax.random as jr

from .msd_sim import MSDConfig, simulate_msd_system
from .testsignals import pink_noise_control


def build_msd_dataset(
    config: MSDConfig,
    dataset_size: int,
    key: jr.PRNGKey,
    band: Tuple[float, float] = (1.0, 100.0),
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Simulate random pink-noise trajectories for MSD training."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")

    forcing_values: list[jnp.ndarray] = []
    reference_states: list[jnp.ndarray] = []
    ts = None
    current_key = key
    for _ in range(dataset_size):
        current_key, forcing_key = jr.split(current_key)
        forcing = pink_noise_control(
            num_samples=config.num_samples,
            dt=config.dt,
            key=forcing_key,
            band=band,
        )
        ts, reference = simulate_msd_system(config, forcing)
        forcing_values.append(forcing.values)
        reference_states.append(reference)

    if ts is None:
        raise RuntimeError("Failed to generate MSD dataset.")

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

    def __iter__(self) -> Iterator[StrategyPhase]:
        return iter(self._phases)

    @property
    def phases(self) -> tuple[StrategyPhase, ...]:
        return self._phases

    @property
    def total_steps(self) -> int:
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


def msd_dataloader(
    forcing_values: jnp.ndarray,
    reference_states: jnp.ndarray,
    batch_size: int,
    *,
    key: jr.PRNGKey,
    strategy: TrainingStrategy | None = None,
):
    """Iterate over MSD samples with optional curriculum strategy."""

    dataset_size = forcing_values.shape[0]
    if batch_size > dataset_size:
        raise ValueError("batch_size cannot exceed dataset size")
    if reference_states.shape[0] != dataset_size:
        raise ValueError("forcing/reference dataset mismatch")

    indices = jnp.arange(dataset_size)
    rng = key
    num_samples = forcing_values.shape[1]

    if strategy is None:
        while True:
            rng, perm_key = jr.split(rng)
            perm = jr.permutation(perm_key, indices)
            start = 0
            end = batch_size
            while end <= dataset_size:
                batch_idx = perm[start:end]
                yield forcing_values[batch_idx], reference_states[batch_idx]
                start = end
                end = start + batch_size
        return

    phases = strategy.phases
    while True:
        for phase in phases:
            steps_remaining = phase.steps
            phase_length = _phase_length(num_samples, phase.length_fraction)
            while steps_remaining > 0:
                rng, perm_key = jr.split(rng)
                perm = jr.permutation(perm_key, indices)
                start = 0
                end = batch_size
                while end <= dataset_size and steps_remaining > 0:
                    batch_idx = perm[start:end]
                    yield (
                        forcing_values[batch_idx, :phase_length],
                        reference_states[batch_idx, :phase_length],
                    )
                    start = end
                    end = start + batch_size
                    steps_remaining -= 1
