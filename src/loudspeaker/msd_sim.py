from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .testsignals import ControlSignal


@dataclass
class MSDConfig:
    mass: float = 0.05  # kg
    natural_frequency: float = 25.0  # Hz
    damping_ratio: float = 0.01
    sample_rate: float = 300.0  # Hz
    num_samples: int = 5
    initial_state: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0], dtype=jnp.float64)
    )

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        return (self.num_samples - 1) * self.dt

    @property
    def _omega(self) -> float:
        return 2 * jnp.pi * self.natural_frequency

    @property
    def stiffness(self) -> float:
        return self.mass * self._omega**2

    @property
    def damping(self) -> float:
        return 2 * self.damping_ratio * self.mass * self._omega


def _build_vector_field(config: MSDConfig, forcing: ControlSignal):
    def vf(t, state, args):
        pos, vel = state
        force = forcing.evaluate(t)
        acc = (force - config.damping * vel - config.stiffness * pos) / config.mass
        return jnp.array([vel, acc])

    return vf


def simulate_msd_system(
    config: MSDConfig,
    forcing: ControlSignal,
    solver: Tsit5 | None = None,
    return_details: bool = False,
) -> Tuple:
    """Simulate the MSD system with a provided forcing signal."""

    solver = solver or Tsit5()
    ts = jnp.linspace(0.0, config.duration, config.num_samples)
    term = ODETerm(_build_vector_field(config, forcing))
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=config.duration,
        dt0=config.dt,
        y0=config.initial_state,
        saveat=SaveAt(ts=ts),
    )

    if not return_details:
        return sol.ts, sol.ys

    forces = jnp.array([forcing.evaluate(float(t)) for t in ts])
    acc = (forces - config.damping * sol.ys[:, 1] - config.stiffness * sol.ys[:, 0]) / config.mass
    return sol.ts, sol.ys, forces, acc
