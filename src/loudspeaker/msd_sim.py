from __future__ import annotations

from dataclasses import dataclass, field
import jax.numpy as jnp
from jax import tree_util
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .testsignals import ControlSignal


@dataclass(frozen=True)
class MSDConfig:
    mass: float = 0.05  # kg
    natural_frequency: float = 25.0  # Hz
    damping_ratio: float = 0.01
    sample_rate: float = 300.0  # Hz
    num_samples: int = 5
    initial_state: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0], dtype=jnp.float32)
    )

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        return float(self.num_samples - 1) * self.dt

    @property
    def _omega(self) -> float:
        return 2 * jnp.pi * self.natural_frequency

    @property
    def stiffness(self) -> float:
        return self.mass * self._omega**2

    @property
    def damping(self) -> float:
        return 2 * self.damping_ratio * self.mass * self._omega


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationResult:
    ts: jnp.ndarray
    states: jnp.ndarray
    forces: jnp.ndarray | None = None
    acceleration: jnp.ndarray | None = None

    def has_details(self) -> bool:
        return self.forces is not None and self.acceleration is not None

    def tree_flatten(self):
        children = (self.ts, self.states, self.forces, self.acceleration)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ts, states, forces, acceleration = children
        return cls(ts=ts, states=states, forces=forces, acceleration=acceleration)


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
    ts: jnp.ndarray | None = None,
    capture_details: bool = False,
) -> SimulationResult:
    """Simulate the MSD system with a provided forcing signal."""

    solver = solver or Tsit5()
    if ts is None:
        ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    else:
        ts = jnp.asarray(ts, dtype=jnp.float32)
    term = ODETerm(_build_vector_field(config, forcing))
    sol = diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=ts[-1],
        dt0=config.dt,
        y0=config.initial_state,
        saveat=SaveAt(ts=ts),
    )

    forces = None
    acc = None
    if capture_details:
        forces = forcing.evaluate_batch(ts)
        acc = (forces - config.damping * sol.ys[:, 1] - config.stiffness * sol.ys[:, 0]) / config.mass

    return SimulationResult(ts=sol.ts, states=sol.ys, forces=forces, acceleration=acc)
