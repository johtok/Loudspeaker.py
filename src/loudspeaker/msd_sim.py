from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax import tree_util

from .testsignals import ControlSignal

ScalarLike: TypeAlias = bool | int | float | jax.Array | np.ndarray


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

    def __post_init__(self: "MSDConfig") -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.natural_frequency <= 0.0:
            raise ValueError("natural_frequency must be positive.")
        if self.damping_ratio < 0.0:
            raise ValueError("damping_ratio cannot be negative.")
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        init = jnp.asarray(self.initial_state, dtype=jnp.float32)
        if init.shape != (2,):
            raise ValueError("initial_state must have shape (2,).")
        object.__setattr__(self, "initial_state", init)

    @property
    def dt(self: "MSDConfig") -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self: "MSDConfig") -> float:
        return float(self.num_samples - 1) * self.dt

    @property
    def _omega(self: "MSDConfig") -> float:
        return 2 * jnp.pi * self.natural_frequency

    @property
    def stiffness(self: "MSDConfig") -> float:
        return self.mass * self._omega**2

    @property
    def damping(self: "MSDConfig") -> float:
        return 2 * self.damping_ratio * self.mass * self._omega


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SimulationResult:
    ts: jnp.ndarray
    states: jnp.ndarray
    forces: jnp.ndarray | None = None
    acceleration: jnp.ndarray | None = None

    def has_details(self: "SimulationResult") -> bool:
        return self.forces is not None and self.acceleration is not None

    def tree_flatten(
        self: "SimulationResult",
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None], None
    ]:
        children = (self.ts, self.states, self.forces, self.acceleration)
        return children, None

    @classmethod
    def tree_unflatten(
        cls: type["SimulationResult"],
        _aux_data: Any,
        children: tuple[
            jnp.ndarray, jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None
        ],
    ) -> "SimulationResult":
        ts, states, forces, acceleration = children
        return cls(ts=ts, states=states, forces=forces, acceleration=acceleration)


def _build_vector_field(
    config: MSDConfig, forcing: ControlSignal
) -> Callable[[ScalarLike, jnp.ndarray, Any], jnp.ndarray]:
    def vf(t: ScalarLike, state: jnp.ndarray, _args: Any) -> jnp.ndarray:
        force_input = cast(float | jnp.ndarray, t)
        pos, vel = state
        force = forcing.evaluate(force_input)
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

    if sol.ts is None or sol.ys is None:
        raise RuntimeError("Solver returned no trajectory.")

    forces = None
    acc = None
    if capture_details:
        forces = forcing.evaluate_batch(sol.ts)
        acc = (
            forces - config.damping * sol.ys[:, 1] - config.stiffness * sol.ys[:, 0]
        ) / config.mass

    return SimulationResult(ts=sol.ts, states=sol.ys, forces=forces, acceleration=acc)
