from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .msd_sim import MSDConfig, SimulationResult
from .testsignals import ControlSignal

ScalarLike: TypeAlias = bool | int | float | jax.Array | np.ndarray


@dataclass(frozen=True)
class NonlinearMSDConfig:
    """Configuration for generating i.i.d. nonlinear MSD samples."""

    mass: float = 0.05
    stiffness: float = 100.0
    damping: float = 0.4
    cubic: float = 5.0
    state_scale: float = 1.0
    control_scale: float = 1.0
    dataset_size: int = 2048

    def __post_init__(self: "NonlinearMSDConfig") -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.stiffness <= 0.0:
            raise ValueError("stiffness must be positive.")
        if self.damping < 0.0:
            raise ValueError("damping cannot be negative.")
        if self.state_scale <= 0.0:
            raise ValueError("state_scale must be positive.")
        if self.control_scale <= 0.0:
            raise ValueError("control_scale must be positive.")
        if self.dataset_size < 2:
            raise ValueError("dataset_size must be at least 2.")


def nonlinear_msd_derivative(
    config: NonlinearMSDConfig, state: jnp.ndarray, control: jnp.ndarray
) -> jnp.ndarray:
    """Exact dynamics for the Duffing-like MSD oscillator."""

    pos, vel = state
    force = control[0]
    restoring = config.stiffness * pos + config.cubic * (pos**3)
    acc = (force - config.damping * vel - restoring) / config.mass
    return jnp.array([vel, acc], dtype=jnp.float32)


def nonlinear_msd_matrix(config: NonlinearMSDConfig, state: jnp.ndarray) -> jnp.ndarray:
    """Linearized state matrix that maps [x, v, u] -> [v, a]."""

    pos, _ = state
    stiffness_term = (-config.stiffness - config.cubic * pos**2) / config.mass
    damping_term = -config.damping / config.mass
    control_term = 1.0 / config.mass
    return jnp.array(
        [
            [0.0, 1.0, 0.0],
            [stiffness_term, damping_term, control_term],
        ],
        dtype=jnp.float32,
    )


def build_nonlinear_msd_training_data(
    config: NonlinearMSDConfig,
    key: jax.Array,
    *,
    include_matrices: bool = False,
) -> (
    Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    | Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Sample random states/controls and evaluate nonlinear MSD derivatives."""

    rng, state_key, control_key = jr.split(key, 3)
    states = config.state_scale * jr.normal(state_key, (config.dataset_size, 2))
    controls = config.control_scale * jr.normal(control_key, (config.dataset_size, 1))
    batched_derivative = eqx.filter_vmap(
        lambda s, u: nonlinear_msd_derivative(config, s, u)
    )
    derivatives = batched_derivative(states, controls)

    if not include_matrices:
        return states, controls, derivatives

    batched_matrix = eqx.filter_vmap(lambda s: nonlinear_msd_matrix(config, s))
    matrices = batched_matrix(states)
    return states, controls, derivatives, matrices


@dataclass(frozen=True)
class NonlinearMSDSimConfig(MSDConfig):
    """Time-domain simulation parameters for the Duffing MSD."""

    cubic: float = 5.0

    def __post_init__(self: "NonlinearMSDSimConfig") -> None:
        super().__post_init__()


def _build_nonlinear_vector_field(
    config: NonlinearMSDSimConfig,
    forcing: ControlSignal,
) -> Callable[[ScalarLike, jnp.ndarray, Any], jnp.ndarray]:
    def vf(t: ScalarLike, state: jnp.ndarray, _args: Any) -> jnp.ndarray:
        force_input = cast(float | jnp.ndarray, t)
        pos, vel = state
        force = forcing.evaluate(force_input)
        restoring = config.stiffness * pos + config.cubic * (pos**3)
        acc = (force - config.damping * vel - restoring) / config.mass
        return jnp.array([vel, acc], dtype=jnp.float32)

    return vf


def simulate_nonlinear_msd_system(
    config: NonlinearMSDSimConfig,
    forcing: ControlSignal,
    solver: Tsit5 | None = None,
    ts: jnp.ndarray | None = None,
    capture_details: bool = False,
) -> SimulationResult:
    """Integrate the nonlinear MSD (Duffing) dynamics."""

    solver = solver or Tsit5()
    if ts is None:
        ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    else:
        ts = jnp.asarray(ts, dtype=jnp.float32)
    term: ODETerm = ODETerm(_build_nonlinear_vector_field(config, forcing))
    sol = diffeqsolve(
        term,
        solver,
        t0=ts[0],
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
        restoring = config.stiffness * sol.ys[:, 0] + config.cubic * (sol.ys[:, 0] ** 3)
        acc = (forces - config.damping * sol.ys[:, 1] - restoring) / config.mass

    return SimulationResult(ts=sol.ts, states=sol.ys, forces=forces, acceleration=acc)
