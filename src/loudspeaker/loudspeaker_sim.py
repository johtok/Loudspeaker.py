from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

from .testsignals import ControlSignal


@dataclass(frozen=True)
class LoudspeakerConfig:
    moving_mass: float = 0.02  # kg
    compliance: float = 1.8e-4  # m/N
    damping: float = 0.4  # N*s/m
    motor_force: float = 7.0  # N/A (Bl product)
    voice_coil_resistance: float = 6.0  # Ohm
    voice_coil_inductance: float = 0.5e-3  # H
    sample_rate: float = 48000.0
    num_samples: int = 512
    initial_state: jnp.ndarray = field(
        default_factory=lambda: jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    )

    @property
    def stiffness(self: "LoudspeakerConfig") -> float:
        return 1.0 / self.compliance

    @property
    def dt(self: "LoudspeakerConfig") -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self: "LoudspeakerConfig") -> float:
        return float(self.num_samples - 1) * self.dt


@dataclass(frozen=True)
class LoudspeakerSimulationResult:
    ts: jnp.ndarray
    states: jnp.ndarray
    voltages: jnp.ndarray | None = None
    coil_force: jnp.ndarray | None = None

    def displacement(self: "LoudspeakerSimulationResult") -> jnp.ndarray:
        return self.states[:, 0]

    def velocity(self: "LoudspeakerSimulationResult") -> jnp.ndarray:
        return self.states[:, 1]

    def coil_current(self: "LoudspeakerSimulationResult") -> jnp.ndarray:
        return self.states[:, 2]


def _build_vector_field(
    config: LoudspeakerConfig, forcing: ControlSignal
) -> callable:
    inv_mass = 1.0 / config.moving_mass
    inv_inductance = 1.0 / config.voice_coil_inductance

    def vf(t: float, state: jnp.ndarray, args: Any) -> jnp.ndarray:
        pos, vel, current = state
        voltage = forcing.evaluate(t)
        acceleration = (
            config.motor_force * current - config.damping * vel - config.stiffness * pos
        ) * inv_mass
        current_rate = (
            voltage
            - config.voice_coil_resistance * current
            - config.motor_force * vel
        ) * inv_inductance
        return jnp.array([vel, acceleration, current_rate], dtype=jnp.float32)

    return vf


def simulate_loudspeaker_system(
    config: LoudspeakerConfig,
    forcing: ControlSignal,
    solver: Tsit5 | None = None,
    ts: jnp.ndarray | None = None,
    capture_details: bool = False,
):
    """Simulate the linear loudspeaker model using diffrax."""

    solver = solver or Tsit5()
    if ts is None:
        ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    else:
        ts = jnp.asarray(ts, dtype=jnp.float32)

    term = ODETerm(_build_vector_field(config, forcing))
    sol = diffeqsolve(
        term,
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=config.dt,
        y0=config.initial_state,
        saveat=SaveAt(ts=ts),
    )

    voltages = None
    coil_force = None
    if capture_details:
        voltages = forcing.evaluate_batch(sol.ts)
        coil_force = config.motor_force * sol.ys[:, 2]

    return LoudspeakerSimulationResult(
        ts=sol.ts,
        states=sol.ys,
        voltages=voltages,
        coil_force=coil_force,
    )
