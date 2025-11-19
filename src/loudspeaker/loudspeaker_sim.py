from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TypeAlias, cast

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Tsit5

from ._integrator import integrate_system
from .testsignals import ControlSignal

ScalarLike: TypeAlias = bool | int | float | jax.Array | np.ndarray


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

    def __post_init__(self: "LoudspeakerConfig") -> None:
        if self.moving_mass <= 0.0:
            raise ValueError("moving_mass must be positive.")
        if self.compliance <= 0.0:
            raise ValueError("compliance must be positive.")
        if self.damping < 0.0:
            raise ValueError("damping cannot be negative.")
        if self.motor_force <= 0.0:
            raise ValueError("motor_force must be positive.")
        if self.voice_coil_resistance <= 0.0:
            raise ValueError("voice_coil_resistance must be positive.")
        if self.voice_coil_inductance <= 0.0:
            raise ValueError("voice_coil_inductance must be positive.")
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2.")
        init = jnp.asarray(self.initial_state, dtype=jnp.float32)
        if init.shape != (3,):
            raise ValueError("initial_state must have shape (3,).")
        object.__setattr__(self, "initial_state", init)

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
class NonlinearLoudspeakerConfig(LoudspeakerConfig):
    """Configuration for loudspeaker models with state-dependent parameters."""

    suspension_cubic: float = 0.0
    force_factor_sag: float = 0.0

    def __post_init__(self: "NonlinearLoudspeakerConfig") -> None:
        super().__post_init__()
        if self.suspension_cubic < 0.0:
            raise ValueError("suspension_cubic cannot be negative.")
        if self.force_factor_sag < 0.0:
            raise ValueError("force_factor_sag cannot be negative.")

    def suspension_gain(
        self: "NonlinearLoudspeakerConfig", displacement: jnp.ndarray
    ) -> jnp.ndarray:
        return 1.0 + self.suspension_cubic * (displacement**2)

    def bl_factor(
        self: "NonlinearLoudspeakerConfig", displacement: jnp.ndarray
    ) -> jnp.ndarray:
        base = 1.0 - self.force_factor_sag * (displacement**2)
        return self.motor_force * jnp.clip(base, min=0.0)


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
) -> Callable[[ScalarLike, jnp.ndarray, Any], jnp.ndarray]:
    inv_mass = 1.0 / config.moving_mass
    inv_inductance = 1.0 / config.voice_coil_inductance

    def vf(t: ScalarLike, state: jnp.ndarray, _args: Any) -> jnp.ndarray:
        force_input = cast(float | jnp.ndarray, t)
        pos, vel, current = state
        voltage = forcing.evaluate(force_input)
        acceleration = (
            config.motor_force * current - config.damping * vel - config.stiffness * pos
        ) * inv_mass
        current_rate = (
            voltage - config.voice_coil_resistance * current - config.motor_force * vel
        ) * inv_inductance
        return jnp.array([vel, acceleration, current_rate], dtype=jnp.float32)

    return vf


def _build_nonlinear_vector_field(
    config: NonlinearLoudspeakerConfig,
    forcing: ControlSignal,
) -> Callable[[ScalarLike, jnp.ndarray, Any], jnp.ndarray]:
    inv_mass = 1.0 / config.moving_mass
    inv_inductance = 1.0 / config.voice_coil_inductance

    def vf(t: ScalarLike, state: jnp.ndarray, _args: Any) -> jnp.ndarray:
        force_input = cast(float | jnp.ndarray, t)
        pos, vel, current = state
        voltage = forcing.evaluate(force_input)
        stiffness_term = config.stiffness * config.suspension_gain(pos) * pos
        bl = config.bl_factor(pos)
        coil_force = bl * current
        acceleration = (coil_force - config.damping * vel - stiffness_term) * inv_mass
        current_rate = (
            voltage - config.voice_coil_resistance * current - bl * vel
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

    vector_field = _build_vector_field(config, forcing)
    ts_values, states = integrate_system(
        config,
        vector_field,
        solver=solver,
        ts=ts,
    )

    voltages = None
    coil_force = None
    if capture_details:
        voltages = forcing.evaluate_batch(ts_values)
        coil_force = config.motor_force * states[:, 2]

    return LoudspeakerSimulationResult(
        ts=ts_values,
        states=states,
        voltages=voltages,
        coil_force=coil_force,
    )


def simulate_nonlinear_loudspeaker_system(
    config: NonlinearLoudspeakerConfig,
    forcing: ControlSignal,
    solver: Tsit5 | None = None,
    ts: jnp.ndarray | None = None,
    capture_details: bool = False,
) -> LoudspeakerSimulationResult:
    """Simulate the loudspeaker with nonlinear suspension or motor force."""

    vector_field = _build_nonlinear_vector_field(config, forcing)
    ts_values, states = integrate_system(
        config,
        vector_field,
        solver=solver,
        ts=ts,
    )

    voltages = None
    coil_force = None
    if capture_details:
        voltages = forcing.evaluate_batch(ts_values)
        bl_values = config.bl_factor(states[:, 0])
        coil_force = bl_values * states[:, 2]

    return LoudspeakerSimulationResult(
        ts=ts_values,
        states=states,
        voltages=voltages,
        coil_force=coil_force,
    )
