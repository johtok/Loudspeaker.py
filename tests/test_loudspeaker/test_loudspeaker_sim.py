from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from loudspeaker.loudspeaker_sim import (
    LoudspeakerConfig,
    LoudspeakerSimulationResult,
    NonlinearLoudspeakerConfig,
    simulate_loudspeaker_system,
    simulate_nonlinear_loudspeaker_system,
)
from loudspeaker.models import LoudspeakerSimulationModel
from loudspeaker.testsignals import build_control_signal


def test_simulate_loudspeaker_system_returns_states_with_details():
    config = LoudspeakerConfig(num_samples=32, sample_rate=2000.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing = build_control_signal(ts, jnp.ones_like(ts))
    result = simulate_loudspeaker_system(config, forcing, capture_details=True)
    assert isinstance(result, LoudspeakerSimulationResult)
    assert result.states.shape == (config.num_samples, 3)
    assert result.voltages is not None
    assert result.coil_force is not None
    # Access helpers ensure field order matches expectation.
    assert result.displacement().shape == (config.num_samples,)
    assert result.velocity().shape == (config.num_samples,)
    assert result.coil_current().shape == (config.num_samples,)


def test_nonlinear_loudspeaker_matches_linear_when_nonlinear_terms_zero():
    config = LoudspeakerConfig(num_samples=64, sample_rate=2000.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.ones_like(ts))
    linear = simulate_loudspeaker_system(config, control)
    nonlinear_config = NonlinearLoudspeakerConfig(
        moving_mass=config.moving_mass,
        compliance=config.compliance,
        damping=config.damping,
        motor_force=config.motor_force,
        voice_coil_resistance=config.voice_coil_resistance,
        voice_coil_inductance=config.voice_coil_inductance,
        sample_rate=config.sample_rate,
        num_samples=config.num_samples,
        initial_state=config.initial_state,
        suspension_cubic=0.0,
        force_factor_sag=0.0,
    )
    nonlinear = simulate_nonlinear_loudspeaker_system(nonlinear_config, control)
    mask = ~(jnp.isnan(linear.states) | jnp.isnan(nonlinear.states))
    linear_clean = jnp.where(mask, linear.states, 0.0)
    nonlinear_clean = jnp.where(mask, nonlinear.states, 0.0)
    chex.assert_trees_all_close(linear_clean, nonlinear_clean, atol=1e-6)


def test_nonlinear_loudspeaker_restores_toward_equilibrium_without_forcing():
    config = NonlinearLoudspeakerConfig(
        num_samples=32,
        sample_rate=48000.0,
        suspension_cubic=0.5,
        force_factor_sag=0.0,
        initial_state=jnp.array([0.002, 0.0, 0.0], dtype=jnp.float32),
    )
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.zeros_like(ts))
    result = simulate_nonlinear_loudspeaker_system(
        config, control, capture_details=True
    )
    assert result.coil_force is not None
    # Nonlinear stiffness adds extra restoring force pushing displacement toward zero.
    assert jnp.sign(result.states[1, 1]) == -jnp.sign(config.initial_state[0])


def test_loudspeaker_simulator_accepts_custom_time_grid():
    config = LoudspeakerConfig(num_samples=32, sample_rate=800.0)
    ts = jnp.linspace(0.0, config.duration / 2.0, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.ones_like(ts))
    result = simulate_loudspeaker_system(config, control, ts=ts)
    chex.assert_trees_all_close(result.ts, ts)


def test_nonlinear_loudspeaker_simulator_accepts_custom_grid():
    config = NonlinearLoudspeakerConfig(
        num_samples=16, sample_rate=400.0, suspension_cubic=0.1
    )
    ts = jnp.linspace(0.0, config.duration / 4.0, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.zeros_like(ts))
    result = simulate_nonlinear_loudspeaker_system(config, control, ts=ts)
    chex.assert_trees_all_close(result.ts, ts)


def test_loudspeaker_simulation_model_facade(monkeypatch):
    config = LoudspeakerConfig(num_samples=4)
    sim_model = LoudspeakerSimulationModel(config)
    linear = sim_model.linear_model()
    assert linear.weight.shape[0] == 3

    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.ones_like(ts))
    called = {}

    def _fake_sim(cfg, ctrl, capture_details):
        called["config"] = cfg
        called["capture_details"] = capture_details
        return "ok"

    monkeypatch.setattr(
        "loudspeaker.models.simulation.simulate_loudspeaker_system", _fake_sim
    )
    result = sim_model.simulate(control, capture_details=True)
    assert result == "ok"
    assert called["config"] is config
    assert called["capture_details"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moving_mass": 0.0},
        {"compliance": 0.0},
        {"damping": -0.1},
        {"motor_force": 0.0},
        {"voice_coil_resistance": 0.0},
        {"voice_coil_inductance": 0.0},
        {"sample_rate": 0.0},
        {"num_samples": 1},
        {"initial_state": jnp.zeros(4)},
    ],
)
def test_loudspeaker_config_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        LoudspeakerConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"suspension_cubic": -0.1},
        {"force_factor_sag": -0.1},
    ],
)
def test_nonlinear_loudspeaker_config_validates_terms(kwargs):
    with pytest.raises(ValueError):
        NonlinearLoudspeakerConfig(**kwargs)
