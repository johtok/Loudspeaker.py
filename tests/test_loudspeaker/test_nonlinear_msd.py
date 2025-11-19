from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.nonlinear_msd import (
    NonlinearMSDConfig,
    NonlinearMSDSimConfig,
    build_nonlinear_msd_training_data,
    simulate_nonlinear_msd_system,
)
from loudspeaker.testsignals import build_control_signal


def test_build_nonlinear_dataset_shapes_and_stats():
    config = NonlinearMSDConfig(dataset_size=32)
    key = jr.PRNGKey(0)
    states, controls, derivatives = build_nonlinear_msd_training_data(config, key)
    chex.assert_shape(states, (config.dataset_size, 2))
    chex.assert_shape(controls, (config.dataset_size, 1))
    chex.assert_shape(derivatives, (config.dataset_size, 2))
    # ensure stochasticity of samples
    chex.assert_trees_all_close(jnp.mean(states, axis=0), jnp.zeros(2), atol=0.5)


def test_build_nonlinear_dataset_can_return_matrices():
    config = NonlinearMSDConfig(dataset_size=16)
    key = jr.PRNGKey(1)
    states, controls, derivatives, matrices = build_nonlinear_msd_training_data(
        config,
        key,
        include_matrices=True,
    )
    chex.assert_shape(matrices, (config.dataset_size, 2, 3))
    expected = jnp.broadcast_to(
        jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32), (config.dataset_size, 3)
    )
    chex.assert_trees_all_close(matrices[:, 0, :], expected, atol=1e-6)


def test_simulate_nonlinear_matches_linear_when_cubic_zero():
    base_config = MSDConfig(num_samples=64, sample_rate=400.0)
    ts = jnp.linspace(
        0.0, base_config.duration, base_config.num_samples, dtype=jnp.float32
    )
    forcing = build_control_signal(ts, jnp.ones_like(ts))
    linear_result = simulate_msd_system(base_config, forcing)
    nonlinear_config = NonlinearMSDSimConfig(
        mass=base_config.mass,
        natural_frequency=base_config.natural_frequency,
        damping_ratio=base_config.damping_ratio,
        sample_rate=base_config.sample_rate,
        num_samples=base_config.num_samples,
        initial_state=base_config.initial_state,
        cubic=0.0,
    )
    nonlinear_result = simulate_nonlinear_msd_system(nonlinear_config, forcing)
    chex.assert_trees_all_close(
        linear_result.states, nonlinear_result.states, atol=1e-6
    )


def test_simulate_nonlinear_includes_cubic_restoring_force():
    config = NonlinearMSDSimConfig(num_samples=32, sample_rate=200.0, cubic=10.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing = build_control_signal(ts, jnp.zeros_like(ts))
    # Start away from equilibrium to trigger cubic restoring force.
    displaced_config = NonlinearMSDSimConfig(
        mass=config.mass,
        natural_frequency=config.natural_frequency,
        damping_ratio=config.damping_ratio,
        sample_rate=config.sample_rate,
        num_samples=config.num_samples,
        initial_state=jnp.array([0.05, 0.0], dtype=jnp.float32),
        cubic=config.cubic,
    )
    result = simulate_nonlinear_msd_system(
        displaced_config, forcing, capture_details=True
    )
    chex.assert_trees_all_close(result.states[0], displaced_config.initial_state)
    # Ensure the cubic force accelerates the mass back toward zero displacement.
    assert jnp.sign(result.acceleration[0]) == -jnp.sign(
        displaced_config.initial_state[0]
    )


def test_simulate_nonlinear_msd_custom_time_grid():
    config = NonlinearMSDSimConfig(num_samples=10, sample_rate=100.0)
    ts = jnp.linspace(0.0, config.duration / 2.0, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.zeros_like(ts))
    result = simulate_nonlinear_msd_system(config, control, ts=ts)
    chex.assert_trees_all_close(result.ts, ts)


def test_simulate_nonlinear_msd_capture_details_match_control():
    config = NonlinearMSDSimConfig(num_samples=12, sample_rate=120.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.sin(2 * jnp.pi * ts * 5.0))
    result = simulate_nonlinear_msd_system(config, control, capture_details=True)
    assert result.forces is not None
    assert result.acceleration is not None
    chex.assert_trees_all_close(result.forces, control.values)
    chex.assert_shape(result.acceleration, (config.num_samples,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mass": 0.0},
        {"stiffness": 0.0},
        {"damping": -0.1},
        {"state_scale": 0.0},
        {"control_scale": -1.0},
        {"dataset_size": 1},
    ],
)
def test_nonlinear_msd_config_validates_parameters(kwargs):
    with pytest.raises(ValueError):
        NonlinearMSDConfig(**kwargs)
