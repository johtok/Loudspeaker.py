from __future__ import annotations

import numpy as np
import chex
import jax.numpy as jnp

from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.testsignals import build_control_signal, complex_tone_control


def _zero_control(config: MSDConfig):
    ts = jnp.linspace(0.0, config.duration, config.num_samples)
    return build_control_signal(ts, jnp.zeros_like(ts))


def test_simulate_msd_system_with_zero_forcing_stays_at_rest():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    result = simulate_msd_system(config, control)
    chex.assert_trees_all_close(result.ts, control.ts)
    chex.assert_trees_all_close(result.states, jnp.zeros_like(result.states))


def test_simulate_msd_system_return_details_matches_force():
    config = MSDConfig(num_samples=6)
    control = _zero_control(config)
    result = simulate_msd_system(
        config,
        control,
        capture_details=True,
    )
    assert result.has_details()
    assert result.forces is not None
    assert result.acceleration is not None
    chex.assert_trees_all_close(result.forces, control.values)
    chex.assert_shape(result.states, (config.num_samples, 2))
    chex.assert_shape(result.acceleration, (config.num_samples,))
    chex.assert_trees_all_close(result.acceleration, jnp.zeros_like(result.acceleration))


def test_simulate_msd_system_accepts_custom_time_grid():
    config = MSDConfig(num_samples=8)
    control = _zero_control(config)
    custom_ts = jnp.linspace(0.0, config.duration / 2.0, config.num_samples)
    result = simulate_msd_system(config, control, ts=custom_ts)
    chex.assert_trees_all_close(result.ts, custom_ts)
    chex.assert_shape(result.states, (config.num_samples, 2))


def test_msd_config_scaling_relations():
    config = MSDConfig(mass=0.12, natural_frequency=30.0, damping_ratio=0.05)
    omega = 2 * jnp.pi * config.natural_frequency
    chex.assert_trees_all_close(config.stiffness, config.mass * omega**2)
    chex.assert_trees_all_close(config.damping, 2 * config.damping_ratio * config.mass * omega)


def test_simulation_obeys_newtonian_dynamics():
    config = MSDConfig(num_samples=256, sample_rate=400.0)
    control = complex_tone_control(
        num_samples=config.num_samples,
        dt=config.dt,
        frequencies=(8.0, 15.0),
        amplitudes=(0.5, 1.25),
    )
    result = simulate_msd_system(config, control)

    ts_np = np.asarray(result.ts)
    pos = np.asarray(result.states[:, 0])
    vel = np.asarray(result.states[:, 1])
    numeric_acc = np.gradient(vel, ts_np)

    reconstructed_force = (
        config.mass * numeric_acc + config.damping * vel + config.stiffness * pos
    )
    expected_force = np.asarray(control.evaluate_batch(result.ts))
    # Ignore boundary artifacts from the numerical gradient.
    assert np.allclose(
        reconstructed_force[2:-2],
        expected_force[2:-2],
        atol=5e-2,
        rtol=5e-2,
    )
