from __future__ import annotations

from dataclasses import dataclass

import chex
import jax.numpy as jnp

from loudspeaker._integrator import integrate_system


@dataclass(frozen=True)
class DummyConfig:
    sample_rate: float
    num_samples: int
    initial_state: jnp.ndarray

    @property
    def dt(self) -> float:
        return 1.0 / self.sample_rate

    @property
    def duration(self) -> float:
        return float(self.num_samples - 1) * self.dt


def _zero_vector_field(t: float, state: jnp.ndarray, _args: object) -> jnp.ndarray:
    del t, _args
    return jnp.zeros_like(state)


def test_integrate_system_defaults_to_config_grid():
    config = DummyConfig(
        sample_rate=10.0,
        num_samples=5,
        initial_state=jnp.array([1.0, -1.0], dtype=jnp.float32),
    )
    ts, states = integrate_system(config, _zero_vector_field)

    expected_ts = jnp.linspace(
        0.0, config.duration, config.num_samples, dtype=jnp.float32
    )
    chex.assert_trees_all_close(ts, expected_ts)
    expected_states = jnp.tile(config.initial_state[None, :], (config.num_samples, 1))
    chex.assert_trees_all_close(states, expected_states)


def test_integrate_system_respects_custom_time_grid():
    config = DummyConfig(
        sample_rate=5.0,
        num_samples=6,
        initial_state=jnp.array([0.0, 0.5], dtype=jnp.float32),
    )
    custom_ts = jnp.linspace(
        0.0, config.duration / 2.0, config.num_samples, dtype=jnp.float32
    )
    ts, states = integrate_system(config, _zero_vector_field, ts=custom_ts)

    chex.assert_trees_all_close(ts, custom_ts)
    chex.assert_shape(states, (config.num_samples, config.initial_state.shape[0]))
