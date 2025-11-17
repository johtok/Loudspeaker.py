from __future__ import annotations

import jax.numpy as jnp

from loudspeaker.loudspeaker_sim import (
    LoudspeakerConfig,
    LoudspeakerSimulationResult,
    simulate_loudspeaker_system,
)
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
