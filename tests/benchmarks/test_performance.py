from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from loudspeaker.data import build_msd_dataset
from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.neuralode import LinearMSDModel, build_loss_fn
from loudspeaker.testsignals import build_control_signal


def _deterministic_control(num_samples: int, dt: float, key: jr.PRNGKey, **kwargs):
    del key, kwargs
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples)
    values = jnp.sin(2 * jnp.pi * 5.0 * ts)
    return build_control_signal(ts, values)


@pytest.mark.benchmark(group="msd")
def test_msd_simulator_benchmark(benchmark):
    config = MSDConfig(num_samples=256, sample_rate=500.0)
    control = _deterministic_control(config.num_samples, config.dt, jr.PRNGKey(0))

    def run():
        result = simulate_msd_system(config, control)
        return jax.block_until_ready(result.states)

    benchmark(run)


@pytest.mark.benchmark(group="loss")
def test_loss_function_benchmark(benchmark):
    config = MSDConfig(num_samples=128, sample_rate=400.0)
    dataset = build_msd_dataset(
        config,
        dataset_size=4,
        key=jr.PRNGKey(1),
        forcing_fn=_deterministic_control,
    )
    model = LinearMSDModel(config)
    loss_fn = build_loss_fn(
        ts=dataset.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    batch = (dataset.forcing[:2], dataset.reference[:2])

    def run():
        loss = loss_fn(model, batch)
        return jax.block_until_ready(loss)

    benchmark(run)
