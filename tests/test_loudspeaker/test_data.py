from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from loudspeaker.data import MSDDataset, StaticTrainingStrategy, build_msd_dataset, msd_dataloader
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.testsignals import build_control_signal


def _constant_control(num_samples: int, dt: float, key: jr.PRNGKey, band):
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
    amplitude = jr.normal(key, ()).astype(jnp.float32)
    values = jnp.ones_like(ts) * amplitude
    return build_control_signal(ts, values)


def _scaled_control(num_samples: int, dt: float, key: jr.PRNGKey, *, scale: float = 1.0, **_):
    del key  # deterministic control for testing
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
    values = jnp.ones_like(ts) * jnp.float32(scale)
    return build_control_signal(ts, values)


def test_build_msd_dataset_uses_pink_noise_control(monkeypatch):
    monkeypatch.setattr("loudspeaker.data.pink_noise_control", _constant_control)
    config = MSDConfig(num_samples=6)
    dataset = build_msd_dataset(
        config,
        dataset_size=4,
        key=jr.PRNGKey(0),
    )
    assert isinstance(dataset, MSDDataset)
    expected_ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    chex.assert_trees_all_close(dataset.ts, expected_ts)
    chex.assert_shape(dataset.forcing, (4, config.num_samples))
    chex.assert_shape(dataset.reference, (4, config.num_samples, 2))
    # Backwards-compatible tuple unpacking
    ts, forcing, reference = dataset
    chex.assert_trees_all_close(ts, dataset.ts)
    chex.assert_trees_all_close(forcing, dataset.forcing)
    chex.assert_trees_all_close(reference, dataset.reference)


def test_build_msd_dataset_validates_size(monkeypatch):
    monkeypatch.setattr("loudspeaker.data.pink_noise_control", _constant_control)
    config = MSDConfig()
    with pytest.raises(ValueError):
        build_msd_dataset(config, dataset_size=0, key=jr.PRNGKey(0))


def test_build_msd_dataset_accepts_custom_forcing_fn():
    config = MSDConfig(num_samples=4)
    dataset = build_msd_dataset(
        config,
        dataset_size=2,
        key=jr.PRNGKey(0),
        forcing_fn=_scaled_control,
        forcing_kwargs={"scale": 2.0},
    )
    assert bool(jnp.all(dataset.forcing == 2.0))


def test_msd_dataloader_without_strategy_emits_batches():
    num_samples = 5
    dataset_size = 3
    forcing = jnp.arange(dataset_size * num_samples, dtype=jnp.float32).reshape(dataset_size, num_samples)
    reference = jnp.stack([jnp.column_stack([row, row]) for row in forcing])
    loader = msd_dataloader(
        forcing,
        reference,
        batch_size=1,
        key=jr.PRNGKey(0),
    )
    batch_forcing, batch_reference = next(loader)
    chex.assert_shape(batch_forcing, (1, num_samples))
    chex.assert_shape(batch_reference, (1, num_samples, 2))
    assert any(jnp.allclose(batch_forcing[0], forcing[idx]) for idx in range(dataset_size))


def test_msd_dataloader_with_strategy_truncates_sequences():
    num_samples = 10
    dataset_size = 4
    forcing = jnp.tile(jnp.arange(num_samples, dtype=jnp.float32), (dataset_size, 1))
    reference = jnp.stack([jnp.column_stack([row, row]) for row in forcing])
    strategy = StaticTrainingStrategy(steps=1, length_fraction=0.4)
    expected_length = max(2, int(num_samples * 0.4))
    loader = msd_dataloader(
        forcing,
        reference,
        batch_size=2,
        key=jr.PRNGKey(0),
        strategy=strategy,
    )
    batch_forcing, batch_reference = next(loader)
    chex.assert_shape(batch_forcing, (2, expected_length))
    chex.assert_shape(batch_reference, (2, expected_length, 2))


def test_msd_dataloader_validates_inputs():
    forcing = jnp.zeros((2, 3))
    reference = jnp.zeros((3, 3, 2))
    with pytest.raises(ValueError):
        next(
            msd_dataloader(
                forcing,
                reference,
                batch_size=2,
                key=jr.PRNGKey(0),
            )
        )
