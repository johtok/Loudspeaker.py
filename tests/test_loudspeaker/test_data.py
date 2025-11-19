from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from loudspeaker.data import (
    MSDDataset,
    StaticTrainingStrategy,
    StrategyPhase,
    TrainTestSplit,
    TrainingStrategy,
    _phase_length,
    build_loudspeaker_dataset,
    build_msd_dataset,
    msd_dataloader,
)
from loudspeaker.loudspeaker_sim import LoudspeakerConfig
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.testsignals import build_control_signal


def _constant_control(num_samples: int, dt: float, key: jr.PRNGKey, band):
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
    amplitude = jr.normal(key, ()).astype(jnp.float32)
    values = jnp.ones_like(ts) * amplitude
    return build_control_signal(ts, values)


def _scaled_control(
    num_samples: int, dt: float, key: jr.PRNGKey, *, scale: float = 1.0, **_
):
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
    expected_ts = jnp.linspace(
        0.0, config.duration, config.num_samples, dtype=jnp.float32
    )
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


def test_msd_dataset_iterator_matches_attributes(monkeypatch):
    monkeypatch.setattr("loudspeaker.data.pink_noise_control", _constant_control)
    config = MSDConfig(num_samples=5)
    dataset = build_msd_dataset(config, dataset_size=2, key=jr.PRNGKey(42))
    iterator = iter(dataset)
    chex.assert_trees_all_close(next(iterator), dataset.ts)
    chex.assert_trees_all_close(next(iterator), dataset.forcing)
    chex.assert_trees_all_close(next(iterator), dataset.reference)
    with pytest.raises(StopIteration):
        next(iterator)


def test_build_loudspeaker_dataset_matches_config(monkeypatch):
    monkeypatch.setattr("loudspeaker.data.pink_noise_control", _constant_control)
    config = LoudspeakerConfig(num_samples=8)
    ts, forcing, states = build_loudspeaker_dataset(
        config,
        dataset_size=3,
        key=jr.PRNGKey(0),
    )
    expected_ts = jnp.linspace(
        0.0, config.duration, config.num_samples, dtype=jnp.float32
    )
    chex.assert_trees_all_close(ts, expected_ts)
    chex.assert_shape(forcing, (3, config.num_samples))
    chex.assert_shape(states, (3, config.num_samples, 3))


def test_build_loudspeaker_dataset_validates_size():
    config = LoudspeakerConfig()
    with pytest.raises(ValueError):
        build_loudspeaker_dataset(config, dataset_size=0, key=jr.PRNGKey(0))


def test_build_loudspeaker_dataset_accepts_custom_forcing():
    config = LoudspeakerConfig(num_samples=10)
    _, forcing, _ = build_loudspeaker_dataset(
        config,
        dataset_size=1,
        key=jr.PRNGKey(0),
        forcing_fn=_scaled_control,
        forcing_kwargs={"scale": 3.0},
    )
    chex.assert_trees_all_close(forcing, jnp.ones_like(forcing) * 3.0)


def test_msd_dataloader_without_strategy_emits_batches():
    num_samples = 5
    dataset_size = 3
    forcing = jnp.arange(dataset_size * num_samples, dtype=jnp.float32).reshape(
        dataset_size, num_samples
    )
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
    assert any(
        jnp.allclose(batch_forcing[0], forcing[idx]) for idx in range(dataset_size)
    )


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


def test_msd_dataloader_full_batch_matches_reference():
    num_samples = 6
    dataset_size = 3
    forcing = jnp.arange(dataset_size * num_samples, dtype=jnp.float32).reshape(
        dataset_size, num_samples
    )
    reference = jnp.stack([jnp.column_stack([row * 2.0, row * 3.0]) for row in forcing])
    loader = msd_dataloader(
        forcing,
        reference,
        batch_size=dataset_size,
        key=jr.PRNGKey(1),
    )
    batch_forcing, batch_reference = next(loader)
    chex.assert_shape(batch_forcing, forcing.shape)
    chex.assert_shape(batch_reference, reference.shape)
    actual_forcing = {tuple(map(float, row)) for row in jnp.asarray(batch_forcing)}
    expected_forcing = {tuple(map(float, row)) for row in jnp.asarray(forcing)}
    assert actual_forcing == expected_forcing
    actual_reference = {
        tuple(map(float, row.ravel())) for row in jnp.asarray(batch_reference)
    }
    expected_reference = {
        tuple(map(float, row.ravel())) for row in jnp.asarray(reference)
    }
    assert actual_reference == expected_reference


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


def test_msd_dataloader_rejects_large_batches():
    forcing = jnp.zeros((2, 3))
    reference = jnp.zeros((2, 3, 2))
    loader = msd_dataloader
    with pytest.raises(ValueError):
        next(loader(forcing, reference, batch_size=3, key=jr.PRNGKey(0)))


def test_strategy_phase_validates_inputs():
    with pytest.raises(ValueError):
        StrategyPhase(steps=0, length_fraction=0.5)
    with pytest.raises(ValueError):
        StrategyPhase(steps=1, length_fraction=0.0)


def test_training_strategy_iteration_and_total_steps():
    phases = (
        StrategyPhase(steps=2, length_fraction=0.5),
        StrategyPhase(steps=1, length_fraction=1.0),
    )
    strategy = TrainingStrategy(phases)
    assert tuple(strategy) == phases
    assert strategy.total_steps == 3


def test_training_strategy_requires_phase():
    with pytest.raises(ValueError):
        TrainingStrategy(())


def test_static_training_strategy_uses_minimum_length():
    strategy = StaticTrainingStrategy(steps=1, length_fraction=0.01)
    # _phase_length enforces two samples minimum.
    assert (
        _phase_length(num_samples=10, fraction=strategy.phases[0].length_fraction) == 2
    )


def test_train_test_split_rejects_mismatched_lengths():
    forcing = jnp.zeros((3, 4))
    reference = jnp.zeros((2, 4, 2))
    with pytest.raises(ValueError):
        TrainTestSplit.from_dataset(forcing, reference, train_fraction=0.5)


def test_train_test_split_requires_two_samples():
    forcing = jnp.zeros((1, 5))
    reference = jnp.zeros((1, 5, 2))
    with pytest.raises(ValueError):
        TrainTestSplit.from_dataset(forcing, reference, train_fraction=0.9)


def test_train_test_split_evaluation_helpers():
    forcing = jnp.arange(12, dtype=jnp.float32).reshape(3, 4)
    reference = jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2)
    split = TrainTestSplit.from_dataset(forcing, reference, train_fraction=0.66)
    assert split.train_size == 2
    assert split.test_size == 1
    eval_batch = split.evaluation_batch()
    chex.assert_shape(eval_batch[0], (1, 4))
    chex.assert_shape(eval_batch[1], (1, 4, 2))
    all_batches = split.evaluation_batches()
    assert len(all_batches) == split.test_size
    chex.assert_trees_all_close(
        (all_batches[0][0], all_batches[0][1]), (eval_batch[0], eval_batch[1])
    )
