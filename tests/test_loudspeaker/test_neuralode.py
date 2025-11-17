from __future__ import annotations

import itertools

import chex
import jax.numpy as jnp
import jax.random as jr
import optax
import pytest

from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.neuralode import LinearMSDModel, build_loss_fn, train_model
from loudspeaker.testsignals import build_control_signal


def _zero_control(config: MSDConfig):
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    return build_control_signal(ts, jnp.zeros_like(ts))


def test_linear_msd_model_behaves_like_matrix_multiplication():
    config = MSDConfig()
    model = LinearMSDModel(config=config)
    inputs = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    chex.assert_trees_all_close(model(inputs), model.weight @ inputs)


def test_build_loss_fn_with_defaults_matches_perfect_prediction():
    config = MSDConfig(num_samples=6)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        forcing=control,
        reference=reference,
    )
    model = LinearMSDModel(config=config)
    chex.assert_trees_all_close(loss_fn(model, None), jnp.array(0.0))


def test_build_loss_fn_handles_minibatches():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch_forcing = jnp.stack([control.values, control.values])
    batch_reference = jnp.stack([reference, reference])
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    model = LinearMSDModel(config=config)
    loss = loss_fn(model, (batch_forcing, batch_reference))
    assert loss.shape == ()


def test_build_loss_fn_accepts_custom_control_builder():
    config = MSDConfig(num_samples=4)
    calls = []

    def tracking_builder(ts, values):
        calls.append(ts)
        return build_control_signal(ts, values)

    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (control.values[None, ...], reference[None, ...])

    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        control_builder=tracking_builder,
    )
    model = LinearMSDModel(config=config)
    _ = loss_fn(model, batch)
    assert len(calls) == 1


def test_train_model_without_dataloader_records_history():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        forcing=control,
        reference=reference,
    )
    model = LinearMSDModel(config=config)
    optimizer = optax.sgd(learning_rate=1e-2)
    trained, history = train_model(
        model,
        loss_fn,
        optimizer=optimizer,
        num_steps=3,
        dataloader=None,
    )
    assert len(history) == 3
    assert trained is not None


def test_train_model_with_dataloader_consumes_batches():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (jnp.stack([control.values, control.values]), jnp.stack([reference, reference]))
    dataloader = itertools.repeat(batch)
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    model = LinearMSDModel(config=config)
    optimizer = optax.sgd(learning_rate=1e-2)
    _, history = train_model(
        model,
        loss_fn,
        optimizer=optimizer,
        num_steps=2,
        dataloader=dataloader,
    )
    assert len(history) == 2


def test_training_reduces_loss_against_true_dynamics():
    config = MSDConfig(num_samples=64, sample_rate=400.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing_values = jnp.sin(2 * jnp.pi * 5.0 * ts).astype(jnp.float32)
    control = build_control_signal(ts, forcing_values)
    reference_states = simulate_msd_system(config, control).states
    batch = (forcing_values[None, ...], reference_states[None, ...])

    loss_fn = build_loss_fn(
        ts=ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    optimizer = optax.sgd(learning_rate=1e-2)
    model = LinearMSDModel(config=config, perturbation=0.1, key=jr.PRNGKey(0))
    baseline_loss = float(loss_fn(model, batch))

    dataloader = itertools.repeat(batch)
    trained, history = train_model(
        model,
        loss_fn,
        optimizer,
        num_steps=5,
        dataloader=dataloader,
    )
    final_loss = float(loss_fn(trained, batch))
    assert final_loss <= baseline_loss
    assert history[0] == pytest.approx(baseline_loss, rel=1e-6)
    assert history[-1] <= history[0]
