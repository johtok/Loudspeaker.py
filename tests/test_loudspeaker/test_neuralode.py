from __future__ import annotations

import itertools

import chex
import jax.numpy as jnp
import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import optax
import pytest
from diffrax import PIDController

from loudspeaker.msd_sim import MSDConfig, simulate_msd_system
from loudspeaker.neuralode import (
    LinearLoudspeakerModel,
    LinearMSDModel,
    LoudspeakerConfig,
    NeuralODE,
    ReservoirMSDModel,
    build_loss_fn,
    plot_neural_ode_loss,
    plot_neural_ode_predictions,
    predict_neural_ode,
    solve_with_model,
    train_model,
    train_neural_ode,
)
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
    assert len(history) == 4
    assert history[0] == 0.0
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
    assert len(history) == 3


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
    assert history[1] == pytest.approx(baseline_loss, rel=1e-6)
    assert history[-1] <= history[1]


def test_solve_with_model_runs_with_pid_controller():
    config = MSDConfig(num_samples=16, sample_rate=200.0)
    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    forcing = build_control_signal(ts, jnp.ones_like(ts))
    model = LinearMSDModel(config=config)
    controller = PIDController(rtol=1e-4, atol=1e-4)
    states = solve_with_model(
        model,
        ts,
        forcing,
        config.initial_state,
        config.dt,
        stepsize_controller=controller,
        rtol=1e-4,
        atol=1e-4,
    )
    chex.assert_shape(states, (config.num_samples, 2))


def test_linear_loudspeaker_model_solves():
    config = LoudspeakerConfig(num_samples=64, sample_rate=8000.0)
    ts = jnp.linspace(0.0, (config.num_samples - 1) * config.dt, config.num_samples, dtype=jnp.float32)
    forcing = build_control_signal(ts, jnp.cos(2 * jnp.pi * 50.0 * ts))
    model = LinearLoudspeakerModel(config=config, key=jr.PRNGKey(42))
    states = solve_with_model(model, ts, forcing, config.initial_state, config.dt, rtol=1e-4, atol=1e-4)
    chex.assert_shape(states, (config.num_samples, 3))


def test_reservoir_model_produces_expected_state_dimension():
    num_samples = 20
    ts = jnp.linspace(0.0, 0.019, num_samples, dtype=jnp.float32)
    forcing = build_control_signal(ts, jnp.ones_like(ts))
    model = ReservoirMSDModel(state_size=4, key=jr.PRNGKey(0))
    initial_state = jnp.zeros(4, dtype=jnp.float32)
    states = solve_with_model(model, ts, forcing, initial_state, dt=ts[1] - ts[0])
    chex.assert_shape(states, (num_samples, 4))


def test_train_neural_ode_wrapper_records_history():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (control.values[None, ...], reference[None, ...])
    dataloader = itertools.repeat(batch)
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    neural_ode = NeuralODE(
        model=LinearMSDModel(config=config),
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=1e-2),
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=3,
    )
    trained = train_neural_ode(neural_ode, dataloader)
    assert trained.history
    assert len(trained.history) == 4


def test_predict_neural_ode_returns_predictions_and_targets():
    config = MSDConfig(num_samples=6)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (control.values[None, ...], reference[None, ...])
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    neural_ode = NeuralODE(
        model=LinearMSDModel(config=config),
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=1e-2),
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=1,
    )
    predictions, targets = predict_neural_ode(neural_ode, iter([batch]), max_batches=1)
    assert predictions.shape == targets.shape
    assert predictions.shape[0] == 1


def test_plot_neural_ode_loss_returns_axis():
    config = MSDConfig(num_samples=4)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (control.values[None, ...], reference[None, ...])
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    neural_ode = NeuralODE(
        model=LinearMSDModel(config=config),
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=1e-2),
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=2,
    )
    neural_ode.history = [0.0, 0.1, 0.05]
    ax = plot_neural_ode_loss(neural_ode)
    assert ax is not None
    assert len(ax.lines) == 1


def test_plot_neural_ode_predictions_returns_axes():
    config = MSDConfig(num_samples=5)
    control = _zero_control(config)
    reference = jnp.zeros((config.num_samples, 2), dtype=jnp.float32)
    batch = (control.values[None, ...], reference[None, ...])
    loss_fn = build_loss_fn(
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
    )
    neural_ode = NeuralODE(
        model=LinearMSDModel(config=config),
        loss_fn=loss_fn,
        optimizer=optax.sgd(learning_rate=1e-2),
        ts=control.ts,
        initial_state=config.initial_state,
        dt=config.dt,
        num_steps=1,
    )
    axes = plot_neural_ode_predictions(neural_ode, iter([batch]), max_batches=1)
    assert isinstance(axes, tuple)
    assert len(axes) == 2
    assert all(ax is not None for ax in axes)
