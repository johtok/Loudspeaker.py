from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from loudspeaker.testsignals import (
    ControlSignal,
    build_control_signal,
    complex_tone_control,
    pink_noise_control,
)


def _series_strategy():
    """Generate pairs of (num_samples, value_array) for interpolation tests."""

    count_strategy = st.integers(min_value=3, max_value=12)
    element_strategy = st.floats(
        min_value=-5.0,
        max_value=5.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return count_strategy.flatmap(
        lambda count: st.tuples(
            st.just(count),
            hnp.arrays(
                dtype=np.float32,
                shape=(count,),
                elements=element_strategy,
            ),
        )
    )


@settings(deadline=None)
@given(_series_strategy(), st.floats(min_value=0.001, max_value=0.05))
def test_build_control_signal_roundtrip(series, dt):
    num_samples, values_np = series
    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples)
    values = jnp.asarray(values_np, dtype=jnp.float32)

    control = build_control_signal(ts, values)
    chex.assert_shape(control.values, values.shape)
    chex.assert_trees_all_close(control.evaluate_batch(ts), values, atol=1e-5, rtol=1e-5)


def test_control_signal_evaluate_batch_matches_scalar_calls():
    ts = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    values = jnp.sin(ts * jnp.pi).astype(jnp.float32)
    control = build_control_signal(ts, values)

    query_ts = ts[::2]
    expected = jnp.array([control.evaluate(float(t)) for t in query_ts])
    chex.assert_trees_all_close(control.evaluate_batch(query_ts), expected)


def test_complex_tone_control_matches_manual_sum():
    num_samples = 64
    dt = 1e-2
    freqs = jnp.array([2.0, 5.0], dtype=jnp.float32)
    amplitudes = jnp.array([0.5, 1.5], dtype=jnp.float32)
    phases = jnp.array([0.0, jnp.pi / 4.0], dtype=jnp.float32)

    control = complex_tone_control(
        num_samples=num_samples,
        dt=dt,
        frequencies=freqs,
        amplitudes=amplitudes,
        phases=phases,
    )
    manual = jnp.sum(
        amplitudes[:, None] * jnp.sin(2 * jnp.pi * freqs[:, None] * control.ts + phases[:, None]),
        axis=0,
    )
    chex.assert_trees_all_close(control.values, manual)


def test_complex_tone_control_defaults_amplitudes_and_phases():
    control = complex_tone_control(num_samples=8, dt=0.1, frequencies=(5.0,))
    chex.assert_shape(control.values, (8,))


def test_complex_tone_control_validates_length_mismatch():
    with pytest.raises(ValueError):
        complex_tone_control(
            num_samples=16,
            dt=0.01,
            frequencies=(1.0, 2.0),
            amplitudes=(1.0,),
            phases=(0.0, 0.1),
        )


def test_pink_noise_control_is_deterministic_for_fixed_key():
    pytest.importorskip("colorednoise")
    num_samples = 32
    dt = 1e-2
    key = jr.PRNGKey(0)
    band = (1.0, 20.0)
    control_a = pink_noise_control(num_samples, dt, key, band=band)
    control_b = pink_noise_control(num_samples, dt, key, band=band)
    chex.assert_trees_all_close(control_a.values, control_b.values)
    chex.assert_trees_all_close(control_a.evaluate_batch(control_a.ts), control_a.values)


def test_pink_noise_has_higher_low_frequency_power():
    pytest.importorskip("colorednoise")
    num_samples = 2048
    dt = 1.0 / 400.0
    key = jr.PRNGKey(5)
    band = (1.0, 90.0)
    control = pink_noise_control(num_samples, dt, key, band=band)
    values = np.asarray(control.values - jnp.mean(control.values))
    spectrum = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(num_samples, dt)
    power = np.abs(spectrum) ** 2
    low_mask = (freqs >= 1.0) & (freqs < 10.0)
    mid_mask = (freqs >= 10.0) & (freqs < 30.0)
    high_mask = (freqs >= 40.0) & (freqs < 80.0)
    low_power = power[low_mask].mean()
    mid_power = power[mid_mask].mean()
    high_power = power[high_mask].mean()
    assert low_power > high_power
    assert mid_power > high_power


def test_control_signal_tree_roundtrip():
    ts = jnp.linspace(0.0, 1.0, 5, dtype=jnp.float32)
    values = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float32)
    control = build_control_signal(ts, values)
    leaves, aux = control.tree_flatten()
    rebuilt = ControlSignal.tree_unflatten(aux, leaves)
    chex.assert_trees_all_close(rebuilt.values, control.values)


def test_pink_noise_control_validates_band(monkeypatch):
    pytest.importorskip("colorednoise")
    monkeypatch.setattr("loudspeaker.testsignals.powerlaw_psd_gaussian", lambda exponent, n, random_state=None: np.ones(n))
    with pytest.raises(ValueError):
        pink_noise_control(num_samples=16, dt=1e-3, key=jr.PRNGKey(0), band=(0.0, 1.0))
    with pytest.raises(ValueError):
        pink_noise_control(num_samples=16, dt=1e-3, key=jr.PRNGKey(0), band=(10.0, 5000.0))


def test_pink_noise_control_handles_filter_fallback(monkeypatch):
    pytest.importorskip("colorednoise")

    def fake_noise(exponent, n, random_state=None):
        return np.linspace(0.0, 1.0, n)

    monkeypatch.setattr("loudspeaker.testsignals.powerlaw_psd_gaussian", fake_noise)

    call_state = {"attempts": 0}

    def fake_sosfiltfilt(*args, **kwargs):
        call_state["attempts"] += 1
        raise ValueError("force fallback")

    def fake_sosfilt(sos, data):
        return np.asarray(data) + 1.0

    monkeypatch.setattr("scipy.signal.sosfiltfilt", fake_sosfiltfilt)
    monkeypatch.setattr("scipy.signal.sosfilt", fake_sosfilt)

    control = pink_noise_control(
        num_samples=8,
        dt=1e-3,
        key=jr.PRNGKey(1),
        band=(1.0, 2.0),
    )
    assert call_state["attempts"] >= 2
    chex.assert_shape(control.values, (8,))


def test_pink_noise_control_handles_zero_padlen(monkeypatch):
    pytest.importorskip("colorednoise")
    monkeypatch.setattr("loudspeaker.testsignals.powerlaw_psd_gaussian", lambda exponent, n, random_state=None: np.ones(n))
    monkeypatch.setattr("loudspeaker.testsignals.signal.butter", lambda *args, **kwargs: np.zeros((0, 6)))

    def fake_sosfiltfilt(*args, **kwargs):
        raise ValueError("force fallback")

    monkeypatch.setattr("scipy.signal.sosfiltfilt", fake_sosfiltfilt)
    monkeypatch.setattr("scipy.signal.sosfilt", lambda sos, data: np.asarray(data))

    control = pink_noise_control(
        num_samples=4,
        dt=1e-3,
        key=jr.PRNGKey(2),
        band=(1.0, 1.5),
    )
    chex.assert_shape(control.values, (4,))
