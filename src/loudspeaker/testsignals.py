from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import jax.numpy as jnp
import jax.random as jr
from diffrax import CubicInterpolation, backward_hermite_coefficients


@dataclass
class ControlSignal:
    """Callable wrapper for forcing signals sampled on a grid."""

    ts: jnp.ndarray
    values: jnp.ndarray
    interpolation: CubicInterpolation

    def evaluate(self, t: float) -> float:
        return self.interpolation.evaluate(t)


def _build_control(ts: jnp.ndarray, values: jnp.ndarray) -> ControlSignal:
    coeffs = backward_hermite_coefficients(ts, values)
    interpolation = CubicInterpolation(ts, coeffs)
    return ControlSignal(ts=ts, values=values, interpolation=interpolation)


def complex_tone_control(
    num_samples: int,
    dt: float,
    frequencies: Iterable[float] = (20.0, 24.0),
    amplitudes: Iterable[float] | None = None,
    phases: Iterable[float] | None = None,
) -> ControlSignal:
    """Complex-tone forcing sampled on the simulation grid."""

    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples)
    freqs_tuple = tuple(frequencies)
    if amplitudes is None:
        amplitudes = jnp.ones(len(freqs_tuple))
    else:
        amplitudes = jnp.array(list(amplitudes))
    if phases is None:
        phases = jnp.zeros(len(freqs_tuple))
    else:
        phases = jnp.array(list(phases))

    freqs = jnp.array(freqs_tuple)
    signal = jnp.zeros_like(ts)
    for amp, freq, phase in zip(amplitudes, freqs, phases):
        signal = signal + amp * jnp.sin(2 * jnp.pi * freq * ts + phase)
    return _build_control(ts, signal)


def pink_noise_control(
    num_samples: int,
    dt: float,
    key: jr.PRNGKey,
    band: Tuple[float, float] = (1.0, 100.0),
    exponent: float = 1.0,
    amplitude: float = 1.0,
) -> ControlSignal:
    """Band-passed pink noise forcing with cubic interpolation."""

    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples)
    white = jr.normal(key, (num_samples,))
    freqs = jnp.fft.fftfreq(num_samples, dt)
    spectrum = jnp.fft.fft(white)
    mag = jnp.where(
        freqs == 0.0,
        0.0,
        1.0 / (jnp.abs(freqs) ** (exponent / 2)),
    )
    mag = jnp.where(jnp.isfinite(mag), mag, 0.0)
    f_low, f_high = band
    mask = (jnp.abs(freqs) >= f_low) & (jnp.abs(freqs) <= f_high)
    filtered = spectrum * mag * mask
    pink_noise = jnp.fft.ifft(filtered).real
    pink_noise = pink_noise / (jnp.std(pink_noise) + 1e-8) * amplitude
    return _build_control(ts, pink_noise)
