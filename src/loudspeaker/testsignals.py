from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from scipy import signal
from jax import tree_util

try:  # Lazy import to avoid hard dependency at import time.
    from colorednoise import powerlaw_psd_gaussian
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when needed
    powerlaw_psd_gaussian = None


@tree_util.register_pytree_node_class
@dataclass
class ControlSignal:
    """Callable wrapper for forcing signals sampled on a grid."""

    ts: jnp.ndarray
    values: jnp.ndarray
    interpolation: CubicInterpolation

    def evaluate(self, t: float) -> float:
        return self.interpolation.evaluate(t)

    def evaluate_batch(self, ts: jnp.ndarray) -> jnp.ndarray:
        """Vectorized evaluation over a batch of time samples."""
        return jax.vmap(self.interpolation.evaluate)(ts)

    def tree_flatten(self):
        children = (self.ts, self.values, self.interpolation)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        ts, values, interpolation = children
        return cls(ts=ts, values=values, interpolation=interpolation)


def build_control_signal(ts: jnp.ndarray, values: jnp.ndarray) -> ControlSignal:
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
    freqs = jnp.atleast_1d(jnp.asarray(frequencies, dtype=jnp.float64))

    if amplitudes is None:
        amplitudes = jnp.ones_like(freqs)
    else:
        amplitudes = jnp.asarray(amplitudes, dtype=jnp.float64)

    if phases is None:
        phases = jnp.zeros_like(freqs)
    else:
        phases = jnp.asarray(phases, dtype=jnp.float64)

    if amplitudes.shape != freqs.shape or phases.shape != freqs.shape:
        raise ValueError("frequencies, amplitudes, and phases must share the same length.")

    omega_t = 2 * jnp.pi * freqs[:, None] * ts[None, :] + phases[:, None]
    components = amplitudes[:, None] * jnp.sin(omega_t)
    signal = jnp.sum(components, axis=0)
    return build_control_signal(ts, signal)


def pink_noise_control(
    num_samples: int,
    dt: float,
    key: jr.PRNGKey,
    band: Tuple[float, float] = (1.0, 100.0),
    exponent: float = 1.0,
    amplitude: float = 1.0,
) -> ControlSignal:
    """Band-passed pink noise forcing built from the colorednoise library."""

    if powerlaw_psd_gaussian is None:  # pragma: no cover - informative error path
        raise ImportError(
            "colorednoise is required for pink_noise_control. Install with `pip install colorednoise`."
        )

    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples)
    seed = int(jr.randint(key, (), 0, 2**31 - 1))
    base_np = np.asarray(
        powerlaw_psd_gaussian(
            0.0 if band is None else exponent, num_samples, random_state=seed
        ),
        dtype=float,
    )

    if band is not None:
        f_low, f_high = band
        if f_low <= 0 or f_high <= f_low:
            raise ValueError("band must satisfy 0 < f_low < f_high.")
        fs = 1.0 / dt
        nyquist = fs / 2.0
        if f_high >= nyquist:
            raise ValueError(f"Upper band edge {f_high} exceeds Nyquist {nyquist}.")
        sos = signal.butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos")
        # Zero-phase IIR filtering before converting to JAX arrays.
        try:
            base_np = signal.sosfiltfilt(sos, base_np)
        except ValueError:
            padlen = max(min(num_samples - 1, 3 * sos.shape[0] * 2), 0)
            if padlen > 0:
                try:
                    base_np = signal.sosfiltfilt(sos, base_np, padlen=padlen)
                except ValueError:
                    forward = signal.sosfilt(sos, base_np)
                    base_np = signal.sosfilt(sos, forward[::-1])[::-1]
            else:
                forward = signal.sosfilt(sos, base_np)
                base_np = signal.sosfilt(sos, forward[::-1])[::-1]

    base = jnp.asarray(base_np, dtype=jnp.float64)

    scaled = base / (jnp.std(base) + 1e-8) * amplitude
    return build_control_signal(ts, scaled)
