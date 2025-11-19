from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Self

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from diffrax import CubicInterpolation, backward_hermite_coefficients
from jax import config as jax_config
from jax import tree_util
from scipy import signal

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

    def evaluate(self: Self, t: float | jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(self.interpolation.evaluate(t), dtype=jnp.float32)

    def evaluate_batch(self: Self, ts: jnp.ndarray) -> jnp.ndarray:
        """Vectorized evaluation over a batch of time samples."""
        if ts is self.ts:
            return self.values
        batched_eval = eqx.filter_vmap(self.interpolation.evaluate)
        return jnp.asarray(batched_eval(ts), dtype=jnp.float32)

    def tree_flatten(
        self: Self,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray, CubicInterpolation], None]:
        children = (self.ts, self.values, self.interpolation)
        return children, None

    @classmethod
    def tree_unflatten(
        cls: type[Self],
        _aux_data: Any,
        children: tuple[jnp.ndarray, jnp.ndarray, CubicInterpolation],
    ) -> Self:
        ts, values, interpolation = children
        return cls(ts=ts, values=values, interpolation=interpolation)


def build_control_signal(ts: jnp.ndarray, values: jnp.ndarray) -> ControlSignal:
    ts32 = jnp.asarray(ts, dtype=jnp.float32)
    values32 = jnp.asarray(values, dtype=jnp.float32)
    interp_ts = ts32.astype(jnp.float64) if jax_config.jax_enable_x64 else ts32  # type: ignore
    interp_values = (
        values32.astype(jnp.float64) if jax_config.jax_enable_x64 else values32  # type: ignore
    )
    coeffs = backward_hermite_coefficients(interp_ts, interp_values)
    interpolation = CubicInterpolation(interp_ts, coeffs)
    return ControlSignal(ts=ts32, values=values32, interpolation=interpolation)


def complex_tone_control(
    num_samples: int,
    dt: float,
    frequencies: Iterable[float] = (20.0, 24.0),
    amplitudes: Iterable[float] | None = None,
    phases: Iterable[float] | None = None,
) -> ControlSignal:
    """Complex-tone forcing sampled on the simulation grid."""

    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
    freqs = jnp.atleast_1d(jnp.asarray(frequencies, dtype=jnp.float32))

    if amplitudes is None:
        amplitudes_arr = jnp.ones_like(freqs)
    else:
        amplitudes_arr = jnp.asarray(amplitudes, dtype=jnp.float32)

    if phases is None:
        phases_arr = jnp.zeros_like(freqs)
    else:
        phases_arr = jnp.asarray(phases, dtype=jnp.float32)

    if amplitudes_arr.shape != freqs.shape or phases_arr.shape != freqs.shape:
        raise ValueError(
            "frequencies, amplitudes, and phases must share the same length."
        )

    omega_t = 2 * jnp.pi * freqs[:, None] * ts[None, :] + phases_arr[:, None]
    components = amplitudes_arr[:, None] * jnp.sin(omega_t)
    signal = jnp.sum(components, axis=0)
    return build_control_signal(ts, signal)


def _normalize(values: jnp.ndarray, amplitude: float) -> jnp.ndarray:
    scale = jnp.std(values) + 1e-8
    return amplitude * values / scale


def pink_noise_control(
    num_samples: int,
    dt: float,
    key: jax.Array,
    band: tuple[float, float] | None = (1.0, 100.0),
    exponent: float = 1.0,
    amplitude: float = 1.0,
) -> ControlSignal:
    """Band-passed pink noise forcing built from the colorednoise library."""

    if powerlaw_psd_gaussian is None:  # pragma: no cover - informative error path
        raise ImportError(
            "colorednoise is required for pink_noise_control. Install with `pip install colorednoise`."
        )

    ts = jnp.linspace(0.0, dt * (num_samples - 1), num_samples, dtype=jnp.float32)
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
        sos = np.asarray(
            signal.butter(4, [f_low, f_high], btype="bandpass", fs=fs, output="sos"),
            dtype=float,
        )
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

    base = jnp.asarray(base_np, dtype=jnp.float32)

    scaled = _normalize(base, amplitude)
    return build_control_signal(ts, scaled)
