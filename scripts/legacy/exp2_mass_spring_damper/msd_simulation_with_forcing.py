#!/usr/bin/env python3
"""
Mass-Spring-Damper Simulation with Forcing - Comprehensive Implementation
========================================================================

This script implements a comprehensive mass-spring-damper simulation with external forcing,
combining features from multiple referenced files. It includes:

1. Numerical ODE Solution with advanced diffrax solvers (Kvaerno5 for stiff systems)
2. External forcing integration following diffrax patterns
3. Pink noise forcing generation inspired by Julia implementations
4. 3D phase space visualizations and normalized phase plots
5. Comprehensive data generation and visualization pipeline

Key Features:
- Advanced solver selection (Kvaerno5 for stiff systems, Tsit5 for non-stiff)
- Pink noise generation with bandpass filtering
- 3D phase space visualization (position-velocity-acceleration)
- Normalized phase plots with multiple scaling methods
- Comprehensive error handling and solver statistics
- 64-bit precision support for numerical accuracy

Requirements:
- diffrax, equinox, jax, jax.numpy, jax.scipy
- matplotlib, numpy
- scipy (for pink noise generation)
- optax (optional, for future ML extensions)

Usage:
    python scripts/exp2_mass_spring_damper/msd_simulation_with_forcing.py
    # Or run sections individually in Jupyter or IDE with #%% support
"""

# %%
# Core imports and setup
import math
import time
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from scipy import signal

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)

print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())
print("64-bit precision enabled:", jax.config.jax_enable_x64)


# %%
# Configuration and enums
class SolverType(Enum):
    """Available ODE solver types."""

    KVAERNO5 = "Kvaerno5"  # For stiff systems
    TSIT5 = "Tsit5"  # For non-stiff systems
    DOPRI5 = "Dopri5"  # For general purpose


class ForcingType(Enum):
    """Available forcing signal types."""

    PINK_NOISE = "pink_noise"
    SINE = "sine"
    COMPLEX_SINE = "complex_sine"
    CHIRP = "chirp"
    STEP = "step"


class NormalizationType(Enum):
    """Available normalization methods."""

    STANDARDIZE = "standardize"  # (x - mean) / std
    MINMAX = "minmax"  # (x - min) / (max - min)
    UNIT_VECTOR = "unit_vector"  # x / ||x||
    NONE = "none"


@dataclass
class MSDConfig:
    """Configuration class for mass-spring-damper simulation."""

    # Physical parameters
    mass: float = 0.05  # kg
    natural_frequency: float = 25.0  # Hz
    damping_ratio: float = 0.01

    # Simulation parameters
    sample_rate: int = 1000  # Hz
    simulation_time: float = 0.1  # seconds
    initial_conditions: Tuple[float, float] = (0.0, 0.0)  # [position, velocity]

    # Forcing parameters
    forcing_type: ForcingType = ForcingType.PINK_NOISE
    forcing_amplitude: float = 1.0
    frequency_range: Tuple[float, float] = (0.01, 400.0)  # Hz
    pink_noise_exponent: float = 1.0  # For 1/f^alpha noise

    # Solver parameters
    solver_type: SolverType = SolverType.KVAERNO5
    rtol: float = 1e-8
    atol: float = 1e-8
    dt0: Optional[float] = None

    # Visualization parameters
    normalize_plots: bool = True
    normalization_type: NormalizationType = NormalizationType.STANDARDIZE
    save_plots: bool = True
    plot_format: str = "png"

    # Data generation parameters
    batch_size: int = 10
    seed: int = 1234

    @property
    def dt(self) -> float:
        """Time step."""
        return 1.0 / self.sample_rate

    @property
    def num_steps(self) -> int:
        """Number of time steps."""
        return int(self.simulation_time * self.sample_rate)

    @property
    def omega_n(self) -> float:
        """Natural frequency in rad/s."""
        return self.natural_frequency * 2 * math.pi

    @property
    def stiffness(self) -> float:
        """Spring stiffness."""
        return self.mass * self.omega_n**2

    @property
    def damping_coefficient(self) -> float:
        """Damping coefficient."""
        return 2 * self.damping_ratio * self.mass * self.omega_n

    @property
    def time_vector(self) -> jnp.ndarray:
        """Time vector for simulation."""
        return jnp.linspace(0, self.simulation_time, self.num_steps)


# %%
# Pink noise generation
class PinkNoiseGenerator:
    """Generates pink noise (1/f noise) with configurable parameters."""

    def __init__(self, config: MSDConfig):
        self.config = config
        self.key = jr.PRNGKey(config.seed)

    def generate_pink_noise(
        self, length: int, key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """
        Generate pink noise using the Voss-McCartney algorithm.

        Args:
            length: Length of the noise signal
            key: JAX random key

        Returns:
            Pink noise signal
        """
        if key is None:
            self.key, key = jr.split(self.key)

        # Generate white noise
        white_noise = jr.normal(key, (length,))

        # Apply 1/f filtering in frequency domain
        frequencies = jnp.fft.fftfreq(length, self.config.dt)
        # Avoid division by zero
        mask = frequencies > 1e-10
        pink_spectrum = jnp.ones(length, dtype=complex)
        pink_spectrum = pink_spectrum.at[mask].set(
            1.0 / (frequencies[mask] ** (self.config.pink_noise_exponent / 2))
        )

        # Apply bandpass filtering
        f_low, f_high = self.config.frequency_range
        bandpass_mask = (jnp.abs(frequencies) >= f_low) & (
            jnp.abs(frequencies) <= f_high
        )
        pink_spectrum = pink_spectrum * bandpass_mask

        # Transform back to time domain
        pink_noise = jnp.fft.ifft(pink_spectrum * jnp.fft.fft(white_noise)).real

        # Normalize and scale
        pink_noise = pink_noise / jnp.std(pink_noise) * self.config.forcing_amplitude

        return pink_noise

    def generate_forcing_signal(
        self, length: int, key: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        """Generate forcing signal based on configured type."""
        if self.config.forcing_type == ForcingType.PINK_NOISE:
            return self.generate_pink_noise(length, key)
        elif self.config.forcing_type == ForcingType.SINE:
            t = jnp.linspace(0, self.config.simulation_time, length)
            f = (self.config.frequency_range[0] + self.config.frequency_range[1]) / 2
            return self.config.forcing_amplitude * jnp.sin(2 * math.pi * f * t)
        elif self.config.forcing_type == ForcingType.COMPLEX_SINE:
            t = jnp.linspace(0, self.config.simulation_time, length)
            f1 = self.config.frequency_range[0] + 5.0
            f2 = self.config.frequency_range[1] - 50.0
            return self.config.forcing_amplitude * (
                jnp.sin(2 * math.pi * f1 * t) + 0.5 * jnp.sin(2 * math.pi * f2 * t)
            )
        elif self.config.forcing_type == ForcingType.CHIRP:
            t = jnp.linspace(0, self.config.simulation_time, length)
            f0, f1 = self.config.frequency_range
            return self.config.forcing_amplitude * signal.chirp(
                t, f0, self.config.simulation_time, f1
            )
        elif self.config.forcing_type == ForcingType.STEP:
            return jnp.ones(length) * self.config.forcing_amplitude
        else:
            raise ValueError(f"Unknown forcing type: {self.config.forcing_type}")


# %%
# Normalization functions
class Normalizer:
    """Provides various normalization methods for simulation data."""

    @staticmethod
    def standardize(
        x: jnp.ndarray, eps: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Standardize: (x - mean) / std

        Each column (trajectory) is normalized by its own mean and standard deviation.
        For state data with columns [position, velocity, acceleration], each will be
        normalized independently: position/std(position), velocity/std(velocity),
        acceleration/std(acceleration).

        This ensures all trajectories are on the same scale for proper comparison
        and visualization.

        Args:
            x: Input array with shape (n_samples, n_features)
            eps: Small value to avoid division by zero

        Returns:
            normalized: Standardized array
            mean: Mean values used for normalization
            std: Standard deviation values used for normalization
        """
        mean = jnp.mean(x, axis=0, keepdims=True)
        std = jnp.std(x, axis=0, keepdims=True)
        normalized = (x - mean) / (std + eps)
        return normalized, mean, std

    @staticmethod
    def minmax(
        x: jnp.ndarray, eps: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """MinMax normalization: (x - min) / (max - min)"""
        min_val = jnp.min(x, axis=0, keepdims=True)
        max_val = jnp.max(x, axis=0, keepdims=True)
        normalized = (x - min_val) / (max_val - min_val + eps)
        return normalized, min_val, max_val

    @staticmethod
    def unit_vector(
        x: jnp.ndarray, eps: float = 1e-8
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Unit vector normalization: x / ||x||"""
        norm = jnp.linalg.norm(x, axis=1, keepdims=True)
        normalized = x / (norm + eps)
        return normalized, norm

    @staticmethod
    def normalize_data(
        data: jnp.ndarray, method: NormalizationType = NormalizationType.STANDARDIZE
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply normalization and return normalized data with parameters."""
        if method == NormalizationType.NONE:
            return data, {}
        elif method == NormalizationType.STANDARDIZE:
            normalized, mean, std = Normalizer.standardize(data)
            return normalized, {"mean": mean, "std": std}
        elif method == NormalizationType.MINMAX:
            normalized, min_val, max_val = Normalizer.minmax(data)
            return normalized, {"min": min_val, "max": max_val}
        elif method == NormalizationType.UNIT_VECTOR:
            normalized, norm = Normalizer.unit_vector(data)
            return normalized, {"norm": norm}
        else:
            raise ValueError(f"Unknown normalization method: {method}")


# %%
# MSD System Definition
class MassSpringDamperSystem(eqx.Module):
    """Mass-spring-damper system with external forcing."""

    mass: float
    stiffness: float
    damping_coefficient: float

    def __init__(self, config: MSDConfig):
        self.mass = config.mass
        self.stiffness = config.stiffness
        self.damping_coefficient = config.damping_coefficient

    def __call__(
        self, t: float, state: jnp.ndarray, forcing_interp: Any
    ) -> jnp.ndarray:
        """
        ODE function for mass-spring-damper system with forcing.

        Args:
            t: Time
            state: [position, velocity]
            forcing_interp: Interpolated forcing function

        Returns:
            Derivatives [velocity, acceleration]
        """
        position, velocity = state[0], state[1]

        # Evaluate forcing at current time
        forcing = forcing_interp.evaluate(t) if forcing_interp is not None else 0.0

        # Equations of motion
        acceleration = (
            forcing - self.damping_coefficient * velocity - self.stiffness * position
        ) / self.mass

        return jnp.array([velocity, acceleration])


# %%
# Simulation core
class MSDSimulator:
    """Main simulator for mass-spring-damper systems with forcing."""

    def __init__(self, config: MSDConfig):
        self.config = config
        self.system = MassSpringDamperSystem(config)
        self.noise_generator = PinkNoiseGenerator(config)

        # Select solver based on configuration
        if config.solver_type == SolverType.KVAERNO5:
            self.solver = diffrax.Kvaerno5()
        elif config.solver_type == SolverType.TSIT5:
            self.solver = diffrax.Tsit5()
        elif config.solver_type == SolverType.DOPRI5:
            self.solver = diffrax.Dopri5()
        else:
            raise ValueError(f"Unknown solver type: {config.solver_type}")

    def simulate_single(
        self,
        forcing_signal: Optional[jnp.ndarray] = None,
        initial_state: Optional[jnp.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Simulate a single MSD system response.

        Args:
            forcing_signal: External forcing signal
            initial_state: Initial [position, velocity]

        Returns:
            Dictionary with simulation results
        """
        # Setup initial conditions
        if initial_state is None:
            initial_state = jnp.array(self.config.initial_conditions)

        # Generate forcing signal if not provided
        if forcing_signal is None:
            forcing_signal = self.noise_generator.generate_forcing_signal(
                self.config.num_steps
            )

        # Create interpolation for forcing signal
        time_vector = self.config.time_vector
        coeffs = diffrax.backward_hermite_coefficients(time_vector, forcing_signal)
        forcing_interp = diffrax.CubicInterpolation(time_vector, coeffs)

        # Setup ODE solve
        term = diffrax.ODETerm(self.system)
        stepsize_controller = diffrax.PIDController(
            rtol=self.config.rtol, atol=self.config.atol
        )
        saveat = diffrax.SaveAt(ts=time_vector)

        # Set initial time step if specified
        dt0 = self.config.dt0 if self.config.dt0 is not None else self.config.dt * 0.1

        try:
            # Solve ODE
            start_time = time.time()
            solution = diffrax.diffeqsolve(
                term,
                self.solver,
                t0=time_vector[0],
                t1=time_vector[-1],
                dt0=dt0,
                y0=initial_state,
                args=forcing_interp,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
            )
            solve_time = time.time() - start_time

            # Extract results
            position = solution.ys[:, 0]
            velocity = solution.ys[:, 1]
            acceleration = jnp.gradient(velocity, self.config.dt)

            # Compute forcing values at save points
            forcing_values = jax.vmap(forcing_interp.evaluate)(time_vector)

            # Calculate solver statistics
            stats = {
                "num_steps": solution.stats["num_steps"],
                "num_accepted_steps": solution.stats["num_accepted_steps"],
                "num_rejected_steps": solution.stats["num_rejected_steps"],
                "solve_time": solve_time,
                "successful": True,
            }

            # Check for numerical issues
            if jnp.any(jnp.isnan(position)) or jnp.any(jnp.isnan(velocity)):
                warnings.warn("NaN values detected in simulation results")
                stats["successful"] = False

            return {
                "time": time_vector,
                "position": position,
                "velocity": velocity,
                "acceleration": acceleration,
                "forcing": forcing_values,
                "state_trajectory": solution.ys,
                "stats": stats,
                "config": self.config,
            }

        except Exception as e:
            warnings.warn(f"Simulation failed: {str(e)}")
            return {
                "time": time_vector,
                "position": jnp.zeros_like(time_vector),
                "velocity": jnp.zeros_like(time_vector),
                "acceleration": jnp.zeros_like(time_vector),
                "forcing": forcing_signal,
                "state_trajectory": jnp.zeros((len(time_vector), 2)),
                "stats": {"successful": False, "error": str(e)},
                "config": self.config,
            }

    def simulate_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate multiple forcing signals in batch.

        Args:
            batch_size: Number of simulations to run

        Returns:
            Dictionary with batch simulation results
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        # Generate batch of forcing signals
        forcing_signals = []
        for i in range(batch_size):
            key = jr.fold_in(jr.PRNGKey(self.config.seed), i)
            forcing = self.noise_generator.generate_forcing_signal(
                self.config.num_steps, key
            )
            forcing_signals.append(forcing)

        forcing_signals = jnp.stack(forcing_signals)

        # Simulate each forcing signal
        results = []
        for i in range(batch_size):
            result = self.simulate_single(forcing_signals[i])
            results.append(result)

        # Stack results
        batch_results = {
            "time": results[0]["time"],
            "positions": jnp.stack([r["position"] for r in results]),
            "velocities": jnp.stack([r["velocity"] for r in results]),
            "accelerations": jnp.stack([r["acceleration"] for r in results]),
            "forcings": jnp.stack([r["forcing"] for r in results]),
            "stats": [r["stats"] for r in results],
            "config": self.config,
        }

        return batch_results


# %%
# Visualization functions
class MSDVisualizer:
    """Comprehensive visualization tools for MSD simulation results."""

    def __init__(self, config: MSDConfig):
        self.config = config
        self.normalizer = Normalizer()

    def plot_3d_phase_space(
        self, result: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Create 3D phase space plot of position-velocity-acceleration."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Extract data
        pos = result["position"]
        vel = result["velocity"]
        acc = result["acceleration"]

        # Create 3D plot
        scatter = ax.scatter(
            pos, vel, acc, c=result["time"], cmap="viridis", alpha=0.7, s=10
        )

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_zlabel("Acceleration")
        ax.set_title("3D Phase Space: Position-Velocity-Acceleration")

        # Add colorbar
        plt.colorbar(scatter, ax=ax, label="Time")

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"3D phase space plot saved as {save_path}")

        plt.show()

    def plot_normalized_phase_plots(
        self, result: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Create normalized phase plots with separate subplots."""
        # Prepare data for normalization
        state_data = jnp.column_stack(
            [result["position"], result["velocity"], result["acceleration"]]
        )

        # Normalize data
        normalized_data, norm_params = Normalizer.normalize_data(
            state_data, self.config.normalization_type
        )

        pos_norm, vel_norm, acc_norm = (
            normalized_data[:, 0],
            normalized_data[:, 1],
            normalized_data[:, 2],
        )

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Position-Velocity phase plot
        axes[0, 0].plot(pos_norm, vel_norm, "b-", linewidth=1, alpha=0.8)
        axes[0, 0].set_xlabel("Normalized Position")
        axes[0, 0].set_ylabel("Normalized Velocity")
        axes[0, 0].set_title("Position-Velocity Phase Plot")
        axes[0, 0].grid(True, alpha=0.3)

        # Position-Acceleration phase plot
        axes[0, 1].plot(pos_norm, acc_norm, "r-", linewidth=1, alpha=0.8)
        axes[0, 1].set_xlabel("Normalized Position")
        axes[0, 1].set_ylabel("Normalized Acceleration")
        axes[0, 1].set_title("Position-Acceleration Phase Plot")
        axes[0, 1].grid(True, alpha=0.3)

        # Velocity-Acceleration phase plot
        axes[1, 0].plot(vel_norm, acc_norm, "g-", linewidth=1, alpha=0.8)
        axes[1, 0].set_xlabel("Normalized Velocity")
        axes[1, 0].set_ylabel("Normalized Acceleration")
        axes[1, 0].set_title("Velocity-Acceleration Phase Plot")
        axes[1, 0].grid(True, alpha=0.3)

        # Time series comparison
        axes[1, 1].plot(result["time"], pos_norm, "b-", label="Position", alpha=0.7)
        axes[1, 1].plot(result["time"], vel_norm, "r-", label="Velocity", alpha=0.7)
        axes[1, 1].plot(result["time"], acc_norm, "g-", label="Acceleration", alpha=0.7)
        axes[1, 1].set_xlabel("Time")
        axes[1, 1].set_ylabel("Normalized Values")
        axes[1, 1].set_title("Normalized Time Series")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Normalized phase plots saved as {save_path}")

        plt.show()

    def plot_time_domain(self, result: Dict[str, Any], save_path: Optional[str] = None):
        """Plot forcing signal and system response over time."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        time = result["time"]

        # Forcing signal
        axes[0].plot(time, result["forcing"], "r-", linewidth=1.5)
        axes[0].set_ylabel("Forcing")
        axes[0].set_title("External Forcing Signal")
        axes[0].grid(True, alpha=0.3)

        # Position response
        axes[1].plot(time, result["position"], "b-", linewidth=1.5)
        axes[1].set_ylabel("Position")
        axes[1].set_title("Position Response")
        axes[1].grid(True, alpha=0.3)

        # Velocity response
        axes[2].plot(time, result["velocity"], "g-", linewidth=1.5)
        axes[2].set_ylabel("Velocity")
        axes[2].set_xlabel("Time")
        axes[2].set_title("Velocity Response")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Time domain plot saved as {save_path}")

        plt.show()

    def plot_comparison(
        self, batch_results: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Compare multiple simulation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        time = batch_results["time"]
        positions = batch_results["positions"]
        velocities = batch_results["velocities"]
        forcings = batch_results["forcings"]

        # Forcing signals comparison
        for i in range(min(5, len(forcings))):  # Show up to 5 forcing signals
            axes[0, 0].plot(time, forcings[i], alpha=0.7, label=f"Signal {i + 1}")
        axes[0, 0].set_ylabel("Forcing")
        axes[0, 0].set_title("Forcing Signals Comparison")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Position responses comparison
        for i in range(min(5, len(positions))):
            axes[0, 1].plot(time, positions[i], alpha=0.7, label=f"Trial {i + 1}")
        axes[0, 1].set_ylabel("Position")
        axes[0, 1].set_title("Position Responses Comparison")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Velocity responses comparison
        for i in range(min(5, len(velocities))):
            axes[1, 0].plot(time, velocities[i], alpha=0.7, label=f"Trial {i + 1}")
        axes[1, 0].set_ylabel("Velocity")
        axes[1, 0].set_xlabel("Time")
        axes[1, 0].set_title("Velocity Responses Comparison")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Phase plot comparison
        for i in range(min(5, len(positions))):
            axes[1, 1].plot(
                positions[i], velocities[i], alpha=0.7, label=f"Trial {i + 1}"
            )
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Velocity")
        axes[1, 1].set_title("Phase Plot Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Comparison plot saved as {save_path}")

        plt.show()

    def plot_frequency_analysis(
        self, result: Dict[str, Any], save_path: Optional[str] = None
    ):
        """Perform frequency analysis using FFT."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        time = result["time"]
        dt = time[1] - time[0]
        fs = 1.0 / dt

        # Forcing signal FFT
        forcing_fft = jnp.fft.fft(result["forcing"])
        freqs = jnp.fft.fftfreq(len(time), dt)
        forcing_magnitude = jnp.abs(forcing_fft)

        axes[0, 0].plot(
            freqs[: len(freqs) // 2], forcing_magnitude[: len(freqs) // 2], "r-"
        )
        axes[0, 0].set_xlabel("Frequency (Hz)")
        axes[0, 0].set_ylabel("Magnitude")
        axes[0, 0].set_title("Forcing Signal Spectrum")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim(0, fs / 2)

        # Position response FFT
        position_fft = jnp.fft.fft(result["position"])
        position_magnitude = jnp.abs(position_fft)

        axes[0, 1].plot(
            freqs[: len(freqs) // 2], position_magnitude[: len(freqs) // 2], "b-"
        )
        axes[0, 1].set_xlabel("Frequency (Hz)")
        axes[0, 1].set_ylabel("Magnitude")
        axes[0, 1].set_title("Position Response Spectrum")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim(0, fs / 2)

        # Transfer function estimation (simplified)
        if jnp.any(result["forcing"]):  # Avoid division by zero
            tf_magnitude = position_magnitude / (forcing_magnitude + 1e-10)
            axes[1, 0].plot(
                freqs[: len(freqs) // 2],
                20 * jnp.log10(tf_magnitude[: len(freqs) // 2]),
                "g-",
            )
            axes[1, 0].set_xlabel("Frequency (Hz)")
            axes[1, 0].set_ylabel("Magnitude (dB)")
            axes[1, 0].set_title("Estimated Transfer Function")
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xlim(0, fs / 2)

        # Phase response
        tf_phase = jnp.angle(position_fft) - jnp.angle(forcing_fft)
        axes[1, 1].plot(freqs[: len(freqs) // 2], tf_phase[: len(freqs) // 2], "m-")
        axes[1, 1].set_xlabel("Frequency (Hz)")
        axes[1, 1].set_ylabel("Phase (radians)")
        axes[1, 1].set_title("Transfer Function Phase")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, fs / 2)

        plt.tight_layout()

        if save_path and self.config.save_plots:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Frequency analysis plot saved as {save_path}")

        plt.show()


# %%
# Main execution functions
def run_single_simulation(config: MSDConfig) -> Dict[str, Any]:
    """Run a single MSD simulation with visualization."""
    print("Running single MSD simulation...")
    print(f"Configuration: {config.__dict__}")

    # Initialize simulator
    simulator = MSDSimulator(config)

    # Run simulation
    result = simulator.simulate_single()

    # Check if simulation was successful
    if not result["stats"]["successful"]:
        print("Warning: Simulation failed!")
        return result

    print("Simulation completed successfully!")
    print(f"Solver statistics: {result['stats']}")

    # Initialize visualizer
    visualizer = MSDVisualizer(config)

    # Create visualizations
    print("Generating visualizations...")

    visualizer.plot_time_domain(result, "msd_time_domain.png")
    visualizer.plot_3d_phase_space(result, "msd_3d_phase_space.png")
    visualizer.plot_normalized_phase_plots(result, "msd_normalized_phase.png")
    visualizer.plot_frequency_analysis(result, "msd_frequency_analysis.png")

    return result


def run_batch_simulation(config: MSDConfig) -> Dict[str, Any]:
    """Run batch MSD simulations with comparison visualization."""
    print(f"Running batch simulation with {config.batch_size} trials...")

    # Initialize simulator
    simulator = MSDSimulator(config)

    # Run batch simulation
    batch_results = simulator.simulate_batch()

    # Check results
    successful_simulations = sum(
        1 for stats in batch_results["stats"] if stats.get("successful", False)
    )
    print(f"Successful simulations: {successful_simulations}/{config.batch_size}")

    # Initialize visualizer
    visualizer = MSDVisualizer(config)

    # Create comparison visualization
    visualizer.plot_comparison(batch_results, "msd_batch_comparison.png")

    return batch_results


def demonstrate_solver_comparison():
    """Demonstrate different solvers on the same problem."""
    print("Demonstrating solver comparison...")

    # Create configurations for different solvers
    solvers_to_test = [SolverType.KVAERNO5, SolverType.TSIT5, SolverType.DOPRI5]
    results = {}

    for solver in solvers_to_test:
        config = MSDConfig(
            solver_type=solver,
            rtol=1e-6,
            atol=1e-6,
            simulation_time=0.05,  # Shorter time for comparison
        )

        simulator = MSDSimulator(config)
        result = simulator.simulate_single()

        results[solver.value] = {"result": result, "config": config}

        print(f"{solver.value}: {result['stats']}")

    # Compare results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Time series comparison
    for solver_name, data in results.items():
        result = data["result"]
        axes[0, 0].plot(
            result["time"], result["position"], label=solver_name, linewidth=2
        )
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("Position")
    axes[0, 0].set_title("Position Response Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Phase plot comparison
    for solver_name, data in results.items():
        result = data["result"]
        axes[0, 1].plot(
            result["position"], result["velocity"], label=solver_name, linewidth=2
        )
    axes[0, 1].set_xlabel("Position")
    axes[0, 1].set_ylabel("Velocity")
    axes[0, 1].set_title("Phase Plot Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Solver statistics comparison
    solver_names = list(results.keys())
    step_counts = [
        results[name]["result"]["stats"].get("num_steps", 0) for name in solver_names
    ]
    solve_times = [
        results[name]["result"]["stats"].get("solve_time", 0) for name in solver_names
    ]

    axes[1, 0].bar(solver_names, step_counts, alpha=0.7)
    axes[1, 0].set_ylabel("Number of Steps")
    axes[1, 0].set_title("Solver Step Counts")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(solver_names, solve_times, alpha=0.7)
    axes[1, 1].set_ylabel("Solve Time (s)")
    axes[1, 1].set_title("Solver Times")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("solver_comparison.png", dpi=300, bbox_inches="tight")
    print("Solver comparison plot saved as solver_comparison.png")
    plt.show()

    return results


def main():
    """Main execution function demonstrating all features."""
    print("Mass-Spring-Damper Simulation with Forcing")
    print("=" * 50)

    # Test 1: Single simulation with default settings
    print("\n1. Single Simulation Test")
    config1 = MSDConfig()
    result1 = run_single_simulation(config1)

    # Test 2: Batch simulation
    print("\n2. Batch Simulation Test")
    config2 = MSDConfig(batch_size=5, simulation_time=0.05)
    result2 = run_batch_simulation(config2)

    # Test 3: Different forcing types
    print("\n3. Forcing Type Comparison")
    forcing_types = [ForcingType.PINK_NOISE, ForcingType.SINE, ForcingType.COMPLEX_SINE]

    fig, axes = plt.subplots(
        len(forcing_types), 2, figsize=(15, 4 * len(forcing_types))
    )
    if len(forcing_types) == 1:
        axes = axes.reshape(1, -1)

    for i, forcing_type in enumerate(forcing_types):
        config = MSDConfig(
            forcing_type=forcing_type, simulation_time=0.05, batch_size=1
        )

        simulator = MSDSimulator(config)
        result = simulator.simulate_single()

        # Plot forcing signal
        axes[i, 0].plot(result["time"], result["forcing"], "r-", linewidth=2)
        axes[i, 0].set_xlabel("Time")
        axes[i, 0].set_ylabel("Forcing")
        axes[i, 0].set_title(f"{forcing_type.value} Forcing Signal")
        axes[i, 0].grid(True, alpha=0.3)

        # Plot response
        axes[i, 1].plot(
            result["time"], result["position"], "b-", linewidth=2, label="Position"
        )
        axes[i, 1].plot(
            result["time"], result["velocity"], "g-", linewidth=2, label="Velocity"
        )
        axes[i, 1].set_xlabel("Time")
        axes[i, 1].set_ylabel("Response")
        axes[i, 1].set_title(f"{forcing_type.value} Response")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("forcing_types_comparison.png", dpi=300, bbox_inches="tight")
    print("Forcing types comparison plot saved as forcing_types_comparison.png")
    plt.show()

    # Test 4: Solver comparison
    print("\n4. Solver Comparison Test")
    solver_results = demonstrate_solver_comparison()

    print("\nAll tests completed successfully!")
    print("Check the generated plots for visualization results.")


# %%
# Example usage and testing
if __name__ == "__main__":
    # You can run individual tests by uncommenting the desired sections:

    # Basic single simulation
    # config = MSDConfig()
    # result = run_single_simulation(config)

    # Batch simulation
    # config = MSDConfig(batch_size=3, simulation_time=0.05)
    # batch_result = run_batch_simulation(config)

    # Solver comparison
    # solver_results = demonstrate_solver_comparison()

    # Full demonstration
    main()
