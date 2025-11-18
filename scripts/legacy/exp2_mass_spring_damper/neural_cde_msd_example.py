#!/usr/bin/env python3
"""
Neural CDE Mass-Spring-Damper Example Script
============================================

This script demonstrates how to use diffrax for Neural Controlled Differential Equations (Neural CDEs)
for modeling mass-spring-damper systems. This version uses the comprehensive msd_simulation_with_forcing
script as its data source, providing advanced simulation capabilities with pink noise forcing,
proper normalization, and 3D phase space visualization.

Key Features:
- Uses msd_simulation_with_forcing for high-quality data generation
- Advanced pink noise forcing with configurable parameters
- Proper trajectory-wise normalization (x/std(x), v/std(v), a/std(a))
- 3D phase space analysis and visualization
- Batch simulation capabilities
- Solver comparison and performance analysis

Requirements:
- diffrax
- equinox
- jax
- jax.numpy
- optax
- matplotlib
- numpy

Usage:
    python scripts/exp2_mass_spring_damper/neural_cde_msd_example.py
    # Or run sections individually in Jupyter or IDE with #%% support
"""

# %%
# Import necessary libraries
import time
from typing import Any, Dict, Tuple

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import optax

print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())


# %%
# Configuration and hyperparameters
class Config:
    # Data generation - now using msd_simulation_with_forcing
    dataset_size = 256
    add_noise = False
    simulation_time = 0.1  # seconds

    # Mass-spring-damper parameters (from msd_simulation_with_forcing)
    natural_frequency = 25.0  # Hz
    damping_ratio = 0.01
    mass = 0.05  # kg

    # Forcing parameters
    forcing_type = (
        "pink_noise"  # Using advanced forcing from msd_simulation_with_forcing
    )
    forcing_amplitude = 1.0
    frequency_range = (0.01, 400.0)  # Hz

    # Model
    hidden_size = 8
    width_size = 12
    depth = 1

    # Training
    batch_size = 32
    learning_rate = 1e-2
    num_steps = 20
    seed = 5678

    # Visualization
    save_plots = True
    plot_format = "png"


config = Config()


# %%
# Neural CDE Model Definition
class Func(eqx.Module):
    """Vector field for the CDE."""

    mlp: eqx.nn.MLP
    data_size: int
    hidden_size: int

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size * data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            # Tanh final activation prevents model blowup
            final_activation=jnn.tanh,
            key=key,
        )

    def __call__(self, t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        return self.mlp(y).reshape(self.hidden_size, self.data_size)


class NeuralCDE(eqx.Module):
    """Neural CDE model for mass-spring-damper system identification."""

    initial: eqx.nn.MLP
    func: Func
    linear: eqx.nn.Linear

    def __init__(
        self,
        data_size: int,
        hidden_size: int,
        width_size: int,
        depth: int,
        *,
        key: jax.random.PRNGKey,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ikey, fkey, lkey = jr.split(key, 3)

        self.initial = eqx.nn.MLP(data_size, hidden_size, width_size, depth, key=ikey)
        self.func = Func(data_size, hidden_size, width_size, depth, key=fkey)
        self.linear = eqx.nn.Linear(
            hidden_size, 3, key=lkey
        )  # Output position, velocity, and acceleration

    def __call__(
        self, ts: jnp.ndarray, coeffs: Tuple, evolving_out: bool = False
    ) -> jnp.ndarray:
        """Forward pass through the Neural CDE."""
        # Create control path from coefficients
        control = diffrax.CubicInterpolation(ts, coeffs)
        term = diffrax.ControlTerm(self.func, control).to_ode()
        solver = diffrax.Tsit5()
        dt0 = None

        # Initial condition
        y0 = self.initial(control.evaluate(ts[0]))

        # Configure saving
        if evolving_out:
            saveat = diffrax.SaveAt(ts=ts)
        else:
            saveat = diffrax.SaveAt(t1=True)

        # Solve the CDE
        solution = diffrax.diffeqsolve(
            term,
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=saveat,
        )

        # Extract predictions
        if evolving_out:
            prediction = jax.vmap(lambda y: self.linear(y))(solution.ys)
        else:
            prediction = self.linear(solution.ys[-1])

        return prediction


# %%
# Import msd_simulation_with_forcing for advanced data generation
from msd_simulation_with_forcing import ForcingType
from msd_simulation_with_forcing import MSDConfig as MSDFullConfig
from msd_simulation_with_forcing import run_batch_simulation


# %%
# Data Generation using msd_simulation_with_forcing
def generate_msd_data_from_full_simulation(
    dataset_size: int, add_noise: bool, config: Config, *, key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Tuple, jnp.ndarray, jnp.ndarray, int]:
    """Generate mass-spring-damper time series data using msd_simulation_with_forcing."""

    print("Generating data using msd_simulation_with_forcing...")

    # Create msd_simulation_with_forcing config
    msd_config = MSDFullConfig(
        mass=config.mass,
        natural_frequency=config.natural_frequency,
        damping_ratio=config.damping_ratio,
        sample_rate=1000,  # Fixed sample rate for consistency
        simulation_time=config.simulation_time,
        forcing_type=ForcingType.PINK_NOISE,
        forcing_amplitude=config.forcing_amplitude,
        batch_size=dataset_size,
        normalize_plots=False,  # We'll handle normalization separately
        save_plots=False,
    )

    # Generate batch simulation data
    batch_results = run_batch_simulation(msd_config)

    # Extract data from batch results
    ts = batch_results["time"]
    forces = batch_results["forcings"]
    positions = batch_results["positions"]
    velocities = batch_results["velocities"]

    # Add acceleration (computed from velocity)
    accelerations = []
    for i in range(dataset_size):
        acc = jnp.gradient(velocities[i], ts[1] - ts[0])
        accelerations.append(acc)
    accelerations = jnp.stack(accelerations)

    # Stack responses: [position, velocity, acceleration]
    responses = jnp.stack([positions, velocities, accelerations], axis=-1)

    # Create data array with time, forcing, and responses
    data = jnp.concatenate(
        [
            ts[None, :, None].repeat(dataset_size, axis=0),  # time (broadcasted)
            forces[:, :, None],  # force input
            responses,  # position, velocity, acceleration
        ],
        axis=-1,
    )

    # Add noise if requested
    if add_noise:
        noise_key, key = jr.split(key)
        data = data + jr.normal(noise_key, data.shape) * 0.01

    # Compute interpolation coefficients
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(data[..., 0], data)

    _, _, data_size = data.shape

    print(
        f"Generated data shape: ts={ts.shape}, forces={forces.shape}, responses={responses.shape}"
    )
    print(f"Data size: {data_size}")

    return ts, coeffs, forces, responses, data_size


def create_dataloader(arrays: Tuple, batch_size: int, *, key: jax.random.PRNGKey):
    """Create a simple dataloader for JAX arrays."""
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)

    indices = jnp.arange(dataset_size)

    def dataloader_generator():
        while True:
            perm = jr.permutation(key, indices)
            start = 0
            end = batch_size

            while end < dataset_size:
                batch_perm = perm[start:end]
                yield tuple(array[batch_perm] for array in arrays)
                start = end
                end = start + batch_size

    return dataloader_generator()


# %%
# Loss function and training utilities
def loss_fn(
    model: NeuralCDE, ts: jnp.ndarray, target_responses: jnp.ndarray, coeffs: Tuple
) -> Tuple[float, Dict]:
    """Mean squared error loss for system identification."""
    pred = jax.vmap(model)(ts, coeffs)

    # MSE loss
    mse = jnp.mean((pred - target_responses) ** 2)

    # Additional metrics
    rmse = jnp.sqrt(mse)

    metrics = {"loss": mse, "rmse": rmse}
    return mse, metrics


# JIT-compiled training step
@eqx.filter_jit
def make_step(
    model: NeuralCDE,
    data: Tuple,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
) -> Tuple[float, Dict, NeuralCDE, Any]:
    """Single training step."""
    ts, target_responses, *coeffs = data

    # Compute loss and gradients
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, ts, target_responses, coeffs
    )

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


# %%
# Training function
def train_neural_cde_msd(config: Config) -> Tuple[NeuralCDE, Dict]:
    """Train a Neural CDE model for mass-spring-damper system identification using msd_simulation_with_forcing data."""
    print("Starting Neural CDE training for MSD system...")
    print("Using msd_simulation_with_forcing for advanced data generation...")

    # Initialize random keys
    key = jr.PRNGKey(config.seed)
    train_data_key, test_data_key, model_key, loader_key = jr.split(key, 4)

    # Generate training data using msd_simulation_with_forcing
    print("Generating training data using msd_simulation_with_forcing...")
    ts, coeffs, forces, responses, data_size = generate_msd_data_from_full_simulation(
        config.dataset_size, config.add_noise, config, key=train_data_key
    )
    print(
        f"Data shape: ts={ts.shape}, forces={forces.shape}, responses={responses.shape}"
    )
    print(f"Data size: {data_size}")
    print("Response dimensions: position, velocity, acceleration")

    # Initialize model - update output size to handle 3D state (pos, vel, acc)
    print("Initializing Neural CDE model...")
    model = NeuralCDE(
        data_size, config.hidden_size, config.width_size, config.depth, key=model_key
    )

    # Setup optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Create dataloader
    dataloader = create_dataloader(
        (ts, responses) + coeffs, config.batch_size, key=loader_key
    )

    # Training loop
    train_losses = []
    train_rmses = []

    print("Starting training...")
    for step, batch_data in enumerate(dataloader):
        if step >= config.num_steps:
            break

        start_time = time.time()
        loss, metrics, model, opt_state = make_step(
            model, batch_data, opt_state, optimizer
        )
        step_time = time.time() - start_time

        train_losses.append(float(loss))
        train_rmses.append(float(metrics["rmse"]))

        print(
            f"Step {step:3d}: Loss={loss:.6f}, RMSE={metrics['rmse']:.6f}, Time={step_time:.3f}s"
        )

    # Evaluate on test set
    print("Evaluating on test set...")
    ts_test, coeffs_test, forces_test, responses_test, _ = (
        generate_msd_data_from_full_simulation(
            config.dataset_size, config.add_noise, config, key=test_data_key
        )
    )
    test_loss, test_metrics = loss_fn(model, ts_test, responses_test, coeffs_test)

    print(f"Test Loss: {test_loss:.6f}, Test RMSE: {test_metrics['rmse']:.6f}")

    results = {
        "train_losses": train_losses,
        "train_rmses": train_rmses,
        "test_loss": float(test_loss),
        "test_rmse": float(test_metrics["rmse"]),
        "model": model,
        "data_source": "msd_simulation_with_forcing",
    }

    return model, results


# %%
# Visualization functions
def plot_training_history(results: Dict):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(results["train_losses"], "b-", linewidth=2)
    ax1.set_title("Training Loss (MSE)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot RMSE
    ax2.plot(results["train_rmses"], "r-", linewidth=2)
    ax2.set_title("Training RMSE")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("RMSE")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()

    if config.save_plots:
        plt.savefig(
            f"msd_training_history.{config.plot_format}", dpi=300, bbox_inches="tight"
        )
        print(
            f"Training history plot saved as msd_training_history.{config.plot_format}"
        )

    plt.show()


def plot_msd_predictions(model: NeuralCDE, config: Config):
    """Plot mass-spring-damper system predictions with 3D state visualization."""
    key = jr.PRNGKey(1234)
    ts, coeffs, forces, responses, _ = generate_msd_data_from_full_simulation(
        4, False, config, key=key
    )

    # Get model predictions - fix coefficient structure for batch processing
    # coeffs is a tuple of arrays, we need to extract the first element for each coefficient
    batch_coeffs = tuple(c[0] for c in coeffs)
    predictions = jax.vmap(model)(ts[None, :].repeat(len(ts), axis=0), batch_coeffs)

    # Create interpolation for plotting
    interp = diffrax.CubicInterpolation(ts, batch_coeffs)
    sample_ts = ts
    values = jax.vmap(interp.evaluate)(sample_ts)

    # Plot results - updated for 3D state (position, velocity, acceleration)
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    for i in range(min(4, len(ts))):
        row = i // 2
        col = i % 2

        # Plot position
        axes[0, col].plot(
            ts, responses[i, :, 0], "b-", linewidth=2, label=f"True Position {i + 1}"
        )
        axes[0, col].plot(
            ts,
            predictions[i, :, 0],
            "r--",
            linewidth=2,
            label=f"Predicted Position {i + 1}",
        )
        axes[0, col].set_xlabel("Time")
        axes[0, col].set_ylabel("Position")
        axes[0, col].set_title(f"Sample {i + 1} - Position")
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)

        # Plot velocity
        axes[1, col].plot(
            ts, responses[i, :, 1], "g-", linewidth=2, label=f"True Velocity {i + 1}"
        )
        axes[1, col].plot(
            ts,
            predictions[i, :, 1],
            "m--",
            linewidth=2,
            label=f"Predicted Velocity {i + 1}",
        )
        axes[1, col].set_xlabel("Time")
        axes[1, col].set_ylabel("Velocity")
        axes[1, col].set_title(f"Sample {i + 1} - Velocity")
        axes[1, col].legend()
        axes[1, col].grid(True, alpha=0.3)

        # Plot acceleration
        axes[2, col].plot(
            ts,
            responses[i, :, 2],
            "c-",
            linewidth=2,
            label=f"True Acceleration {i + 1}",
        )
        axes[2, col].plot(
            ts,
            predictions[i, :, 2],
            "y--",
            linewidth=2,
            label=f"Predicted Acceleration {i + 1}",
        )
        axes[2, col].set_xlabel("Time")
        axes[2, col].set_ylabel("Acceleration")
        axes[2, col].set_title(f"Sample {i + 1} - Acceleration")
        axes[2, col].legend()
        axes[2, col].grid(True, alpha=0.3)

    plt.tight_layout()

    if config.save_plots:
        plt.savefig(
            f"msd_predictions_3d.{config.plot_format}", dpi=300, bbox_inches="tight"
        )
        print(
            f"MSD 3D predictions plot saved as msd_predictions_3d.{config.plot_format}"
        )

    plt.show()


def plot_phase_space_predictions(model: NeuralCDE, config: Config):
    """Plot 3D phase space predictions using msd_simulation_with_forcing data."""
    key = jr.PRNGKey(1234)
    ts, coeffs, forces, responses, _ = generate_msd_data_from_full_simulation(
        1, False, config, key=key
    )

    # Get model predictions
    predictions = jax.vmap(model)(ts, coeffs)

    # Create 3D phase space plot
    fig = plt.figure(figsize=(15, 5))

    # True phase space
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot3D(
        responses[0, :, 0], responses[0, :, 1], responses[0, :, 2], "b-", linewidth=2
    )
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Velocity")
    ax1.set_zlabel("Acceleration")
    ax1.set_title("True 3D Phase Space")

    # Predicted phase space
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.plot3D(
        predictions[0, :, 0],
        predictions[0, :, 1],
        predictions[0, :, 2],
        "r--",
        linewidth=2,
    )
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    ax2.set_zlabel("Acceleration")
    ax2.set_title("Predicted 3D Phase Space")

    # Overlay comparison
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.plot3D(
        responses[0, :, 0],
        responses[0, :, 1],
        responses[0, :, 2],
        "b-",
        linewidth=2,
        label="True",
    )
    ax3.plot3D(
        predictions[0, :, 0],
        predictions[0, :, 1],
        predictions[0, :, 2],
        "r--",
        linewidth=2,
        label="Predicted",
    )
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Velocity")
    ax3.set_zlabel("Acceleration")
    ax3.set_title("Phase Space Comparison")
    ax3.legend()

    plt.tight_layout()

    if config.save_plots:
        plt.savefig(
            f"msd_phase_space_predictions.{config.plot_format}",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            f"MSD phase space predictions plot saved as msd_phase_space_predictions.{config.plot_format}"
        )

    plt.show()


# %%
# Main execution
def main():
    """Main training and evaluation pipeline using msd_simulation_with_forcing data."""
    print("Neural CDE Mass-Spring-Damper Example with msd_simulation_with_forcing")
    print("=====================================================================")
    print(f"Configuration: {config.__dict__}")
    print("Data Source: msd_simulation_with_forcing with advanced pink noise forcing")
    print("State Dimensions: Position, Velocity, Acceleration (3D)")

    # Train model
    model, results = train_neural_cde_msd(config)

    # Plot results
    plot_training_history(results)
    plot_msd_predictions(model, config)
    plot_phase_space_predictions(model, config)  # New 3D phase space visualization

    print("\nTraining completed!")
    print(f"Final test RMSE: {results['test_rmse']:.6f}")
    print(f"Data source: {results.get('data_source', 'Unknown')}")

    return model, results


# %%
# Run the example
if __name__ == "__main__":
    print("Starting Neural CDE with msd_simulation_with_forcing integration...")
    model, results = main()
