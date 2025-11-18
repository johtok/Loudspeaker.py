#!/usr/bin/env python3
"""
Neural CDE Example Script
=========================

This script demonstrates how to use diffrax for Neural Controlled Differential Equations (Neural CDEs)
for time series classification. This is a modular script that can be run section by section.

Requirements:
- diffrax
- equinox
- jax
- jax.numpy
- optax
- matplotlib
- torch (for comparison)

Usage:
    python scripts/neural_cde_example.py
    # Or run sections individually in Jupyter or IDE with #%% support
"""

# %%
# Import necessary libraries
import math
import time
from typing import Any, Dict, Tuple

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import matplotlib.pyplot as plt
import optax

print("JAX devices:", jax.devices())
print("JAX backend:", jax.default_backend())


# %%
# Configuration and hyperparameters
class Config:
    # Data generation
    dataset_size = 256
    add_noise = False
    sequence_length = 100

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
    """Neural CDE model for time series classification."""

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
        self.linear = eqx.nn.Linear(hidden_size, 1, key=lkey)

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
            prediction = jax.vmap(lambda y: jnn.sigmoid(self.linear(y))[0])(solution.ys)
        else:
            (prediction,) = jnn.sigmoid(self.linear(solution.ys[-1]))

        return prediction


# %%
# Data Generation
def get_spiral_data(
    dataset_size: int, add_noise: bool, *, key: jax.random.PRNGKey
) -> Tuple[jnp.ndarray, Tuple, jnp.ndarray, int]:
    """Generate spiral time series data for binary classification."""
    theta_key, noise_key = jr.split(key, 2)

    length = config.sequence_length
    theta = jr.uniform(theta_key, (dataset_size,), minval=0, maxval=2 * math.pi)
    y0 = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)

    # Time vector
    ts = jnp.broadcast_to(jnp.linspace(0, 4 * math.pi, length), (dataset_size, length))

    # Spiral dynamics matrix
    matrix = jnp.array([[-0.3, 2], [-2, -0.3]])

    # Generate spirals using matrix exponential
    ys = jax.vmap(
        lambda y0i, ti: jax.vmap(lambda tij: jsp.linalg.expm(tij * matrix) @ y0i)(ti)
    )(y0, ts)

    # Add time as a channel (important for CDEs)
    ys = jnp.concatenate([ts[:, :, None], ys], axis=-1)

    # Make half the spirals counter-clockwise
    ys = ys.at[: dataset_size // 2, :, 1].multiply(-1)

    # Add noise if requested
    if add_noise:
        ys = ys + jr.normal(noise_key, ys.shape) * 0.1

    # Compute interpolation coefficients
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(ts, ys)

    # Labels: first half are clockwise (1), second half counter-clockwise (0)
    labels = jnp.zeros((dataset_size,))
    labels = labels.at[: dataset_size // 2].set(1.0)

    _, _, data_size = ys.shape

    return ts, coeffs, labels, data_size


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
    model: NeuralCDE, ts: jnp.ndarray, labels: jnp.ndarray, coeffs: Tuple
) -> Tuple[float, Dict]:
    """Binary cross-entropy loss with accuracy metric."""
    pred = jax.vmap(model)(ts, coeffs)

    # Binary cross-entropy
    bxe = labels * jnp.log(pred) + (1 - labels) * jnp.log(1 - pred)
    bxe = -jnp.mean(bxe)

    # Accuracy
    acc = jnp.mean((pred > 0.5) == (labels == 1))

    metrics = {"loss": bxe, "accuracy": acc}
    return bxe, metrics


# JIT-compiled training step
@eqx.filter_jit
def make_step(
    model: NeuralCDE,
    data: Tuple,
    opt_state: Any,
    optimizer: optax.GradientTransformation,
) -> Tuple[float, Dict, NeuralCDE, Any]:
    """Single training step."""
    ts, labels, *coeffs = data

    # Compute loss and gradients
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, ts, labels, coeffs
    )

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, metrics, model, opt_state


# %%
# Training function
def train_neural_cde(config: Config) -> Tuple[NeuralCDE, Dict]:
    """Train a Neural CDE model."""
    print("Starting Neural CDE training...")

    # Initialize random keys
    key = jr.PRNGKey(config.seed)
    train_data_key, test_data_key, model_key, loader_key = jr.split(key, 4)

    # Generate training data
    print("Generating training data...")
    ts, coeffs, labels, data_size = get_spiral_data(
        config.dataset_size, config.add_noise, key=train_data_key
    )
    print(f"Data shape: ts={ts.shape}, data_size={data_size}")

    # Initialize model
    print("Initializing model...")
    model = NeuralCDE(
        data_size, config.hidden_size, config.width_size, config.depth, key=model_key
    )

    # Setup optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Create dataloader
    dataloader = create_dataloader(
        (ts, labels) + coeffs, config.batch_size, key=loader_key
    )

    # Training loop
    train_losses = []
    train_accuracies = []

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
        train_accuracies.append(float(metrics["accuracy"]))

        print(
            f"Step {step:3d}: Loss={loss:.4f}, Acc={metrics['accuracy']:.4f}, Time={step_time:.3f}s"
        )

    # Evaluate on test set
    print("Evaluating on test set...")
    ts_test, coeffs_test, labels_test, _ = get_spiral_data(
        config.dataset_size, config.add_noise, key=test_data_key
    )
    test_loss, test_metrics = loss_fn(model, ts_test, labels_test, coeffs_test)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_metrics['accuracy']:.4f}")

    results = {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "model": model,
    }

    return model, results


# %%
# Visualization functions
def plot_training_history(results: Dict):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    ax1.plot(results["train_losses"], "b-", linewidth=2)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.plot(results["train_accuracies"], "r-", linewidth=2)
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if config.save_plots:
        plt.savefig(
            f"training_history.{config.plot_format}", dpi=300, bbox_inches="tight"
        )
        print(f"Training history plot saved as training_history.{config.plot_format}")

    plt.show()


def plot_spiral_predictions(model: NeuralCDE, config: Config):
    """Plot spiral data with model predictions."""
    key = jr.PRNGKey(1234)
    ts, coeffs, labels, _ = get_spiral_data(8, False, key=key)

    # Get model predictions
    predictions = jax.vmap(model)(ts, coeffs)

    # Create interpolation for plotting
    interp = diffrax.CubicInterpolation(ts[0], tuple(c[0] for c in coeffs))
    sample_ts = ts[0]
    values = jax.vmap(interp.evaluate)(sample_ts)

    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # Plot the spiral
    ax.plot(values[:, 0], values[:, 1], values[:, 2], "b-", linewidth=2, label="Spiral")

    # Color by prediction
    colors = plt.cm.viridis(predictions[0])
    ax.scatter(
        values[:, 0],
        values[:, 1],
        values[:, 2],
        c=colors,
        cmap="viridis",
        s=50,
        alpha=0.7,
        label="Prediction",
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    ax.set_title(
        f"Spiral with Neural CDE Predictions\nTrue Label: {labels[0]:.1f}, Prediction: {predictions[0]:.3f}"
    )

    plt.legend()
    plt.tight_layout()

    if config.save_plots:
        plt.savefig(
            f"spiral_predictions.{config.plot_format}", dpi=300, bbox_inches="tight"
        )
        print(
            f"Spiral predictions plot saved as spiral_predictions.{config.plot_format}"
        )

    plt.show()


# %%
# Main execution
def main():
    """Main training and evaluation pipeline."""
    print("Neural CDE Example")
    print("==================")
    print(f"Configuration: {config.__dict__}")

    # Train model
    model, results = train_neural_cde(config)

    # Plot results
    plot_training_history(results)
    plot_spiral_predictions(model, config)

    print("\nTraining completed!")
    print(f"Final test accuracy: {results['test_accuracy']:.3f}")

    return model, results


# %%
# Run the example
if __name__ == "__main__":
    model, results = main()
