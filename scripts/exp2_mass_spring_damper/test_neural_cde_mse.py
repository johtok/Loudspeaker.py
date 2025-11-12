#!/usr/bin/env python3
"""
Test Neural CDE with MSE Loss for State Trajectory Prediction
============================================================

This script demonstrates the updated Neural CDE that uses MSE loss instead of 
classification, following the pattern from neural_ode_diffrax_example.ipynb.

The model now:
- Takes forcing signals as input (time + external force)
- Predicts state trajectories (position + velocity + acceleration)
- Uses MSE loss for training
- Provides per-dimension metrics

Usage:
    pixi run python scripts/exp2_mass_spring_damper/test_neural_cde_mse.py
"""

import sys
sys.path.append('.')

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from scripts.exp2_mass_spring_damper.msd_simulation_with_forcing import (
    MSDConfig as MSDFullConfig,
    ForcingType,
    run_batch_simulation
)

def generate_msd_data_for_cde(dataset_size: int, add_noise: bool, simulation_time: float, *, key):
    """Generate MSD data for Neural CDE state trajectory prediction."""
    
    print("Generating MSD data for Neural CDE...")
    
    # Create msd_simulation_with_forcing config
    msd_config = MSDFullConfig(
        mass=0.05,
        natural_frequency=25.0,
        damping_ratio=0.01,
        sample_rate=1000,
        simulation_time=simulation_time,
        forcing_type=ForcingType.PINK_NOISE,
        forcing_amplitude=1.0,
        batch_size=dataset_size,
        normalize_plots=False,
        save_plots=False
    )
    
    # Generate batch simulation data
    batch_results = run_batch_simulation(msd_config)
    
    # Extract data from batch results
    ts = batch_results['time']
    forces = batch_results['forcings']
    positions = batch_results['positions']
    velocities = batch_results['velocities']
    
    # Add acceleration (computed from velocity)
    accelerations = []
    for i in range(dataset_size):
        acc = jnp.gradient(velocities[i], ts[1] - ts[0])
        accelerations.append(acc)
    accelerations = jnp.stack(accelerations)
    
    # Stack state responses: [position, velocity, acceleration]
    state_responses = jnp.stack([positions, velocities, accelerations], axis=-1)
    
    # Create control path data: [time, force] - this will be the input to the Neural CDE
    control_data = jnp.concatenate([
        ts[None, :, None].repeat(dataset_size, axis=0),  # time
        forces[:, :, None],                              # external forcing
    ], axis=-1)
    
    # Add noise if requested
    if add_noise:
        noise_key, key = jr.split(key)
        control_data = control_data + jr.normal(noise_key, control_data.shape) * 0.01
        state_responses = state_responses + jr.normal(noise_key, state_responses.shape) * 0.01
    
    # Compute interpolation coefficients for the control path
    coeffs = jax.vmap(diffrax.backward_hermite_coefficients)(control_data[..., 0], control_data)
    
    _, _, control_size = control_data.shape
    
    print(f"Generated data:")
    print(f"  - Time points: {len(ts)}")
    print(f"  - Batch size: {dataset_size}")
    print(f"  - Control input shape: {control_data.shape} (time + force)")
    print(f"  - State responses shape: {state_responses.shape} (pos + vel + acc)")
    print(f"  - Control size: {control_size}")
    print(f"  - State size: {state_responses.shape[-1]}")
    
    return ts, coeffs, state_responses, control_data, control_size

class NeuralCDE(eqx.Module):
    """Neural CDE for state trajectory prediction."""
    
    initial: eqx.nn.MLP
    func: eqx.nn.MLP
    linear: eqx.nn.Linear
    
    def __init__(self, control_size: int, hidden_size: int, width_size: int, depth: int, *, key):
        ikey, fkey, lkey = jr.split(key, 3)
        
        self.initial = eqx.nn.MLP(control_size, hidden_size, width_size, depth, key=ikey)
        self.func = eqx.nn.MLP(
            hidden_size + control_size, hidden_size, width_size, depth,
            activation=jax.nn.softplus, final_activation=jax.nn.tanh, key=fkey
        )
        self.linear = eqx.nn.Linear(hidden_size, 3, key=lkey)  # Output position, velocity, acceleration
    
    def __call__(self, ts: jnp.ndarray, coeffs: tuple, *, key=None):
        # Create control path from coefficients - fix coefficient structure
        # coeffs is a tuple of coefficient arrays, each with shape (times-1, ...)
        # We need to ensure the coefficients have the right structure
        
        # Extract individual sample coefficients
        if len(coeffs[0].shape) == 3:  # Batch coefficients (batch_size, times-1, ...)
            # Take first sample for interpolation
            sample_coeffs = tuple(c[0] for c in coeffs)
        else:
            sample_coeffs = coeffs
            
        control = diffrax.CubicInterpolation(ts, sample_coeffs)
        
        def vector_field(t, y, args):
            # Get control input at time t
            control_input = control.evaluate(t)
            # Concatenate hidden state with control input
            combined = jnp.concatenate([y, control_input])
            return self.func(combined)
        
        # Initial condition
        y0 = self.initial(control.evaluate(ts[0]))
        
        # Solve the ODE
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(vector_field),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        )
        
        # Map to output space
        return jax.vmap(self.linear)(solution.ys)

def loss_fn(model, ts, state_targets, coeffs):
    """Mean squared error loss for state trajectory prediction."""
    
    # Process each sample individually to handle coefficient structure
    predictions_list = []
    for i in range(state_targets.shape[0]):
        # Extract coefficients for single sample
        sample_coeffs = tuple(coeffs[j][i] for j in range(len(coeffs)))
        pred = model(ts, sample_coeffs)
        predictions_list.append(pred)
    
    predictions = jnp.stack(predictions_list)
    
    # MSE loss between predicted and true state trajectories
    mse = jnp.mean((state_targets - predictions) ** 2)
    
    # Per-dimension metrics
    pos_mse = jnp.mean((state_targets[:, :, 0] - predictions[:, :, 0]) ** 2)
    vel_mse = jnp.mean((state_targets[:, :, 1] - predictions[:, :, 1]) ** 2)
    acc_mse = jnp.mean((state_targets[:, :, 2] - predictions[:, :, 2]) ** 2)
    
    rmse = jnp.sqrt(mse)
    
    metrics = {
        'loss': mse,
        'rmse': rmse,
        'position_mse': pos_mse,
        'velocity_mse': vel_mse,
        'acceleration_mse': acc_mse
    }
    return mse, metrics

@eqx.filter_jit
def make_step(model, ts, state_targets, coeffs, opt_state, optimizer):
    """Single training step."""
    (loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, ts, state_targets, coeffs
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, metrics, model, opt_state

def train_neural_cde_msd():
    """Train Neural CDE for MSD state trajectory prediction."""
    
    print("Neural CDE State Trajectory Prediction Training")
    print("=" * 50)
    
    # Configuration
    dataset_size = 64
    simulation_time = 0.08
    hidden_size = 16
    width_size = 32
    depth = 2
    learning_rate = 1e-3
    num_steps = 50
    
    # Initialize random keys
    key = jr.PRNGKey(1234)
    data_key, model_key, train_key = jr.split(key, 3)
    
    # Generate training data
    ts, coeffs, state_responses, control_data, control_size = generate_msd_data_for_cde(
        dataset_size, False, simulation_time, key=data_key
    )
    
    # Initialize model
    print(f"\\nInitializing Neural CDE model...")
    model = NeuralCDE(
        control_size=control_size,
        hidden_size=hidden_size,
        width_size=width_size,
        depth=depth,
        key=model_key
    )
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    print(f"Model initialized with:")
    print(f"  - Control input size: {control_size}")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Training steps: {num_steps}")
    
    # Training loop
    train_losses = []
    train_metrics = []
    
    print(f"\\nStarting training...")
    for step in range(num_steps):
        start_time = time.time()
        loss, metrics, model, opt_state = make_step(
            model, ts, state_responses, coeffs, opt_state, optimizer
        )
        step_time = time.time() - start_time
        
        train_losses.append(float(loss))
        train_metrics.append(metrics)
        
        if step % 10 == 0 or step == num_steps - 1:
            print(f"Step {step:3d}: Loss={loss:.6f}, RMSE={metrics['rmse']:.6f}, "
                  f"Pos_MSE={metrics['position_mse']:.6f}, Vel_MSE={metrics['velocity_mse']:.6f}, "
                  f"Acc_MSE={metrics['acceleration_mse']:.6f}, Time={step_time:.3f}s")
    
    # Final evaluation
    final_loss, final_metrics = loss_fn(model, ts, state_responses, coeffs)
    
    print(f"\\nTraining completed!")
    print(f"Final training loss: {final_loss:.6f}")
    print(f"Final training RMSE: {final_metrics['rmse']:.6f}")
    print(f"Final position MSE: {final_metrics['position_mse']:.6f}")
    print(f"Final velocity MSE: {final_metrics['velocity_mse']:.6f}")
    print(f"Final acceleration MSE: {final_metrics['acceleration_mse']:.6f}")
    
    return model, {
        'train_losses': train_losses,
        'train_metrics': train_metrics,
        'final_loss': float(final_loss),
        'final_metrics': final_metrics,
        'model': model,
        'config': {
            'dataset_size': dataset_size,
            'simulation_time': simulation_time,
            'hidden_size': hidden_size,
            'width_size': width_size,
            'depth': depth,
            'learning_rate': learning_rate,
            'num_steps': num_steps
        }
    }

def plot_results(model, results):
    """Plot training results and predictions."""
    config = results['config']
    
    # Generate test data for visualization
    key = jr.PRNGKey(5678)
    ts, coeffs, state_responses, control_data, _ = generate_msd_data_for_cde(
        3, False, config['simulation_time'], key=key
    )
    
    # Get predictions - process each sample individually
    predictions_list = []
    for i in range(3):  # 3 samples for plotting
        sample_coeffs = tuple(coeffs[j][i] for j in range(len(coeffs)))
        pred = model(ts, sample_coeffs)
        predictions_list.append(pred)
    predictions = jnp.stack(predictions_list)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"State responses shape: {state_responses.shape}")
    
    # Plot training history
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training loss
    axes[0, 0].plot(results['train_losses'], 'b-', linewidth=2)
    axes[0, 0].set_title('Training Loss (MSE)')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Training RMSE
    train_rmses = [m['rmse'] for m in results['train_metrics']]
    axes[0, 1].plot(train_rmses, 'r-', linewidth=2)
    axes[0, 1].set_title('Training RMSE')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Per-dimension MSE
    pos_mses = [m['position_mse'] for m in results['train_metrics']]
    vel_mses = [m['velocity_mse'] for m in results['train_metrics']]
    acc_mses = [m['acceleration_mse'] for m in results['train_metrics']]
    
    axes[0, 2].plot(pos_mses, 'b-', linewidth=2, label='Position')
    axes[0, 2].plot(vel_mses, 'g-', linewidth=2, label='Velocity')
    axes[0, 2].plot(acc_mses, 'r-', linewidth=2, label='Acceleration')
    axes[0, 2].set_title('Per-Dimension MSE')
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('MSE')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_yscale('log')
    
    # Sample predictions
    for i in range(min(3, len(predictions))):
        row = 1
        col = i
        
        # Position
        axes[row, col].plot(ts, state_responses[i, :, 0], 'b-', linewidth=2, label='True Position')
        axes[row, col].plot(ts, predictions[i, :, 0], 'r--', linewidth=2, label='Predicted Position')
        axes[row, col].plot(ts, state_responses[i, :, 1], 'g-', linewidth=2, label='True Velocity')
        axes[row, col].plot(ts, predictions[i, :, 1], 'm--', linewidth=2, label='Predicted Velocity')
        axes[row, col].plot(ts, state_responses[i, :, 2], 'c-', linewidth=2, label='True Acceleration')
        axes[row, col].plot(ts, predictions[i, :, 2], 'y--', linewidth=2, label='Predicted Acceleration')
        
        axes[row, col].set_xlabel('Time')
        axes[row, col].set_ylabel('State Values')
        axes[row, col].set_title(f'Sample {i+1} - State Trajectories')
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_cde_msd_mse_training.png', dpi=300, bbox_inches='tight')
    print(f"\\nTraining plots saved as neural_cde_msd_mse_training.png")
    plt.show()

def main():
    """Main training and evaluation."""
    print("Neural CDE with MSE Loss - State Trajectory Prediction")
    print("=" * 60)
    print("This example demonstrates:")
    print("✓ Neural CDE trained with MSE loss instead of classification")
    print("✓ State trajectory prediction from forcing input")
    print("✓ Per-dimension metrics tracking")
    print("✓ Advanced msd_simulation_with_forcing data generation")
    print()
    
    # Train model
    model, results = train_neural_cde_msd()
    
    # Plot results
    plot_results(model, results)
    
    print("\\n" + "="*60)
    print("🎉 Neural CDE MSE training completed successfully!")
    print("The model can now predict mass-spring-damper state trajectories")
    print("from external forcing signals using MSE loss optimization.")

if __name__ == "__main__":
    import time
    main()