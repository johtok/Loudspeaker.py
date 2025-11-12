#!/usr/bin/env python3
"""
Neural ODE Mass-Spring-Damper Example
=====================================

This script demonstrates how to use diffrax for Neural Ordinary Differential Equations (Neural ODEs)
for modeling mass-spring-damper systems. This version uses the comprehensive neural_ode_funcs module
for modular, reusable code patterns based on the example notebooks.

Key Features:
- Uses neural_ode_funcs module for modular architecture
- Mass-spring-damper simulation with pink noise forcing from msd_simulation_with_forcing
- Proper 3D state representation (position, velocity, acceleration)
- Modular configuration system with sensible defaults
- Optional visualization with config control
- Training with comprehensive evaluation metrics
- JAX 64-bit precision for numerical accuracy

Based on patterns from:
- neural_ode_diffrax_example.ipynb
- ode_with_forcing_diffrax.ipynb
- msd_simulation_with_forcing.py

Requirements:
- diffrax
- equinox
- jax
- jax.numpy
- optax
- matplotlib
- numpy

Usage:
    python scripts/exp2_mass_spring_damper/neural_ode_example.py
    # Or run sections individually in Jupyter or IDE with #%% support
"""

#%%
"""Neural ODE Mass-Spring-Damper Example
Based on neural_ode_diffrax_example.ipynb and ode_with_forcing_diffrax.ipynb
"""

# Imports
import math
import time
from typing import Tuple, Dict, Any
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import diffrax as dx
import optax
import matplotlib.pyplot as plt
import numpy as np

# Enable 64-bit precision for numerical accuracy
jax.config.update("jax_enable_x64", True)

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print(f"64-bit precision enabled: {jax.config.jax_enable_x64}")

# Import neural_ode_funcs module
from neural_ode_funcs import *

#%%
# Configuration
config = create_neural_ode_config(
    # Override defaults for MSD application
    hidden_dim=64,
    num_layers=3,
    output_dim=3,  # position, velocity, acceleration
    dataset_size=256,
    num_steps=1000,
    visualization_enabled=False,  # Default off for experiments
    # Data parameters
    simulation_time=0.1,  # seconds
    sample_rate=1000,
    initial_condition_range=(-0.5, 0.5),
    # Training parameters
    learning_rate=1e-3,
    batch_size=32,
    # MSD-specific parameters
    msd_params={
        'mass': 0.05,  # kg
        'natural_frequency': 25.0,  # Hz
        'damping_ratio': 0.01,
        'forcing_amplitude': 1.0,
        'forcing_type': 'pink_noise'
    }
)

print(f"Configuration created:")
print(f"  Model: hidden_dim={config['model']['hidden_dim']}, num_layers={config['model']['num_layers']}")
print(f"  Data: dataset_size={config['data']['dataset_size']}, simulation_time={config['data']['simulation_time']}")
print(f"  Training: num_steps={config['training']['num_steps']}, learning_rate={config['training']['learning_rate']}")

#%%
# Data Generation
print("Generating synthetic data using neural_ode_funcs...")
key = jax.random.PRNGKey(42)
ts, train_data, test_data = generate_synthetic_data(config, key=key)

print(f"Training data shape: {train_data['trajectories'].shape}")
print(f"Test data shape: {test_data['trajectories'].shape}")
print(f"Time points: {ts.shape}")

#%%
# Model Definition
key = jax.random.PRNGKey(123)
model = NeuralODEModel(
    data_size=config['model']['output_dim'],  # position, velocity, acceleration
    hidden_dim=config['model']['hidden_dim'],
    num_layers=config['model']['num_layers'],
    solver_type=config['solver']['solver_type'],
    activation=config['model']['activation'],
    key=key
)

print(f"Model created with {sum(p.size for p in jax.tree_util.tree_leaves(model) if p.ndim > 0)} parameters")

#%%
# Training
print("Starting Neural ODE training for MSD system...")
print(f"Training configuration: {config['training']}")

# Train the model
trained_model, training_history = train_neural_ode(
    model, train_data, config, test_data
)

#%%
# Evaluation
print("\nEvaluating trained model...")
test_loss, test_metrics = evaluate_model_step(trained_model, test_data, config['solver'])
print(f"Final test loss: {test_loss:.6f}")
print(f"Final test RMSE: {test_metrics['rmse']:.6f}")
print(f"Final test relative error: {test_metrics['relative_error']:.6f}")

# Compute additional metrics
predictions = jax.vmap(lambda y0: solve_neural_ode(trained_model, ts, y0, config))(test_data['initial_conditions'])
additional_metrics = compute_metrics(predictions, test_data['trajectories'])
print(f"Additional metrics: {additional_metrics}")

#%%
# Visualization (optional)
if config['visualization']['enabled']:
    print("\nGenerating visualizations...")
    
    # Plot training history
    plot_training_history(training_history, config)
    
    # Plot trajectories
    plot_trajectories(trained_model, test_data, config, num_samples=5)
    
    # Plot phase space
    plot_phase_space(trained_model, test_data, config, num_samples=3)
    
    print(f"Visualizations saved to {config['visualization']['save_dir']}")

#%%
# Main execution summary
print("\n" + "="*60)
print("NEURAL ODE MASS-SPRING-DAMPER EXAMPLE COMPLETED")
print("="*60)
print(f"Final test loss: {test_loss:.6f}")
print(f"Final test RMSE: {test_metrics['rmse']:.6f}")
print(f"Training steps: {len(training_history['train_loss'])}")
print(f"Model parameters: {sum(p.size for p in jax.tree_util.tree_leaves(trained_model) if p.ndim > 0)}")
print(f"Data source: msd_simulation_with_forcing with advanced pink noise forcing")
print(f"State dimensions: Position, Velocity, Acceleration (3D)")
print("="*60)