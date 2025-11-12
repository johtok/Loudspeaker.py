#!/usr/bin/env python3
"""
Neural ODE Functions Module
===========================

This module provides a comprehensive, modular collection of functions for neural ODE experiments,
based on patterns from the diffrax neural ODE examples and mass-spring-damper simulations.

Key Features:
- Configuration management with sensible defaults
- Neural network architectures using equinox.Module
- Data generation for synthetic ODE systems
- Training functions with gradient computation
- ODE solving with multiple solver options
- Evaluation and metrics computation
- Visualization functions with config control
- Support for both 32-bit and 64-bit precision

Architecture:
- Follows JAX/diffrax/equinox patterns
- Modular function-based design
- Comprehensive error handling
- Optional visualization controlled by config
- Reusable and composable functions

Usage:
    from scripts.exp2_mass_spring_damper.neural_ode_funcs import *
    
    # Create configuration
    config = create_neural_ode_config()
    
    # Generate data
    ts, train_data, test_data = generate_synthetic_data(config)
    
    # Create model
    model = NeuralODEModel(config)
    
    # Train model
    trained_model, history = train_neural_ode(model, train_data, config)
    
    # Evaluate
    metrics = evaluate_model(trained_model, test_data, config)
    
    # Visualize results
    plot_training_history(history, config)
"""

import math
import time
import warnings
import os
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
import matplotlib.pyplot as plt
import numpy as np
import optax
import signal
import time
from contextlib import contextmanager

print(f"JAX devices: {jax.devices()}")
print(f"JAX backend: {jax.default_backend()}")
print(f"64-bit precision enabled: {jax.config.jax_enable_x64}")


#%%
# Timeout and utility functions
@contextmanager
def timeout_context(seconds: int = 600):
    """Context manager for timeout functionality."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    # Set the signal handler and a alarm
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def add_timeout_to_function(func, timeout_seconds: int = 600):
    """Decorator to add timeout to a function."""
    def wrapper(*args, **kwargs):
        with timeout_context(timeout_seconds):
            return func(*args, **kwargs)
    return wrapper


#%%
# Configuration Management
@dataclass
class NeuralODEConfig:
    """Comprehensive configuration for neural ODE experiments."""
    
    # Model parameters
    hidden_dim: int = 64
    num_layers: int = 3
    output_dim: int = 3  # position, velocity, acceleration
    activation: str = 'softplus'  # 'softplus', 'tanh', 'relu'
    
    # Training parameters
    learning_rate: float = 1e-3
    num_steps: int = 1000
    batch_size: int = 32
    weight_decay: float = 1e-4
    optimizer: str = 'adam'  # 'adam', 'adabelief', 'sgd'
    
    # Solver parameters
    dt: float = 1e-3
    solver_type: str = 'tsit5'  # 'tsit5', 'kvaerno5', 'dopri5'
    rtol: float = 1e-3
    atol: float = 1e-6
    adaptive_steps: bool = True
    
    # Data parameters
    dataset_size: int = 256
    test_split: float = 0.2
    noise_level: float = 0.01
    simulation_time: float = 1.0
    sample_rate: int = 100
    initial_condition_range: Tuple[float, float] = (-1.0, 1.0)
    
    # Forcing parameters
    forcing_enabled: bool = True
    forcing_type: str = 'pink_noise'  # 'pink_noise', 'sine', 'chirp', 'step'
    forcing_amplitude: float = 1.0
    forcing_frequency_range: Tuple[float, float] = (0.1, 10.0)
    forcing_exponent: float = 1.0  # for pink noise
    
    # Visualization parameters
    visualization_enabled: bool = False  # Default off for experiments
    save_dir: str = 'exp/'
    plot_format: str = 'png'
    dpi: int = 300
    
    # Evaluation parameters
    eval_frequency: int = 100
    early_stopping: bool = True
    patience: int = 50
    
    # Precision and numerical parameters
    use_64bit: bool = True
    gradient_clipping: float = 1.0


def create_neural_ode_config(**kwargs) -> Dict[str, Any]:
    """
    Create a comprehensive neural ODE configuration with sensible defaults.
    
    Args:
        **kwargs: Override default configuration values
        
    Returns:
        Dictionary with neural ODE configuration
    """
    # Extract MSD-specific parameters before creating NeuralODEConfig
    msd_params = kwargs.pop('msd_params', {})
    
    # Create config with remaining parameters
    config = NeuralODEConfig(**kwargs)
    
    # Add computed properties
    config_dict = {
        'model': {
            'hidden_dim': config.hidden_dim,
            'num_layers': config.num_layers,
            'output_dim': config.output_dim,
            'activation': config.activation,
        },
        'training': {
            'learning_rate': config.learning_rate,
            'num_steps': config.num_steps,
            'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'optimizer': config.optimizer,
        },
        'solver': {
            'dt': config.dt,
            'solver_type': config.solver_type,
            'rtol': config.rtol,
            'atol': config.atol,
            'adaptive_steps': config.adaptive_steps,
        },
        'data': {
            'dataset_size': config.dataset_size,
            'test_split': config.test_split,
            'noise_level': config.noise_level,
            'simulation_time': config.simulation_time,
            'sample_rate': config.sample_rate,
            'initial_condition_range': config.initial_condition_range,
        },
        'forcing': {
            'enabled': config.forcing_enabled,
            'type': config.forcing_type,
            'amplitude': config.forcing_amplitude,
            'frequency_range': config.forcing_frequency_range,
            'exponent': config.forcing_exponent,
        },
        'visualization': {
            'enabled': config.visualization_enabled,
            'save_dir': config.save_dir,
            'format': config.plot_format,
            'dpi': config.dpi,
            'bbox_inches': 'tight'
        },
        'evaluation': {
            'eval_frequency': config.eval_frequency,
            'early_stopping': config.early_stopping,
            'patience': config.patience,
        },
        'numerical': {
            'use_64bit': config.use_64bit,
            'gradient_clipping': config.gradient_clipping,
        }
    }
    
    # Add MSD parameters if provided
    if msd_params:
        config_dict['msd_params'] = msd_params
    
    return config_dict


#%%
# Neural Network Architecture
class NeuralODEFunc(eqx.Module):
    """Vector field function for neural ODE using dense layer architecture."""
    
    # 6 parameters in 2x3 matrix as specified
    weight_matrix: jnp.ndarray
    activation_fn: Callable
    
    def __init__(self, data_size: int, hidden_dim: int, num_layers: int,
                 activation: str = 'softplus', *, key: jax.random.PRNGKey, **kwargs):
        super().__init__(**kwargs)
        
        # Set activation function
        if activation == 'softplus':
            self.activation_fn = jnn.softplus
        elif activation == 'tanh':
            self.activation_fn = jnp.tanh
        elif activation == 'relu':
            self.activation_fn = jnn.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Create 2x3 weight matrix (6 parameters) without bias as specified
        # This is a simple dense layer: output = W @ input
        keys = jr.split(key, 1)
        self.weight_matrix = jr.normal(keys[0], (data_size, data_size)) * 0.1
        
    def __call__(self, t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        """
        Compute vector field f(t, y, args) using 2x3 weight matrix.
        
        Args:
            t: Time
            y: State vector (3D: position, velocity, acceleration)
            args: Additional arguments (e.g., forcing)
            
        Returns:
            Vector field evaluation
        """
        # Apply weight matrix transformation: y' = W @ y
        # This creates a linear transformation with exactly 6 parameters for 3D state
        return jnp.dot(self.weight_matrix, y)


class NeuralODEModel(eqx.Module):
    """Complete neural ODE model with initial condition mapping."""
    
    func: NeuralODEFunc
    initial_mapping: eqx.nn.MLP
    solver: Any
    
    def __init__(self, data_size: int, hidden_dim: int, num_layers: int, 
                 solver_type: str = 'tsit5', activation: str = 'softplus', 
                 *, key: jax.random.PRNGKey, **kwargs):
        super().__init__(**kwargs)
        
        # Create vector field function
        fkey, ikey = jr.split(key, 2)
        self.func = NeuralODEFunc(
            data_size, hidden_dim, num_layers, activation, key=fkey
        )
        
        # Initial condition mapping
        self.initial_mapping = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=hidden_dim // 2,
            depth=2,
            activation=self.func.activation_fn,
            final_activation=lambda x: x,  # Linear output
            key=ikey,
        )
        
        # Set up solver (stored for configuration, not state)
        if solver_type == 'tsit5':
            self.solver = diffrax.Tsit5()
        elif solver_type == 'kvaerno5':
            self.solver = diffrax.Kvaerno5()
        elif solver_type == 'dopri5':
            self.solver = diffrax.Dopri5()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
    
    def __call__(self, ts: jnp.ndarray, y0: jnp.ndarray,
                 solver_config: Dict[str, Any] = None) -> jnp.ndarray:
        """
        Solve neural ODE from initial condition.
        
        Args:
            ts: Time points for solution
            y0: Initial condition
            solver_config: Optional solver configuration override
            
        Returns:
            Solution trajectory
        """
        # Map initial condition if needed
        mapped_y0 = self.initial_mapping(y0)
        
        # Use provided solver config or defaults
        if solver_config is None:
            solver_config = {
                'solver': self.solver,
                'rtol': 1e-3,
                'atol': 1e-6,
                'adaptive_steps': True
            }
        
        # Extract solver parameters
        solver = solver_config.get('solver', self.solver)
        rtol = solver_config.get('rtol', 1e-3)
        atol = solver_config.get('atol', 1e-6)
        adaptive_steps = solver_config.get('adaptive_steps', True)
        
        try:
            # Set up ODE solve
            term = diffrax.ODETerm(self.func)
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol) if adaptive_steps else diffrax.ConstantStepSize()
            
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0] if len(ts) > 1 else None,
                y0=mapped_y0,
                stepsize_controller=stepsize_controller,
                saveat=diffrax.SaveAt(ts=ts),
            )
            
            return solution.ys
            
        except Exception as e:
            warnings.warn(f"ODE solve failed: {str(e)}")
            # Return zeros with same shape as expected solution
            return jnp.zeros((len(ts), len(y0)))


#%%
# Data Generation
def generate_pink_noise_bandpassed(length: int, sample_rate: float,
                                   freq_range: Tuple[float, float],
                                   key: jax.random.PRNGKey) -> jnp.ndarray:
    """
    Generate bandpassed pink noise for forcing signal.
    
    Args:
        length: Length of the signal
        sample_rate: Sampling frequency in Hz
        freq_range: Tuple of (low_freq, high_freq) in Hz
        key: JAX random key
        
    Returns:
        Bandpassed pink noise signal
    """
    # Generate white noise
    white_noise = jr.normal(key, (length,))
    
    # Apply 1/f filtering in frequency domain
    frequencies = jnp.fft.fftfreq(length, 1.0/sample_rate)
    pink_spectrum = jnp.ones(length, dtype=complex)
    
    # Avoid division by zero and apply 1/f filtering
    mask = jnp.abs(frequencies) > 1e-10
    pink_spectrum = pink_spectrum.at[mask].set(1.0 / jnp.sqrt(jnp.abs(frequencies[mask])))
    
    # Apply bandpass filter
    f_low, f_high = freq_range
    bandpass_mask = (jnp.abs(frequencies) >= f_low) & (jnp.abs(frequencies) <= f_high)
    pink_spectrum = pink_spectrum * bandpass_mask
    
    # Transform back to time domain
    pink_noise = jnp.fft.ifft(pink_spectrum * jnp.fft.fft(white_noise)).real
    
    # Normalize and scale
    pink_noise = pink_noise / jnp.std(pink_noise)
    
    return pink_noise

def generate_synthetic_data(config: Dict[str, Any], *,
                            key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Generate synthetic ODE data for testing neural ODE models.
    For MSD applications, this will use msd_simulation_with_forcing when available.
    
    Args:
        config: Neural ODE configuration
        key: JAX random key
        
    Returns:
        Tuple of (time_points, train_data, test_data)
    """
    # Check if we should use MSD simulation for more realistic data
    if 'msd_params' in config:
        return _generate_msd_data_with_forcing_exp2(config, key)
    
    # Fallback to generic synthetic data generation
    return _generate_generic_synthetic_data(config, key)

def _generate_generic_synthetic_data(config: Dict[str, Any], key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, Any], Dict[str, Any]]:
    """Generate generic synthetic ODE data."""
    data_config = config['data']
    
    # Create time vector
    num_points = int(data_config['simulation_time'] * data_config['sample_rate'])
    ts = jnp.linspace(0, data_config['simulation_time'], num_points)
    
    # Generate initial conditions
    ic_key, data_key = jr.split(key, 2)
    ic_range = data_config['initial_condition_range']
    initial_conditions = jr.uniform(
        ic_key,
        (data_config['dataset_size'], config['model']['output_dim']),
        minval=ic_range[0],
        maxval=ic_range[1]
    )
    
    # Generate forcing signals if enabled
    if config['forcing']['enabled']:
        forcing_key, data_key = jr.split(data_key, 2)
        forcing_signals = _generate_forcing_signals(
            config['forcing'], num_points, data_config['dataset_size'], forcing_key
        )
    else:
        forcing_signals = None
    
    # Generate trajectories using a known ODE system
    trajectories = _generate_trajectories(
        ts, initial_conditions, config, forcing_signals, data_key
    )
    
    # Add noise
    if data_config['noise_level'] > 0:
        noise_key, data_key = jr.split(data_key, 2)
        noise = jr.normal(noise_key, trajectories.shape) * data_config['noise_level']
        trajectories = trajectories + noise
    
    # Split into train/test
    test_size = int(data_config['dataset_size'] * data_config['test_split'])
    train_size = data_config['dataset_size'] - test_size
    
    train_data = {
        'ts': ts,
        'initial_conditions': initial_conditions[:train_size],
        'trajectories': trajectories[:train_size],
        'forcing': forcing_signals[:train_size] if forcing_signals is not None else None
    }
    
    test_data = {
        'ts': ts,
        'initial_conditions': initial_conditions[train_size:],
        'trajectories': trajectories[train_size:],
        'forcing': forcing_signals[train_size:] if forcing_signals is not None else None
    }
    
    return ts, train_data, test_data

def _generate_msd_data_with_forcing_exp2(config: Dict[str, Any], key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Generate MSD data for Exp2 with specific parameters:
    - Pink noise bandwidth 1Hz-100Hz
    - Timeseries 5 samples (or 50 for Exp3)
    - Sampling frequency 300Hz
    - Pink noise bandpassed to 1-50Hz
    - Mass spring damper peak tuned to 25Hz
    """
    print("Generating data for Exp2 with specific MSD parameters...")
    
    # Extract data parameters
    data_config = config['data']
    sample_rate = 300  # Fixed sampling frequency
    simulation_time = data_config.get('simulation_time', 0.1)  # seconds
    num_points = int(simulation_time * sample_rate)
    dataset_size = data_config['dataset_size']  # 5 for Exp2, 50 for Exp3
    
    # Create time vector
    ts = jnp.linspace(0, simulation_time, num_points)
    
    # MSD physical parameters (tuned for 25Hz peak)
    mass = 0.05  # kg
    natural_frequency = 25.0  # Hz (peak at 25Hz)
    damping_ratio = 0.01
    stiffness = mass * (2 * jnp.pi * natural_frequency)**2
    damping_coefficient = 2 * damping_ratio * mass * (2 * jnp.pi * natural_frequency)
    
    # Generate forcing signals (pink noise bandpassed to 1-50Hz)
    forcing_key, data_key = jr.split(key, 2)
    forcing_signals = []
    
    for i in range(dataset_size):
        signal_key = jr.fold_in(forcing_key, i)
        # Generate pink noise with bandwidth 1Hz-100Hz, then bandpass to 1-50Hz
        pink_noise = generate_pink_noise_bandpassed(
            num_points, sample_rate, (1.0, 50.0), signal_key
        )
        forcing_signals.append(pink_noise)
    
    forcing_signals = jnp.stack(forcing_signals)
    
    # Simulate MSD system response for each forcing signal
    trajectories = []
    
    for i in range(dataset_size):
        forcing_signal = forcing_signals[i]
        
        # Create interpolation for forcing signal
        coeffs = diffrax.backward_hermite_coefficients(ts, forcing_signal)
        forcing_interp = diffrax.CubicInterpolation(ts, coeffs)
        
        # Initial conditions (small random values)
        ic_key = jr.fold_in(data_key, i)
        y0 = jr.uniform(ic_key, (3,), minval=-0.1, maxval=0.1)
        
        # Define MSD ODE function
        def msd_ode(t, y, args):
            pos, vel, acc = y[0], y[1], y[2]
            forcing = forcing_interp.evaluate(t)
            # MSD equation: m*a + c*v + k*x = F(t)
            new_acc = (forcing - damping_coefficient * vel - stiffness * pos) / mass
            return jnp.array([vel, new_acc, 0.0])  # acceleration derivative is 0 for simplicity
        
        # Solve ODE
        try:
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(msd_ode),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0] * 0.1,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            )
            trajectories.append(solution.ys)
        except Exception as e:
            print(f"Warning: ODE solve failed for sample {i}: {e}")
            # Fallback to zeros
            trajectories.append(jnp.zeros((len(ts), 3)))
    
    trajectories = jnp.stack(trajectories)
    
    # Generate initial conditions for neural ODE training
    ic_key, data_key = jr.split(data_key, 2)
    ic_range = data_config['initial_condition_range']
    initial_conditions = jr.uniform(
        ic_key,
        (dataset_size, config['model']['output_dim']),
        minval=ic_range[0],
        maxval=ic_range[1]
    )
    
    # Split into train/test
    test_size = int(dataset_size * data_config['test_split'])
    train_size = dataset_size - test_size
    
    train_data = {
        'ts': ts,
        'initial_conditions': initial_conditions[:train_size],
        'trajectories': trajectories[:train_size],
        'forcing': forcing_signals[:train_size],
    }
    
    test_data = {
        'ts': ts,
        'initial_conditions': initial_conditions[train_size:],
        'trajectories': trajectories[train_size:],
        'forcing': forcing_signals[train_size:],
    }
    
    print(f"Exp2 data generated: ts={ts.shape}, trajectories={trajectories.shape}")
    print(f"Sample rate: {sample_rate}Hz, Simulation time: {simulation_time}s")
    print(f"Dataset size: {dataset_size}, Train: {train_size}, Test: {test_size}")
    print(f"MSD parameters: mass={mass}kg, natural_freq={natural_frequency}Hz")
    
    return ts, train_data, test_data


def _generate_forcing_signals(forcing_config: Dict[str, Any], num_points: int, 
                             batch_size: int, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate forcing signals based on configuration."""
    signals = []
    
    for i in range(batch_size):
        signal_key = jr.fold_in(key, i)
        
        if forcing_config['type'] == 'pink_noise':
            # Generate pink noise
            white_noise = jr.normal(signal_key, (num_points,))
            frequencies = jnp.fft.fftfreq(num_points)
            pink_spectrum = jnp.ones(num_points, dtype=complex)
            mask = jnp.abs(frequencies) > 1e-10
            pink_spectrum = pink_spectrum.at[mask].set(
                1.0 / (jnp.abs(frequencies[mask]) ** (forcing_config['exponent'] / 2))
            )
            pink_noise = jnp.fft.ifft(pink_spectrum * jnp.fft.fft(white_noise)).real
            signal = pink_noise / jnp.std(pink_noise) * forcing_config['amplitude']
            
        elif forcing_config['type'] == 'sine':
            t = jnp.linspace(0, 1.0, num_points)
            f_center = (forcing_config['frequency_range'][0] + forcing_config['frequency_range'][1]) / 2
            signal = forcing_config['amplitude'] * jnp.sin(2 * math.pi * f_center * t)
            
        elif forcing_config['type'] == 'chirp':
            t = jnp.linspace(0, 1.0, num_points)
            f0, f1 = forcing_config['frequency_range']
            signal = forcing_config['amplitude'] * jnp.sin(2 * math.pi * f0 * t + 
                        math.pi * (f1 - f0) * t**2)
        elif forcing_config['type'] == 'step':
            signal = jnp.ones(num_points) * forcing_config['amplitude']
        else:
            signal = jnp.zeros(num_points)
        
        signals.append(signal)
    
    return jnp.stack(signals)


def _generate_trajectories(ts: jnp.ndarray, initial_conditions: jnp.ndarray,
                          config: Dict[str, Any], forcing_signals: jnp.ndarray,
                          key: jax.random.PRNGKey) -> jnp.ndarray:
    """Generate trajectories using a known ODE system."""
    
    def true_dynamics(t: float, y: jnp.ndarray, forcing_val: float, output_dim: int) -> jnp.ndarray:
        """Example: Damped harmonic oscillator with forcing."""
        if output_dim >= 3:
            pos, vel, acc = y[0], y[1], y[2]
            
            # Physical parameters
            mass = 1.0
            stiffness = 25.0  # Natural frequency ~2.5 Hz
            damping = 0.1
            
            # Forcing contribution
            forcing_contribution = forcing_val / mass
            
            # Equations of motion
            new_vel = vel
            new_acc = (forcing_contribution - damping * vel - stiffness * pos) / mass
            new_jerk = 0.0  # For simplicity
            
            derivatives = jnp.array([new_vel, new_acc, new_jerk])
        else:
            # For lower-dimensional systems
            pos = y[0]
            vel = y[1] if len(y) > 1 else 0.0
            
            # Simple harmonic motion
            new_vel = vel
            new_acc = -25.0 * pos - 0.1 * vel + forcing_val
            
            derivatives = jnp.array([new_vel, new_acc])
        
        # Pad with zeros if needed for higher dimensions
        if len(derivatives) < output_dim:
            derivatives = jnp.pad(derivatives, (0, output_dim - len(derivatives)))
        
        return derivatives
    
    trajectories = []
    
    for i, (y0, forcing_signal) in enumerate(zip(initial_conditions, forcing_signals if forcing_signals is not None else [None]*len(initial_conditions))):
        traj_key = jr.fold_in(key, i)
        
        # Create interpolation for forcing signal
        if forcing_signal is not None:
            coeffs = diffrax.backward_hermite_coefficients(ts, forcing_signal)
            forcing_interp = diffrax.CubicInterpolation(ts, coeffs)
        else:
            forcing_interp = None
        
        # Define ODE function
        def ode_func(t, y, args):
            forcing_val = forcing_interp.evaluate(t) if forcing_interp is not None else 0.0
            return true_dynamics(t, y, forcing_val, config['model']['output_dim'])
        
        # Solve ODE
        try:
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(ode_func),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=(ts[1] - ts[0]) * 0.1,
                y0=y0,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-8),
            )
            trajectories.append(solution.ys)
        except Exception as e:
            warnings.warn(f"Trajectory generation failed for sample {i}: {str(e)}")
            trajectories.append(jnp.zeros((len(ts), len(y0))))
    
    return jnp.stack(trajectories)


def prepare_neural_ode_data(data: Dict[str, Any], config: Dict[str, Any]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Prepare data in format suitable for neural ODE training.
    
    Args:
        data: Raw data dictionary
        config: Neural ODE configuration
        
    Returns:
        Tuple of (input_data, target_data)
    """
    # Extract components
    ts = data['ts']
    initial_conditions = data['initial_conditions']
    trajectories = data['trajectories']
    forcing = data['forcing']
    
    # Create input data
    if forcing is not None:
        # Include forcing in input
        batch_size, seq_len, _ = trajectories.shape
        forcing_expanded = forcing[:, :, None]  # Add feature dimension
        time_expanded = ts[None, :, None].repeat(batch_size, axis=0)
        input_data = jnp.concatenate([time_expanded, forcing_expanded, trajectories], axis=-1)
    else:
        input_data = trajectories
    
    # Target is the trajectory to predict
    target_data = trajectories
    
    return input_data, target_data


#%%
# Training Functions
def loss_fn_neural_ode(model: NeuralODEModel, ts: jnp.ndarray, 
                       y0_batch: jnp.ndarray, target_batch: jnp.ndarray,
                       solver_config: Dict[str, Any] = None) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Mean squared error loss function for neural ODE training.
    
    Args:
        model: Neural ODE model
        ts: Time points
        y0_batch: Batch of initial conditions
        target_batch: Batch of target trajectories
        solver_config: Solver configuration
        
    Returns:
        Tuple of (loss_value, metrics_dict)
    """
    # Predict trajectories
    predictions = jax.vmap(lambda y0: model(ts, y0, solver_config))(y0_batch)
    
    # Compute MSE loss
    mse_loss = jnp.mean((target_batch - predictions) ** 2)
    
    # Per-dimension metrics
    dim_mse = jnp.mean((target_batch - predictions) ** 2, axis=(0, 1))
    dim_rmse = jnp.sqrt(dim_mse)
    
    # Total RMSE
    rmse = jnp.sqrt(mse_loss)
    
    # Relative error
    rel_error = jnp.mean(jnp.abs(target_batch - predictions) / (jnp.abs(target_batch) + 1e-8))
    
    metrics = {
        'loss': mse_loss,
        'rmse': rmse,
        'relative_error': rel_error,
        'dim_mse': dim_mse,
        'dim_rmse': dim_rmse,
    }
    
    return mse_loss, metrics


def loss_fn_neural_ode_with_timeout(model: NeuralODEModel, ts: jnp.ndarray,
                                   y0_batch: jnp.ndarray, target_batch: jnp.ndarray,
                                   solver_config: Dict[str, Any] = None,
                                   timeout_seconds: int = 600) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Loss function with timeout protection.
    """
    with timeout_context(timeout_seconds):
        return loss_fn_neural_ode(model, ts, y0_batch, target_batch, solver_config)


def train_step(model: NeuralODEModel, optimizer: optax.GradientTransformation,
               opt_state: Any, batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
               solver_config: Dict[str, Any] = None) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], 
                                                             NeuralODEModel, Any]:
    """
    Single training step with gradient computation and parameter update.
    
    Args:
        model: Neural ODE model
        optimizer: Optax optimizer
        opt_state: Optimizer state
        batch: Tuple of (ts, y0_batch, target_batch)
        solver_config: Solver configuration
        
    Returns:
        Tuple of (loss, metrics, updated_model, updated_opt_state)
    """
    ts, y0_batch, target_batch = batch
    
    # Compute loss and gradients
    (loss, metrics), grads = eqx.filter_value_and_grad(
        loss_fn_neural_ode, has_aux=True
    )(model, ts, y0_batch, target_batch, solver_config)
    
    # Apply gradient clipping
    grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    new_model = eqx.apply_updates(model, updates)
    
    return loss, metrics, new_model, new_opt_state


def train_neural_ode(model: NeuralODEModel, train_data: Dict[str, Any], 
                    config: Dict[str, Any], test_data: Optional[Dict[str, Any]] = None) -> Tuple[NeuralODEModel, Dict[str, Any]]:
    """
    Complete training loop for neural ODE with progress tracking.
    
    Args:
        model: Initial neural ODE model
        train_data: Training data dictionary
        config: Training configuration
        test_data: Optional test data for validation
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    train_config = config['training']
    eval_config = config['evaluation']
    solver_config = config['solver']
    
    # Setup optimizer
    if train_config['optimizer'] == 'adam':
        optimizer = optax.adam(train_config['learning_rate'])
    elif train_config['optimizer'] == 'adabelief':
        optimizer = optax.adabelief(train_config['learning_rate'])
    elif train_config['optimizer'] == 'sgd':
        optimizer = optax.sgd(train_config['learning_rate'], momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {train_config['optimizer']}")
    
    # Add weight decay if specified
    if train_config['weight_decay'] > 0:
        # Use a simple weight decay approach since add_decoupled_weight_decay might not be available
        optimizer = optax.chain(
            optimizer,
            optax.add_decayed_weights(train_config['weight_decay'])
        )
    
    # Initialize optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Prepare data
    ts = train_data['ts']
    y0_batch = train_data['initial_conditions']
    target_batch = train_data['trajectories']
    
    # Create training batches
    dataset_size = y0_batch.shape[0]
    indices = jnp.arange(dataset_size)
    
    # Training history
    history = {
        'train_loss': [],
        'train_rmse': [],
        'train_rel_error': [],
        'train_dim_mse': [],
        'test_loss': [],
        'test_rmse': [],
        'test_rel_error': [],
        'test_dim_mse': [],
        'step_times': [],
        'best_loss': float('inf'),
        'best_model': None,
        'early_stopping_counter': 0
    }
    
    print(f"Starting training for {train_config['num_steps']} steps...")
    print(f"Dataset size: {dataset_size}, Batch size: {train_config['batch_size']}")
    
    # Training loop
    for step in range(train_config['num_steps']):
        start_time = time.time()
        
        # Create mini-batch
        batch_indices = jr.choice(jr.PRNGKey(step), indices, (train_config['batch_size'],), replace=False)
        batch_y0 = y0_batch[batch_indices]
        batch_targets = target_batch[batch_indices]
        
        batch = (ts, batch_y0, batch_targets)
        
        # Training step
        loss, metrics, model, opt_state = train_step(
            model, optimizer, opt_state, batch, solver_config
        )
        
        step_time = time.time() - start_time
        history['step_times'].append(step_time)
        
        # Store metrics
        history['train_loss'].append(float(loss))
        history['train_rmse'].append(float(metrics['rmse']))
        history['train_rel_error'].append(float(metrics['relative_error']))
        history['train_dim_mse'].append(metrics['dim_mse'])
        
        # Evaluation
        if test_data is not None and step % eval_config['eval_frequency'] == 0:
            test_loss, test_metrics = evaluate_model_step(model, test_data, solver_config)
            history['test_loss'].append(float(test_loss))
            history['test_rmse'].append(float(test_metrics['rmse']))
            history['test_rel_error'].append(float(test_metrics['relative_error']))
            history['test_dim_mse'].append(test_metrics['dim_mse'])
            
            # Early stopping check
            if eval_config['early_stopping']:
                if test_loss < history['best_loss']:
                    history['best_loss'] = float(test_loss)
                    history['best_model'] = jax.tree_util.tree_map(lambda x: x.copy() if hasattr(x, 'copy') else x, model)
                    history['early_stopping_counter'] = 0
                else:
                    history['early_stopping_counter'] += 1
                    
                if history['early_stopping_counter'] >= eval_config['patience']:
                    print(f"Early stopping at step {step}")
                    if history['best_model'] is not None:
                        model = history['best_model']
                    break
        
        # Print progress
        if step % 100 == 0 or step == train_config['num_steps'] - 1:
            print(f"Step {step:4d}: Loss={loss:.6f}, RMSE={metrics['rmse']:.6f}, "
                  f"Time={step_time:.3f}s")
    
    print("Training completed!")
    return model, history


#%%
# Solving Functions
def solve_neural_ode(model: NeuralODEModel, ts: jnp.ndarray, y0: jnp.ndarray,
                    config: Dict[str, Any] = None) -> jnp.ndarray:
    """
    Solve neural ODE with given parameters.
    
    Args:
        model: Trained neural ODE model
        ts: Time points for solution
        y0: Initial condition
        config: Optional solver configuration
        
    Returns:
        Solution trajectory
    """
    if config is None:
        config = {}
    
    solver_config = config.get('solver', {
        'solver': diffrax.Tsit5(),
        'rtol': 1e-3,
        'atol': 1e-6,
        'adaptive_steps': True
    })
    
    return model(ts, y0, solver_config)


def solve_with_forcing(model: NeuralODEModel, ts: jnp.ndarray, y0: jnp.ndarray,
                      forcing_signal: jnp.ndarray, config: Dict[str, Any] = None) -> jnp.ndarray:
    """
    Solve ODE with external forcing terms.
    
    Args:
        model: Neural ODE model (should support forcing)
        ts: Time points
        y0: Initial condition
        forcing_signal: External forcing signal
        config: Solver configuration
        
    Returns:
        Solution trajectory
    """
    if config is None:
        config = {}
    
    # This is a placeholder - actual implementation would depend on how the model
    # is designed to handle forcing signals
    solver_config = config.get('solver', {
        'solver': diffrax.Tsit5(),
        'rtol': 1e-3,
        'atol': 1e-6,
        'adaptive_steps': True
    })
    
    # For now, just solve without forcing (would need model modification to support forcing)
    return model(ts, y0, solver_config)


#%%
# Evaluation Functions
def evaluate_model(model: NeuralODEModel, test_data: Dict[str, Any], 
                  config: Dict[str, Any]) -> Dict[str, float]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained neural ODE model
        test_data: Test data dictionary
        config: Configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    return evaluate_model_step(model, test_data, config.get('solver', {}))


def evaluate_model_step(model: NeuralODEModel, test_data: Dict[str, Any],
                       solver_config: Dict[str, Any]) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Single evaluation step for model assessment.
    
    Args:
        model: Neural ODE model
        test_data: Test data
        solver_config: Solver configuration
        
    Returns:
        Tuple of (loss_value, metrics_dict)
    """
    ts = test_data['ts']
    y0_batch = test_data['initial_conditions']
    target_batch = test_data['trajectories']
    
    return loss_fn_neural_ode(model, ts, y0_batch, target_batch, solver_config)


def compute_metrics(predictions: jnp.ndarray, targets: jnp.ndarray) -> Dict[str, float]:
    """
    Compute per-dimension MSE, RMSE, and other metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        
    Returns:
        Dictionary of metrics
    """
    # MSE and RMSE per dimension
    dim_mse = jnp.mean((targets - predictions) ** 2, axis=(0, 1))
    dim_rmse = jnp.sqrt(dim_mse)
    
    # Total metrics
    total_mse = jnp.mean((targets - predictions) ** 2)
    total_rmse = jnp.sqrt(total_mse)
    
    # Relative error
    rel_error = jnp.mean(jnp.abs(targets - predictions) / (jnp.abs(targets) + 1e-8))
    
    # Maximum error
    max_error = jnp.max(jnp.abs(targets - predictions))
    
    # R-squared
    ss_res = jnp.sum((targets - predictions) ** 2)
    ss_tot = jnp.sum((targets - jnp.mean(targets, axis=(0, 1), keepdims=True)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    return {
        'total_mse': float(total_mse),
        'total_rmse': float(total_rmse),
        'relative_error': float(rel_error),
        'max_error': float(max_error),
        'r2_score': float(r2),
        'dim_mse': dim_mse,
        'dim_rmse': dim_rmse,
    }


def differentiability_test(model: NeuralODEModel, ts: jnp.ndarray, 
                          y0: jnp.ndarray, config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Test gradient flow through the model.
    
    Args:
        model: Neural ODE model
        ts: Time points
        y0: Initial condition
        config: Configuration
        
    Returns:
        Dictionary of differentiability test results
    """
    solver_config = config.get('solver', {
        'solver': diffrax.Tsit5(),
        'rtol': 1e-3,
        'atol': 1e-6,
        'adaptive_steps': True
    })
    
    def test_loss_fn(params):
        test_model = eqx.apply_updates(model, {'func.out_scale': params['out_scale']})
        pred = test_model(ts, y0, solver_config)
        return jnp.mean(pred ** 2)
    
    # Test if gradients can be computed
    try:
        grad_fn = jax.grad(test_loss_fn)
        params = {'func.out_scale': model.func.out_scale}
        grads = grad_fn(params)
        
        # Check if gradients are finite
        gradients_finite = jnp.all(jnp.isfinite(grads))
        gradients_nonzero = jnp.any(jnp.abs(grads) > 1e-10)
        
        return {
            'gradients_computable': True,
            'gradients_finite': gradients_finite,
            'gradients_nonzero': gradients_nonzero,
            'test_passed': gradients_finite and gradients_nonzero
        }
        
    except Exception as e:
        return {
            'gradients_computable': False,
            'gradients_finite': False,
            'gradients_nonzero': False,
            'test_passed': False,
            'error': str(e)
        }


#%%
# Visualization Functions
def plot_training_history(history: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Plot training progress visualization if enabled in configuration.
    
    Args:
        history: Training history dictionary
        config: Configuration with visualization settings
    """
    # Check if visualization is enabled
    if not config.get('visualization', {}).get('enabled', False):
        return
    
    # Extract visualization settings with defaults
    save_dir = config.get('visualization', {}).get('save_dir', 'exp/')
    format_ = config.get('visualization', {}).get('format', 'png')
    dpi = config.get('visualization', {}).get('dpi', 300)
    
    # Create save directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create directory {save_dir}: {e}")
        return
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    if history.get('test_loss'):
        eval_steps = range(0, len(history['train_loss']), config.get('evaluation', {}).get('eval_frequency', 100))
        axes[0, 0].plot(eval_steps[:len(history['test_loss'])], history['test_loss'],
                       'r-', linewidth=2, label='Test Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training RMSE
    axes[0, 1].plot(history['train_rmse'], 'g-', linewidth=2, label='Training RMSE')
    if history.get('test_rmse'):
        eval_steps = range(0, len(history['train_rmse']), config.get('evaluation', {}).get('eval_frequency', 100))
        axes[0, 1].plot(eval_steps[:len(history['test_rmse'])], history['test_rmse'],
                       'm-', linewidth=2, label='Test RMSE')
    axes[0, 1].set_title('Training RMSE')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-dimension MSE
    if history.get('train_dim_mse'):
        dim_mse_array = jnp.array(history['train_dim_mse'])
        for dim in range(dim_mse_array.shape[1]):
            axes[1, 0].plot(dim_mse_array[:, dim], label=f'Dim {dim}', linewidth=2)
        axes[1, 0].set_title('Per-Dimension MSE')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Step times
    axes[1, 1].plot(history['step_times'], 'orange', linewidth=2)
    axes[1, 1].set_title('Step Time')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Time (s)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot (only save, don't show)
    try:
        filename = f"{save_dir}/neural_ode_training_history.{format_}"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Training history saved to {filename}")
    except Exception as e:
        print(f"Warning: Could not save training history plot: {e}")
        plt.close(fig)


def plot_trajectories(model: NeuralODEModel, test_data: Dict[str, Any],
                      config: Dict[str, Any], num_samples: int = 5) -> None:
    """
    Plot true vs predicted trajectory comparison if enabled in configuration.
    
    Args:
        model: Trained neural ODE model
        test_data: Test data dictionary
        config: Configuration
        num_samples: Number of samples to plot
    """
    # Check if visualization is enabled
    if not config.get('visualization', {}).get('enabled', False):
        return
    
    # Extract visualization settings with defaults
    save_dir = config.get('visualization', {}).get('save_dir', 'exp/')
    format_ = config.get('visualization', {}).get('format', 'png')
    dpi = config.get('visualization', {}).get('dpi', 300)
    
    # Create save directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create directory {save_dir}: {e}")
        return
    
    ts = test_data['ts']
    y0_batch = test_data['initial_conditions']
    target_batch = test_data['trajectories']
    
    # Get predictions
    predictions = jax.vmap(lambda y0: solve_neural_ode(model, ts, y0, config))(y0_batch)
    
    # Create plots
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, len(y0_batch))):
        for dim in range(min(3, target_batch.shape[2])):  # Plot up to 3 dimensions
            axes[i, dim].plot(ts, target_batch[i, :, dim], 'b-', linewidth=2,
                             label=f'True Dim {dim}')
            axes[i, dim].plot(ts, predictions[i, :, dim], 'r--', linewidth=2,
                             label=f'Predicted Dim {dim}')
            axes[i, dim].set_xlabel('Time')
            axes[i, dim].set_ylabel(f'State {dim}')
            axes[i, dim].grid(True, alpha=0.3)
            if i == 0:
                axes[i, dim].legend()
    
    plt.tight_layout()
    
    # Save plot (only save, don't show)
    try:
        filename = f"{save_dir}/neural_ode_trajectories.{format_}"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Trajectories saved to {filename}")
    except Exception as e:
        print(f"Warning: Could not save trajectory plot: {e}")
        plt.close(fig)


def plot_phase_space(model: NeuralODEModel, test_data: Dict[str, Any],
                     config: Dict[str, Any], num_samples: int = 3) -> None:
    """
    Plot phase space visualization if enabled in configuration.
    
    Args:
        model: Trained neural ODE model
        test_data: Test data dictionary
        config: Configuration
        num_samples: Number of samples to plot
    """
    # Check if visualization is enabled
    if not config.get('visualization', {}).get('enabled', False):
        return
    
    # Extract visualization settings with defaults
    save_dir = config.get('visualization', {}).get('save_dir', 'exp/')
    format_ = config.get('visualization', {}).get('format', 'png')
    dpi = config.get('visualization', {}).get('dpi', 300)
    
    # Create save directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Warning: Could not create directory {save_dir}: {e}")
        return
    
    ts = test_data['ts']
    y0_batch = test_data['initial_conditions']
    target_batch = test_data['trajectories']
    
    # Get predictions
    predictions = jax.vmap(lambda y0: solve_neural_ode(model, ts, y0, config))(y0_batch)
    
    # Create phase space plots
    fig = plt.figure(figsize=(15, 5 * num_samples))
    
    for i in range(min(num_samples, len(y0_batch))):
        # 3D phase space plot if we have 3 dimensions
        if target_batch.shape[2] >= 3:
            ax = fig.add_subplot(num_samples, 3, i * 3 + 1, projection='3d')
            ax.plot3D(target_batch[i, :, 0], target_batch[i, :, 1], target_batch[i, :, 2],
                       'b-', linewidth=2, label='True')
            ax.plot3D(predictions[i, :, 0], predictions[i, :, 1], predictions[i, :, 2],
                       'r--', linewidth=2, label='Predicted')
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_zlabel('Dim 2')
            ax.set_title(f'Sample {i+1} - 3D Phase Space')
            ax.legend()
        
        # 2D phase plots
        if target_batch.shape[2] >= 2:
            ax = fig.add_subplot(num_samples, 3, i * 3 + 2)
            ax.plot(target_batch[i, :, 0], target_batch[i, :, 1], 'b-', linewidth=2,
                   label='True')
            ax.plot(predictions[i, :, 0], predictions[i, :, 1], 'r--', linewidth=2,
                   label='Predicted')
            ax.set_xlabel('Dim 0')
            ax.set_ylabel('Dim 1')
            ax.set_title(f'Sample {i+1} - Phase Plot')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    
    # Save plot (only save, don't show)
    try:
        filename = f"{save_dir}/neural_ode_phase_space.{format_}"
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Phase space plot saved to {filename}")
    except Exception as e:
        print(f"Warning: Could not save phase space plot: {e}")
        plt.close(fig)


#%%
# Utility Functions
def setup_jax_environment(use_64bit: bool = True) -> None:
    """
    Setup JAX environment with appropriate settings.
    
    Args:
        use_64bit: Whether to enable 64-bit precision
    """
    if use_64bit:
        jax.config.update("jax_enable_x64", True)
    
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"64-bit precision: {jax.config.jax_enable_x64}")


def create_dataloader(data: Dict[str, Any], batch_size: int, *, 
                     key: jax.random.PRNGKey) -> Callable:
    """
    Create a dataloader for neural ODE training.
    
    Args:
        data: Data dictionary with 'initial_conditions' and 'trajectories'
        batch_size: Batch size
        key: JAX random key
        
    Returns:
        Generator function that yields batches
    """
    y0_batch = data['initial_conditions']
    target_batch = data['trajectories']
    ts = data['ts']
    
    dataset_size = y0_batch.shape[0]
    indices = jnp.arange(dataset_size)
    
    def dataloader():
        while True:
            # Shuffle indices
            perm = jr.permutation(key, indices)
            
            # Create batches
            start = 0
            while start + batch_size <= dataset_size:
                batch_indices = perm[start:start + batch_size]
                yield (ts, y0_batch[batch_indices], target_batch[batch_indices])
                start += batch_size
    
    return dataloader


#%%
# Example Usage and Testing
def main_example():
    """Example of using the neural ODE functions."""
    print("Neural ODE Functions Example")
    print("=" * 40)
    
    # Setup environment
    setup_jax_environment(use_64bit=True)
    
    # Create configuration
    config = create_neural_ode_config(
        hidden_dim=32,
        num_layers=2,
        output_dim=3,
        dataset_size=128,
        num_steps=500,
        batch_size=16,
        visualization_enabled=True,
        simulation_time=1.0,
        sample_rate=50
    )
    
    print("Configuration created:")
    for section, params in config.items():
        print(f"  {section}: {params}")
    
    # Generate data
    print("\nGenerating synthetic data...")
    key = jr.PRNGKey(42)
    ts, train_data, test_data = generate_synthetic_data(config, key=key)
    
    print(f"Training data shape: {train_data['trajectories'].shape}")
    print(f"Test data shape: {test_data['trajectories'].shape}")
    
    # Create model
    print("\nCreating neural ODE model...")
    model_key = jr.PRNGKey(123)
    model = NeuralODEModel(
        data_size=config['model']['output_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        solver_type=config['solver']['solver_type'],
        activation=config['model']['activation'],
        key=model_key
    )
    
    # Train model
    print("\nTraining model...")
    trained_model, history = train_neural_ode(model, train_data, config, test_data)
    
    # Evaluate
    print("\nEvaluating model...")
    test_loss, test_metrics = evaluate_model_step(trained_model, test_data, config['solver'])
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_training_history(history, config)
    plot_trajectories(trained_model, test_data, config)
    plot_phase_space(trained_model, test_data, config)
    
    print("\nExample completed successfully!")
    return trained_model, history, config


if __name__ == "__main__":
    # Run the example
    model, history, config = main_example()