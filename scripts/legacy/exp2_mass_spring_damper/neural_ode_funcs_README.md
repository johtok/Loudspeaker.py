# Neural ODE Functions Module

This module provides a comprehensive, modular collection of functions for neural ODE experiments, designed to be reusable and composable for various scientific computing applications.

## Overview

The `neural_ode_funcs.py` module is structured around the following key components:

1. **Configuration Management** - Centralized, flexible configuration system
2. **Neural Network Architecture** - Equinox-based neural ODE models
3. **Data Generation** - Synthetic ODE data generation with forcing
4. **Training Functions** - Complete training pipeline with gradient computation
5. **Solving Functions** - ODE solving with multiple solver options
6. **Evaluation Functions** - Comprehensive model evaluation and metrics
7. **Visualization Functions** - Optional plotting with config control

## Installation and Dependencies

The module requires the following dependencies (available in the pixi environment):

```bash
# Core JAX ecosystem
jax>=0.4.0
jaxlib>=0.4.0
diffrax>=0.3.0
equinox>=0.6.0

# Optimization and utilities
optax>=0.1.0

# Visualization
matplotlib>=3.5.0
numpy>=1.21.0
```

## Quick Start

```python
import jax.random as jr
from scripts.exp2_mass_spring_damper.neural_ode_funcs import *

# Setup JAX environment
setup_jax_environment(use_64bit=True)

# Create configuration
config = create_neural_ode_config(
    hidden_dim=64,
    num_layers=3,
    output_dim=3,
    dataset_size=256,
    num_steps=1000,
    visualization_enabled=True
)

# Generate synthetic data
key = jr.PRNGKey(42)
ts, train_data, test_data = generate_synthetic_data(config, key=key)

# Create and train model
model_key = jr.PRNGKey(123)
model = NeuralODEModel(
    data_size=config['model']['output_dim'],
    hidden_dim=config['model']['hidden_dim'],
    num_layers=config['model']['num_layers'],
    key=model_key
)

trained_model, history = train_neural_ode(model, train_data, config, test_data)

# Evaluate and visualize
metrics = evaluate_model(trained_model, test_data, config)
plot_training_history(history, config)
plot_trajectories(trained_model, test_data, config)
```

## Configuration System

The module uses a comprehensive configuration system with sensible defaults:

### Configuration Structure

```python
config = {
    'model': {
        'hidden_dim': 64,
        'num_layers': 3,
        'output_dim': 3,  # position, velocity, acceleration
        'activation': 'softplus'  # 'softplus', 'tanh', 'relu'
    },
    'training': {
        'learning_rate': 1e-3,
        'num_steps': 1000,
        'batch_size': 32,
        'weight_decay': 1e-4,
        'optimizer': 'adam'  # 'adam', 'adabelief', 'sgd'
    },
    'solver': {
        'dt': 1e-3,
        'solver_type': 'tsit5',  # 'tsit5', 'kvaerno5', 'dopri5'
        'rtol': 1e-3,
        'atol': 1e-6,
        'adaptive_steps': True
    },
    'data': {
        'dataset_size': 256,
        'test_split': 0.2,
        'noise_level': 0.01,
        'simulation_time': 1.0,
        'sample_rate': 100,
        'initial_condition_range': (-1.0, 1.0)
    },
    'forcing': {
        'enabled': True,
        'type': 'pink_noise',  # 'pink_noise', 'sine', 'chirp', 'step'
        'amplitude': 1.0,
        'frequency_range': (0.1, 10.0),
        'exponent': 1.0
    },
    'visualization': {
        'enabled': False,  # Default off for experiments
        'save_dir': 'exp/',
        'format': 'png',
        'dpi': 300
    },
    'evaluation': {
        'eval_frequency': 100,
        'early_stopping': True,
        'patience': 50
    },
    'numerical': {
        'use_64bit': True,
        'gradient_clipping': 1.0
    }
}
```

### Creating Custom Configurations

```python
# Override specific parameters
config = create_neural_ode_config(
    hidden_dim=128,
    learning_rate=5e-4,
    num_steps=2000,
    solver_type='kvaerno5',  # For stiff systems
    forcing_type='sine',
    visualization_enabled=True
)

# Or modify existing config
config['model']['activation'] = 'tanh'
config['training']['optimizer'] = 'adabelief'
```

## Core Components

### 1. Neural Network Architecture

#### NeuralODEFunc
The vector field function using MLP architecture:

```python
class NeuralODEFunc(eqx.Module):
    """Vector field function for neural ODE using MLP architecture."""
    
    def __init__(self, data_size: int, hidden_dim: int, num_layers: int, 
                 activation: str = 'softplus', *, key: jax.random.PRNGKey):
        # Initialize MLP with learnable output scaling
        
    def __call__(self, t: float, y: jnp.ndarray, args: Any) -> jnp.ndarray:
        # Compute vector field f(t, y, args)
        return self.out_scale * self.mlp(y)
```

#### NeuralODEModel
The complete neural ODE model with initial condition mapping:

```python
class NeuralODEModel(eqx.Module):
    """Complete neural ODE model with initial condition mapping."""
    
    def __call__(self, ts: jnp.ndarray, y0: jnp.ndarray, 
                 solver_config: Dict[str, Any] = None) -> jnp.ndarray:
        # Solve neural ODE from initial condition
        return solution.ys
```

### 2. Data Generation

#### Synthetic Data Generation
Generates synthetic ODE data with various forcing types:

```python
ts, train_data, test_data = generate_synthetic_data(config, key=key)

# Data structure:
train_data = {
    'ts': time_points,           # Shape: (num_points,)
    'initial_conditions': y0,    # Shape: (batch_size, output_dim)
    'trajectories': trajectories, # Shape: (batch_size, num_points, output_dim)
    'forcing': forcing_signals    # Shape: (batch_size, num_points) or None
}
```

#### Supported Forcing Types
- **Pink Noise**: 1/f noise with configurable spectral exponent
- **Sine Waves**: Single frequency sinusoidal forcing
- **Chirp Signals**: Frequency-swept forcing
- **Step Inputs**: Constant amplitude step functions

### 3. Training Pipeline

#### Complete Training Loop
```python
trained_model, history = train_neural_ode(
    model, train_data, config, test_data=test_data
)

# Training history structure:
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
```

#### Single Training Step
```python
loss, metrics, new_model, new_opt_state = train_step(
    model, optimizer, opt_state, batch, solver_config
)
```

### 4. Evaluation and Metrics

#### Comprehensive Evaluation
```python
metrics = evaluate_model(trained_model, test_data, config)
# Returns: {'total_mse': float, 'total_rmse': float, ...}
```

#### Per-Dimension Metrics
```python
detailed_metrics = compute_metrics(predictions, targets)
# Returns: {
#     'total_mse': float,
#     'total_rmse': float,
#     'relative_error': float,
#     'max_error': float,
#     'r2_score': float,
#     'dim_mse': jnp.ndarray,
#     'dim_rmse': jnp.ndarray
# }
```

#### Differentiability Test
```python
diff_test = differentiability_test(model, ts, y0, config)
# Returns: {
#     'gradients_computable': bool,
#     'gradients_finite': bool,
#     'gradients_nonzero': bool,
#     'test_passed': bool
# }
```

### 5. ODE Solving

#### Basic Neural ODE Solving
```python
solution = solve_neural_ode(model, ts, y0, config)
```

#### Solving with Forcing (Future Extension)
```python
# Placeholder for future forcing support
solution = solve_with_forcing(model, ts, y0, forcing_signal, config)
```

### 6. Visualization

All visualization functions respect the `config['visualization']['enabled']` setting.

#### Training History
```python
plot_training_history(history, config)
# Shows: training/test loss, RMSE, per-dimension MSE, step times
```

#### Trajectory Comparison
```python
plot_trajectories(trained_model, test_data, config, num_samples=5)
# Shows: true vs predicted trajectories for multiple samples
```

#### Phase Space Visualization
```python
plot_phase_space(trained_model, test_data, config, num_samples=3)
# Shows: 3D phase space plots and 2D phase plots
```

## Advanced Usage

### Custom ODE Systems

To use with custom ODE systems, modify the `_generate_trajectories` function:

```python
def custom_dynamics(t: float, y: jnp.ndarray, forcing_val: float) -> jnp.ndarray:
    # Your custom ODE equations here
    return derivatives

# Modify data generation to use custom dynamics
def generate_custom_data(config, key):
    # Use custom_dynamics in _generate_trajectories
    pass
```

### Custom Neural Architectures

Extend the existing classes for custom architectures:

```python
class CustomNeuralODEFunc(eqx.Module):
    def __init__(self, data_size, hidden_dim, *, key):
        # Your custom architecture
        
    def __call__(self, t, y, args):
        # Your custom vector field computation
        return custom_vector_field
```

### Solver Configuration

Configure different solvers for different scenarios:

```python
# For stiff systems
solver_config = {
    'solver': diffrax.Kvaerno5(),
    'rtol': 1e-6,
    'atol': 1e-8,
    'adaptive_steps': True
}

# For non-stiff systems
solver_config = {
    'solver': diffrax.Tsit5(),
    'rtol': 1e-3,
    'atol': 1e-6,
    'adaptive_steps': True
}
```

## Performance Considerations

### 64-bit Precision
```python
setup_jax_environment(use_64bit=True)  # Enabled by default
```

### Gradient Clipping
```python
config = create_neural_ode_config(gradient_clipping=1.0)  # Prevents gradient explosion
```

### Batch Size and Memory
- Larger batch sizes improve training stability but require more memory
- Reduce batch size if encountering memory issues
- Consider using gradient accumulation for effective large batches

### Solver Selection
- **Tsit5**: Good general-purpose solver for non-stiff systems
- **Kvaerno5**: Better for stiff systems (high damping ratios)
- **Dopri5**: Alternative general-purpose solver

## Integration with Existing Code

### Using with Existing Models
```python
# Convert existing model to neural ODE format
def wrap_existing_model(existing_model):
    class WrappedNeuralODE(NeuralODEModel):
        def __init__(self, config, key):
            super().__init__(config['output_dim'], config['hidden_dim'], 
                           config['num_layers'], key=key)
            self.existing_model = existing_model
            
        def __call__(self, ts, y0, solver_config=None):
            # Use existing model in neural ODE framework
            return solve_with_existing_model(self.existing_model, ts, y0)
```

### Data Compatibility
```python
# Convert existing data to neural ODE format
def prepare_existing_data(time_series, initial_conditions):
    data_format = {
        'ts': time_points,
        'initial_conditions': initial_conditions,
        'trajectories': time_series,
        'forcing': None  # or forcing_signals
    }
    return data_format
```

## Error Handling and Debugging

### Common Issues

1. **NaN Values in Training**
   ```python
   # Check for NaN in gradients
   if not jnp.all(jnp.isfinite(grads)):
       print("NaN gradients detected!")
   ```

2. **Solver Convergence Issues**
   ```python
   # Adjust solver tolerances
   config = create_neural_ode_config(
       solver_type='kvaerno5',  # More stable for stiff systems
       rtol=1e-6,               # Tighter tolerances
       atol=1e-8
   )
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size or simulation length
   config = create_neural_ode_config(
       batch_size=16,           # Smaller batches
       simulation_time=0.5      # Shorter simulations
   )
   ```

### Debugging Tools

```python
# Check model differentiability
diff_test = differentiability_test(model, ts, y0, config)
print(f"Differentiability test passed: {diff_test['test_passed']}")

# Monitor training progress
def monitor_training(history):
    if len(history['train_loss']) > 10:
        recent_loss = np.mean(history['train_loss'][-10:])
        if recent_loss > history['best_loss'] * 1.5:
            print("Warning: Training loss increasing!")

# Visual inspection
plot_trajectories(trained_model, test_data, config)
```

## Examples and Use Cases

### Mass-Spring-Damper System Identification
```python
# Configure for MSD system
config = create_neural_ode_config(
    output_dim=3,              # position, velocity, acceleration
    forcing_enabled=True,
    forcing_type='pink_noise',
    solver_type='kvaerno5',    # For potentially stiff dynamics
    dataset_size=512,
    num_steps=2000
)

# Train and evaluate
model, history = train_neural_ode(model, train_data, config, test_data)
metrics = evaluate_model(model, test_data, config)
```

### Chaotic System Modeling
```python
# Configure for chaotic systems
config = create_neural_ode_config(
    hidden_dim=128,            # Larger capacity
    num_layers=4,
    learning_rate=1e-4,        # Lower learning rate
    rtol=1e-8,                 # Higher precision
    atol=1e-10,
    simulation_time=2.0,       # Longer trajectories
    eval_frequency=50          # More frequent evaluation
)
```

### Multi-Modal Forcing Analysis
```python
# Compare different forcing types
forcing_types = ['pink_noise', 'sine', 'chirp', 'step']
results = {}

for forcing_type in forcing_types:
    config = create_neural_ode_config(
        forcing_type=forcing_type,
        visualization_enabled=False  # Disable for batch processing
    )
    
    # Train model
    model, history = train_neural_ode(model, train_data, config, test_data)
    
    # Evaluate
    metrics = evaluate_model(model, test_data, config)
    results[forcing_type] = metrics
    
    print(f"{forcing_type}: RMSE={metrics['total_rmse']:.6f}")
```

## Contributing and Extensions

### Adding New Features

1. **New Solver Types**: Extend solver configuration in `solve_neural_ode`
2. **New Forcing Types**: Add to `_generate_forcing_signals`
3. **New Metrics**: Extend `compute_metrics` function
4. **New Visualizations**: Add new plot functions following the pattern

### Best Practices

1. **Modular Design**: Keep functions focused and composable
2. **Configuration Control**: Use config dict for all parameters
3. **Error Handling**: Include comprehensive error checking
4. **Type Hints**: Use proper type annotations
5. **Documentation**: Document all public functions

## References

This module is based on:
- [Neural ODEs](https://arxiv.org/abs/1806.07366) - Chen et al.
- [Diffrax](https://github.com/patrick-kidger/diffrax) - JAX-based ODE/ SDE solver
- [Equinox](https://github.com/patrick-kidger/equinox) - JAX neural networks
- [Optax](https://github.com/deepmind/optax) - JAX optimization library

For more information, see the inline code documentation and examples.