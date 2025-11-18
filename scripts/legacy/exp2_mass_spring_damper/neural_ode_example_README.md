# Neural ODE Mass-Spring-Damper Example

This document describes the refactored `neural_ode_example.py` script that demonstrates how to use Neural ODEs for mass-spring-damper system identification using the modular `neural_ode_funcs.py` library.

## Overview

The refactored script follows the patterns from the example notebooks while integrating with the comprehensive mass-spring-damper simulation capabilities. It replaces the original Neural CDE implementation with a Neural ODE approach using modular, reusable functions.

## Key Features

- **Modular Architecture**: Uses `neural_ode_funcs.py` for all core functionality
- **Mass-Spring-Damper Integration**: Leverages `msd_simulation_with_forcing.py` for realistic data generation
- **Configuration-Driven**: Uses a comprehensive config system with sensible defaults
- **3D State Representation**: Models position, velocity, and acceleration
- **Optional Visualization**: Comprehensive plotting with config control
- **JAX 64-bit Precision**: Ensures numerical accuracy for ODE solving

## File Structure

### Main Script
- `neural_ode_example.py` - Main refactored example script

### Supporting Modules
- `neural_ode_funcs.py` - Modular neural ODE functions library
- `msd_simulation_with_forcing.py` - Mass-spring-damper simulation with forcing

### Test Files
- `test_neural_ode_integration.py` - Integration test for the refactored example
- `test_neural_ode_funcs.py` - Tests for the neural_ode_funcs module

## Installation Requirements

```bash
pip install jax jaxlib
pip install equinox
pip install diffrax
pip install optax
pip install matplotlib
pip install numpy
pip install scipy
```

## Usage

### Basic Execution
```bash
python scripts/exp2_mass_spring_damper/neural_ode_example.py
```

### Jupyter/IDE Section Execution
The script is organized with `#%%` sections for easy execution in Jupyter notebooks or IDEs with section support:

1. **Imports and Setup** - Import all required libraries
2. **Configuration** - Create comprehensive configuration
3. **Data Generation** - Generate synthetic MSD data with forcing
4. **Model Definition** - Create Neural ODE model
5. **Training** - Train the model with progress tracking
6. **Evaluation** - Evaluate model performance
7. **Visualization** - Generate plots (if enabled)

## Configuration Options

The script uses `create_neural_ode_config()` with the following key parameters:

### Model Parameters
- `hidden_dim`: Hidden layer size (default: 64)
- `num_layers`: Number of layers (default: 3)
- `output_dim`: Output dimension (default: 3 for position/velocity/acceleration)

### Data Parameters
- `dataset_size`: Number of training examples (default: 256)
- `simulation_time`: Simulation duration in seconds (default: 0.1)
- `sample_rate`: Sampling frequency (default: 1000 Hz)
- `msd_params`: Mass-spring-damper specific parameters

### Training Parameters
- `num_steps`: Training steps (default: 1000)
- `learning_rate`: Learning rate (default: 1e-3)
- `batch_size`: Batch size (default: 32)

### Visualization Parameters
- `visualization_enabled`: Enable/disable plots (default: False)
- `save_dir`: Directory for saving plots (default: 'exp/')
- `plot_format`: Image format (default: 'png')

## Data Generation

The script automatically detects if `msd_simulation_with_forcing` is available and uses it for realistic mass-spring-damper data generation with:

- **Pink Noise Forcing**: Advanced 1/f noise generation
- **Physical Parameters**: Mass, stiffness, and damping coefficients
- **3D State Data**: Position, velocity, and acceleration trajectories
- **Batch Simulation**: Efficient generation of multiple training examples

If `msd_simulation_with_forcing` is not available, it falls back to generic synthetic data generation.

## Model Architecture

The `NeuralODEModel` consists of:

1. **NeuralODEFunc**: MLP-based vector field function
2. **Initial Mapping**: MLP for initial condition transformation
3. **Solver Integration**: Diffrax ODE solver with adaptive stepping

### Key Components
- **Vector Field**: `f(t, y) = scale * tanh(MLP(y))`
- **Learnable Scaling**: Prevents model blowup
- **Tanh Activation**: Ensures bounded outputs
- **Adaptive Solvers**: Tsit5, Kvaerno5, or Dopri5

## Training Process

The training follows these steps:

1. **Data Preparation**: Generate or load training data
2. **Model Initialization**: Create neural network parameters
3. **Optimization Loop**: Adam optimizer with gradient clipping
4. **Progress Tracking**: Monitor loss, RMSE, and per-dimension metrics
5. **Early Stopping**: Optional validation-based stopping
6. **Evaluation**: Comprehensive metrics on test set

## Evaluation Metrics

The script computes multiple evaluation metrics:

- **MSE/RMSE**: Mean squared error and root mean squared error
- **Relative Error**: Relative error compared to true values
- **Per-Dimension Metrics**: Separate evaluation for position, velocity, acceleration
- **R² Score**: Coefficient of determination
- **Maximum Error**: Worst-case prediction error

## Visualization Options

When `visualization_enabled=True`, the script generates:

1. **Training History**: Loss and metric progression over time
2. **Trajectory Plots**: True vs predicted time series for multiple samples
3. **Phase Space**: 2D and 3D phase space visualizations
4. **Per-Dimension Analysis**: Separate plots for each state dimension

All plots are saved to the specified `save_dir` with the configured format.

## Integration with Original CDE Example

The refactored script maintains compatibility with the original `neural_cde_msd_example.py` by:

- Using the same mass-spring-damper physics
- Supporting the same forcing types (pink noise, sine, etc.)
- Providing similar evaluation and visualization capabilities
- Maintaining the same configuration structure

However, it replaces the Neural CDE approach with Neural ODEs for better interpretability and computational efficiency.

## Performance Considerations

- **64-bit Precision**: Enabled by default for numerical accuracy
- **JAX Compilation**: Automatic JIT compilation for performance
- **Batch Processing**: Efficient vectorized operations
- **Memory Management**: Proper handling of large datasets
- **Solver Selection**: Adaptive solvers for different stiffness levels

## Error Handling

The script includes comprehensive error handling for:

- **Import Errors**: Graceful fallbacks when optional modules are missing
- **Numerical Issues**: Detection and handling of NaN/inf values
- **Solver Failures**: Robust ODE solving with error recovery
- **Configuration Validation**: Input parameter checking
- **Memory Issues**: Proper handling of large datasets

## Testing

Run the integration test to verify the refactored implementation:

```bash
python scripts/exp2_mass_spring_damper/test_neural_ode_integration.py
```

This test verifies:
- All imports work correctly
- Configuration system functions properly
- Data generation integrates with MSD simulation
- Model creation and training pipeline works
- Basic functionality without full training run

## Future Extensions

The modular design supports easy extensions:

1. **New Solvers**: Add support for additional ODE solvers
2. **Custom Forcing**: Implement new forcing signal types
3. **Advanced Architectures**: Add attention mechanisms or residual connections
4. **Multi-Physics**: Extend to other physical systems
5. **Real-Time Inference**: Optimize for deployment scenarios

## References

- [Neural ODE Paper](https://arxiv.org/abs/1806.07366)
- [Diffrax Documentation](https://docs.kidger.site/diffrax/)
- [Equinox Documentation](https://docs.kidger.site/equinox/)
- [Mass-Spring-Damper Theory](https://en.wikipedia.org/wiki/Harmonic_oscillator)
