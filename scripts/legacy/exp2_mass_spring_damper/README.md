# Mass-Spring-Damper Simulation with Forcing

This comprehensive simulation script implements advanced mass-spring-damper (MSD) system modeling with external forcing, leveraging JAX/diffrax for high-performance numerical computing.

## Features

### Core Capabilities
- **Advanced ODE Solvers**: Uses diffrax with Kvaerno5 (for stiff systems) and Tsit5 (for non-stiff systems)
- **External Forcing Integration**: Supports multiple forcing types including pink noise, sine waves, and complex signals
- **3D Phase Space Visualization**: Creates comprehensive position-velocity-acceleration plots
- **Normalized Phase Plots**: Generates properly scaled phase space visualizations
- **Batch Simulation**: Efficiently run multiple simulations with different forcing signals
- **Frequency Analysis**: Built-in FFT and transfer function estimation
- **64-bit Precision**: Numerical accuracy for scientific computing

### Forcing Signal Types
- **Pink Noise**: 1/f noise with configurable spectral exponent and bandpass filtering
- **Sine Waves**: Single frequency sinusoidal forcing
- **Complex Sine**: Multi-frequency sine combinations
- **Chirp Signals**: Frequency-swept forcing
- **Step Inputs**: Constant amplitude step functions

### Normalization Methods
- **Standardize**: (x - mean) / std - Each trajectory (position, velocity, acceleration) is normalized by its own mean and standard deviation independently: x/std(x), v/std(v), a/std(a)
- **MinMax**: (x - min) / (max - min) - Each trajectory normalized by its own min/max values
- **Unit Vector**: x / ||x|| - Normalize by the vector norm
- **None**: No normalization

## Installation

The script requires the following dependencies (already available in the pixi environment):

```bash
# Core dependencies (included in pixi environment)
jax>=0.4.0
jaxlib>=0.4.0
diffrax>=0.3.0
equinox>=0.6.0
numpy>=1.21.0
scipy>=1.9.0
matplotlib>=3.5.0
```

## Usage

### Basic Usage

```bash
# Run the full demonstration
pixi run python scripts/exp2_mass_spring_damper/msd_simulation_with_forcing.py

# Run with custom configuration
pixi run python -c "
from scripts.exp2_mass_spring_damper.msd_simulation_with_forcing import MSDConfig, run_single_simulation
config = MSDConfig(simulation_time=0.1, natural_frequency=50.0)
result = run_single_simulation(config)
"
```

### Programmatic Usage

```python
import sys
sys.path.append('.')

from scripts.exp2_mass_spring_damper.msd_simulation_with_forcing import (
    MSDConfig, ForcingType, SolverType, run_single_simulation,
    run_batch_simulation, demonstrate_solver_comparison
)

# Configure simulation
config = MSDConfig(
    mass=0.1,                    # kg
    natural_frequency=30.0,      # Hz
    damping_ratio=0.02,          # damping ratio
    sample_rate=2000,            # Hz
    simulation_time=0.2,         # seconds
    forcing_type=ForcingType.PINK_NOISE,
    solver_type=SolverType.KVAERNO5,
    rtol=1e-8,                   # relative tolerance
    atol=1e-8,                   # absolute tolerance
    save_plots=True,             # save visualization outputs
    normalize_plots=True
)

# Run single simulation
result = run_single_simulation(config)

# Run batch simulation
batch_result = run_batch_simulation(config)

# Compare different solvers
solver_results = demonstrate_solver_comparison()
```

### Configuration Options

The `MSDConfig` class provides comprehensive configuration:

```python
config = MSDConfig(
    # Physical parameters
    mass=0.05,                    # Mass (kg)
    natural_frequency=25.0,       # Natural frequency (Hz)
    damping_ratio=0.01,           # Damping ratio

    # Simulation parameters
    sample_rate=1000,             # Sample rate (Hz)
    simulation_time=0.1,          # Simulation duration (s)
    initial_conditions=(0.0, 0.0), # Initial [position, velocity]

    # Forcing parameters
    forcing_type=ForcingType.PINK_NOISE,
    forcing_amplitude=1.0,        # Forcing amplitude
    frequency_range=(0.01, 400.0), # Frequency range (Hz)
    pink_noise_exponent=1.0,      # Pink noise spectral exponent

    # Solver parameters
    solver_type=SolverType.KVAERNO5,
    rtol=1e-8,                    # Relative tolerance
    atol=1e-8,                    # Absolute tolerance
    dt0=None,                     # Initial time step (auto if None)

    # Visualization parameters
    normalize_plots=True,
    normalization_type=NormalizationType.STANDARDIZE,
    save_plots=True,
    plot_format='png',

    # Data generation
    batch_size=10,
    seed=1234
)
```

## Output Files

The script generates several visualization files:

- `msd_time_domain.png` - Time-domain plots of forcing and response
- `msd_3d_phase_space.png` - 3D phase space visualization
- `msd_normalized_phase.png` - Normalized phase space plots
- `msd_frequency_analysis.png` - FFT and frequency response analysis
- `msd_batch_comparison.png` - Comparison of multiple simulations
- `solver_comparison.png` - Comparison of different ODE solvers
- `forcing_types_comparison.png` - Comparison of different forcing types

## Normalization Behavior

The script implements proper trajectory-wise normalization where each state variable (position, velocity, acceleration) is normalized independently by its own statistical properties:

### Standardize Normalization
For state data with columns [position, velocity, acceleration]:
- Position: `position_normalized = (position - mean(position)) / std(position)`
- Velocity: `velocity_normalized = (velocity - mean(velocity)) / std(velocity)`
- Acceleration: `acceleration_normalized = (acceleration - mean(acceleration)) / std(acceleration)`

This ensures that:
- Each trajectory is scaled to have unit variance
- All trajectories are on comparable scales for visualization
- Phase space plots show proper relative relationships
- No single dimension dominates the visualization due to scale differences

### Why Independent Normalization Matters
- **Position** typically has units of meters (e.g., 10^-3 to 10^-6 m)
- **Velocity** has units of m/s (e.g., 10^-1 to 10^-3 m/s)
- **Acceleration** has units of m/s² (e.g., 10^0 to 10^1 m/s²)

Without independent normalization, acceleration would dominate the visualization due to its larger numerical values, masking the dynamics of position and velocity.

## Key Components

### 1. Configuration System
- `MSDConfig`: Dataclass with all simulation parameters
- `ForcingType`: Enum for different forcing signal types
- `SolverType`: Enum for different ODE solvers
- `NormalizationType`: Enum for normalization methods

### 2. Core Simulation Classes
- `PinkNoiseGenerator`: Generates pink noise with configurable parameters
- `MassSpringDamperSystem`: Defines the ODE system dynamics
- `MSDSimulator`: Main simulation orchestrator
- `MSDVisualizer`: Comprehensive visualization tools

### 3. Utility Functions
- `run_single_simulation()`: Single simulation with full visualization
- `run_batch_simulation()`: Batch simulations with comparison plots
- `demonstrate_solver_comparison()`: Compare different solvers

## Performance Considerations

- **64-bit Precision**: Enabled by default for numerical accuracy
- **JAX JIT Compilation**: Automatic for performance optimization
- **Solver Selection**: Kvaerno5 for stiff systems, Tsit5 for non-stiff
- **Adaptive Time Stepping**: Automatic step size adjustment for accuracy

## Example Results

The script demonstrates:

1. **Single Simulation**: Complete analysis of one forcing-response scenario
2. **Batch Analysis**: Multiple simulations with statistical comparison
3. **Solver Comparison**: Performance and accuracy of different ODE solvers
4. **Forcing Type Analysis**: Response characteristics to different input signals

## Integration with Julia Examples

This Python implementation draws inspiration from the Julia `Loudspeaker.jl` package, particularly:
- `exp2_normalized_pink_descent.jl` - Pink noise generation and system identification
- Similar mass-spring-damper modeling approaches
- Comparable visualization techniques

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Memory Issues**: Reduce `batch_size` or `simulation_time` for large simulations
3. **Solver Convergence**: Adjust `rtol` and `atol` or try different solver types
4. **Plotting Issues**: Ensure matplotlib backend is properly configured

### Performance Tips

- Use `Kvaerno5` for stiff systems (high damping ratios)
- Use `Tsit5` for non-stiff systems (low damping ratios)
- Increase tolerances (`rtol`, `atol`) for faster but less accurate results
- Use shorter simulation times for rapid prototyping

## Advanced Usage

### Custom Forcing Signals

```python
# Create custom forcing function
def custom_forcing_generator(config, length):
    t = jnp.linspace(0, config.simulation_time, length)
    # Your custom signal generation here
    return custom_signal

# Use with simulator
simulator = MSDSimulator(config)
# Modify the noise generator or pass custom forcing directly
```

### Custom Solvers

```python
# The script supports any diffrax solver
from diffrax import Euler, Dopri8

# Add to SolverType enum or modify solver selection logic
```

### Custom Visualizations

```python
# Extend MSDVisualizer class with custom plot methods
class ExtendedVisualizer(MSDVisualizer):
    def custom_plot(self, result):
        # Your custom visualization code
        pass
```

## Citation and References

This implementation is based on:
- Diffrax documentation and examples
- JAX ecosystem for high-performance numerical computing
- Mass-spring-damper modeling principles
- Signal processing and control systems theory

For more information, see the inline code documentation and the referenced examples in the loudspeaker thesis project.
