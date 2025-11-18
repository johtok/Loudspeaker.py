# Loudspeaker Py - CFES Testing Framework

A comprehensive Python framework for testing Controlled Functional Expansion Systems (CFES) in Python, featuring both JAX/diffrax and PyTorch implementations. This framework is designed for loudspeaker modeling and general dynamical system identification using modern machine learning techniques.

## 🚀 Features

- **Dual Framework Support**: Implementations in both JAX/diffrax and PyTorch
- **Neural Controlled Differential Equations (Neural CDEs)**: State-of-the-art time series modeling
- **Modular Design**: Clean, well-structured code with separate modules for models, solvers, and utilities
- **Jupyter Integration**: Full Jupyter notebook support with custom kernel
- **TensorBoard Logging**: Comprehensive experiment tracking and visualization
- **Comprehensive Testing**: Extensive test suite for all components
- **CLI Interface**: Command-line tools for training, evaluation, and data generation
- **Scale-Aware Losses**: Normalized MSE (`norm_mse`) is the default loss for examples, keeping optimization stable across targets

## 🎯 Project Direction: Type-Based Functional Training

Loudspeaker.py is evolving toward a fully *type-driven, functional training system*. Core components such as datasets, dataloaders, loss builders, and neural ODE wrappers are now exposed as strongly typed dataclasses and pure functions. Training flows (e.g., `NeuralODE`) pass immutable configs and interchangeable callables rather than relying on implicit globals. This design lets you compose experiments declaratively, swap solvers or forcing generators safely, and leverage static type checkers to validate end-to-end pipelines. Expect future modules to follow the same typed-functional contract so that experiments, scripts, and notebooks can share a single, consistent API surface.

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

This project uses [pyproject.toml](pyproject.toml) for package management. The modern Python packaging standard provides reliable package management and environment management.

### Prerequisites

- Python 3.10 or higher
- pip (comes with Python)

### Quick Setup

1. **Clone and setup the environment:**
   ```bash
   git clone <repository-url>
   cd loudspeaker.py
   pip install -e .[dev]
   ```

2. **Setup Jupyter kernel:**
   ```bash
   python scripts/setup_jupyter_kernel.py
   ```

3. **Verify installation:**
   ```bash
   jupyter lab
   # Select "Loudspeaker Python (CFES)" kernel
   # Run the example_usage.ipynb notebook
   ```

### Installation with Optional Dependencies

Install with specific optional dependencies:

```bash
# Install with Jupyter support
pip install -e .[jupyter]

# Install with development tools
pip install -e .[dev]

# Install everything
pip install -e .[dev,jupyter]
```

### Manual Installation

If you prefer to install dependencies individually:

```bash
# Install core dependencies
pip install numpy scipy pandas matplotlib seaborn plotly
pip install torch torchvision torchaudio
pip install jax jaxlib  # Follow JAX installation guide for GPU support
pip install diffrax equinox optax
pip install tensorboard tqdm rich click

# Install development tools
pip install black isort flake8 mypy pytest pytest-cov pre-commit

# Install Jupyter ecosystem
pip install jupyter jupyterlab ipykernel ipywidgets notebook
```

## 🚀 Quick Start

### 1. JAX/Diffrax Neural CDE Example

```python
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

# Run the complete example
python scripts/neural_cde_example.py
```

This example demonstrates:
- Spiral time series generation
- Neural CDE model definition
- Training with optax
- Visualization of results

### 2. PyTorch CFES Example

```python
import torch
import torch.nn as nn

# Run the PyTorch example
python scripts/pytorch_cfe_example.py
```

This example shows:
- PyTorch Neural CDE implementation
- TensorBoard logging
- Training and validation loops
- Comprehensive visualization

### 3. Using the CLI

```bash
# Train models
loudspeaker-cli train --config configs/example.yaml

# Evaluate models
loudspeaker-cli evaluate --model-path models/best_model.pth --data-path data/test.csv

# Generate data
loudspeaker-cli generate-data --dataset spirals --size 1000 --output data/spirals.csv

# Create visualizations
loudspeaker-cli visualize --input models/best_model.pth --output plots/results.png
```

## 📚 Examples

### Neural CDE with Diffrax

**File**: `scripts/neural_cde_example.py`

A comprehensive example showing how to implement Neural CDEs using JAX and diffrax:

```python
#%%
# Import libraries and setup
import diffrax
import equinox as eqx
import jax.numpy as jnp

#%%
# Define Neural CDE model
class NeuralCDE(eqx.Module):
    # Model implementation...

#%%
# Generate spiral data
ts, coeffs, labels, data_size = get_spiral_data(dataset_size=256)

#%%
# Train model
model, results = train_neural_cde(config)

#%%
# Visualize results
plot_training_history(results)
plot_spiral_predictions(model, config)
```

### PyTorch CFES Implementation

**File**: `scripts/pytorch_cfe_example.py`

PyTorch implementation with TensorBoard logging:

```python
#%%
# Setup PyTorch model
class PyTorchNeuralCDE(nn.Module):
    # Model implementation...

#%%
# Training with TensorBoard
writer = SummaryWriter(log_dir="logs/pytorch_cfe")
# Training loop with logging...

#%%
# Visualization
plot_pytorch_training_history(results)
visualize_spiral_predictions_pytorch(model, config)
```

### Jupyter Notebooks

After running `setup_jupyter_kernel.py`, you can use:

- `example_usage.ipynb` - Basic environment verification
- Open any `.py` script in Jupyter Lab and run sections individually using `#%%` delimiters

## 📁 Project Structure

```
loudspeaker.py/
├── src/loudspeaker/           # Main package
│   ├── models/                   # Neural and dynamical models
│   │   ├── neural_cdes/          # Neural CDE implementations
│   │   ├── dynamical_systems/    # Classical dynamical systems
│   │   └── blackbox_models/      # Black-box model implementations
│   ├── solvers/                  # Differential equation solvers
│   │   ├── ode_solvers/          # ODE solvers
│   │   ├── sde_solvers/          # SDE solvers
│   │   └── cde_solvers/          # CDE solvers
│   ├── data/                     # Data handling utilities
│   ├── utils/                    # General utilities
│   ├── visualization/            # Plotting and visualization
│   └── cli.py                    # Command-line interface
├── scripts/                      # Standalone scripts
│   ├── neural_cde_example.py     # JAX/diffrax example
│   ├── pytorch_cfe_example.py    # PyTorch example
│   ├── setup_jupyter_kernel.py   # Jupyter setup
│   └── ...                       # Other utility scripts
├── tests/                        # Test suite
│   ├── test_models/              # Model tests
│   ├── test_solvers/             # Solver tests
│   └── test_utils/               # Utility tests
├── examples/                     # Example notebooks
├── docs/                         # Documentation
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## 🔧 Usage

### Configuration

Configuration is managed through the `Config` classes in each script or via YAML files:

```python
class Config:
    dataset_size = 256
    batch_size = 32
    learning_rate = 1e-2
    num_steps = 20
    hidden_size = 8
    # ... other parameters
```

### Data Format

Time series data should be formatted as:
- `ts`: Time points (batch_size, seq_len)
- `data`: Input paths (batch_size, seq_len, input_dim)
- `labels`: Target labels (batch_size,)

### Model Training

```python
# JAX/Diffrax
model = NeuralCDE(data_size, hidden_size, width_size, depth, key=key)
model, results = train_neural_cde(config)

# PyTorch
model = PyTorchNeuralCDE(input_dim, hidden_dim, output_dim)
model, results = train_pytorch_cfe(config)
```

### Visualization

Both frameworks include comprehensive visualization:

```python
# Training history
plot_training_history(results)

# Spiral predictions
plot_spiral_predictions(model, config)

# PyTorch specific
plot_pytorch_training_history(results)
visualize_spiral_predictions_pytorch(model, config)
```

### Losses and Metrics

- `build_loss_fn` now defaults to `loss_type="norm_mse"`, which computes a normalized MSE by squaring the normalized RMSE (`nrmse`). This keeps losses dimensionless and makes experiments consistent regardless of signal scale.
- To revert to raw mean squared error, pass `loss_type="mse"` when constructing the loss function or when calling convenience scripts (for example, `main(loss="mse")` in the experiment drivers).
- The metric helpers exposed via `loudspeaker.metrics` include `mse`, `mae`, `nrmse`, and the new `norm_mse`, so you can reuse the same normalization logic for evaluation.

## 🧪 Development

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing --cov-report=html

# Run specific test
pytest tests/test_models/
```

### Code Quality

```bash
# Format code
black src tests scripts && isort src tests scripts

# Type checking
mypy src

# Linting
flake8 src tests scripts
```

### Adding New Features

1. Create new modules in appropriate directories under `src/loudspeaker/`
2. Add tests in `tests/`
3. Update documentation
4. Run the full test suite

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Commands

```bash
# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Format code
black src tests scripts && isort src tests scripts

# Type checking
mypy src

# Lint code
flake8 src tests scripts

# Clean build artifacts
rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage

# Build package
python -m build

# Uninstall package
pip uninstall loudspeaker-py
```

## 📊 Performance Comparison

The framework supports comparing JAX/diffrax and PyTorch implementations:

| Aspect | JAX/Diffrax | PyTorch |
|--------|-------------|---------|
| **Speed** | ⭐⭐⭐⭐⭐ Very fast | ⭐⭐⭐⭐ Fast |
| **Memory** | ⭐⭐⭐⭐ Efficient | ⭐⭐⭐ Good |
| **Ease of use** | ⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Easy |
| **Ecosystem** | ⭐⭐⭐ Growing | ⭐⭐⭐⭐⭐ Mature |
| **Auto-diff** | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very good |

## 🎯 Applications

This framework is particularly well-suited for:

- **Loudspeaker modeling** and system identification
- **Time series classification** and regression
- **Dynamical system identification**
- **Control system design**
- **Signal processing applications**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Diffrax](https://github.com/patrick-kidger/diffrax) - Numerical differential equation solvers in JAX
- [Equinox](https://github.com/patrick-kidger/equinox) - Neural networks in JAX
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [JAX](https://jax.readthedocs.io/) - NumPy on accelerators

## 🔗 Related Projects

- [Neural CDE Paper](https://arxiv.org/abs/1810.01367) - Original Neural CDE paper
- [Diffrax Examples](https://docs.kidger.site/diffrax/) - Diffrax documentation and examples
- [PyTorch Examples](https://github.com/pytorch/examples) - PyTorch example implementations

---

**Note**: This is a research framework under active development. APIs may change as the project evolves.
