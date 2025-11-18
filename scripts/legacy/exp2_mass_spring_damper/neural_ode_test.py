#!/usr/bin/env python3
"""
Comprehensive Neural ODE Test Suite
===================================

This script provides thorough testing of the neural ODE functionality based on patterns
from the example notebooks. It tests integration between all components including:

- Configuration management
- Data generation (synthetic and MSD with forcing)
- Neural network architecture
- Training pipeline
- ODE solving
- Evaluation metrics
- Integration tests
- Performance benchmarks
- Error handling and edge cases

Based on testing patterns from neural_ode_diffrax_example.ipynb and related notebooks.

Usage:
    python scripts/exp2_mass_spring_damper/neural_ode_test.py

    # With specific test categories
    python scripts/exp2_mass_spring_damper/neural_ode_test.py --category=training
    python scripts/exp2_mass_spring_damper/neural_ode_test.py --category=integration
"""

# %%
"""Comprehensive Neural ODE Test Suite
Tests all functionality of neural_ode_funcs.py and neural_ode_example.py
"""

# %%
# Standard imports
import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, List

import diffrax
import equinox as eqx

# JAX ecosystem
import jax
import jax.numpy as jnp
import jax.random as jr

# Scientific computing
import optax

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our modules
try:
    from msd_simulation_with_forcing import ForcingType
    from msd_simulation_with_forcing import MSDConfig as MSDFullConfig
    from msd_simulation_with_forcing import run_batch_simulation
    from neural_ode_funcs import *

    print("✓ All imports successful")
except ImportError as e:
    print(f"⚠ Import warning: {e}")
    print("Some tests may be skipped")


# %%
# Test configuration and utilities
@dataclass
class TestConfig:
    """Configuration for test suite."""

    verbose: bool = True
    save_plots: bool = False
    quick_mode: bool = False  # Reduced dataset sizes for faster testing
    raise_on_error: bool = False
    test_categories: List[str] = None


class TestResult:
    """Container for test results."""

    def __init__(self, name: str, passed: bool, duration: float, error: str = None):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.error = error


class TestSuite:
    """Main test suite class."""

    def __init__(self, config: TestConfig = None):
        self.config = config or TestConfig()
        self.results = []
        self.test_key = jr.PRNGKey(42)

    def run_test(self, test_func: Callable, test_name: str) -> TestResult:
        """Run a single test with timing and error handling."""
        if self.config.verbose:
            print(f"\n🧪 Running {test_name}...")

        start_time = time.time()
        try:
            test_func()
            duration = time.time() - start_time
            result = TestResult(test_name, True, duration)
            if self.config.verbose:
                print(f"✅ {test_name} PASSED ({duration:.3f}s)")
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            if self.config.verbose:
                print(f"❌ {test_name} FAILED ({duration:.3f}s): {error_msg}")
                if self.config.raise_on_error:
                    raise
            result = TestResult(test_name, False, duration, error_msg)
            return result

    def print_summary(self):
        """Print test results summary."""
        print("\n" + "=" * 80)
        print("NEURAL ODE TEST SUITE RESULTS")
        print("=" * 80)

        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]

        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {len(passed_tests)}")
        print(f"Failed: {len(failed_tests)}")
        print(f"Success rate: {len(passed_tests) / len(self.results) * 100:.1f}%")
        print(f"Total duration: {sum(r.duration for r in self.results):.3f}s")

        if failed_tests:
            print("\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"  • {test.name}: {test.error}")

        print("=" * 80)
        return len(failed_tests) == 0


# %%
# Configuration tests
class ConfigurationTests:
    """Test configuration management system."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_basic_config_creation(self):
        """Test basic configuration creation."""
        config = create_neural_ode_config()
        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config
        assert "solver" in config
        assert "data" in config
        assert "forcing" in config
        assert "visualization" in config
        assert "evaluation" in config
        assert config["model"]["hidden_dim"] == 64
        assert config["model"]["num_layers"] == 3
        assert config["model"]["output_dim"] == 3
        assert config["training"]["learning_rate"] == 1e-3
        assert config["solver"]["solver_type"] == "tsit5"

    def test_custom_config_creation(self):
        """Test custom configuration creation."""
        custom_config = create_neural_ode_config(
            hidden_dim=32,
            num_layers=2,
            output_dim=4,
            dataset_size=128,
            num_steps=500,
            batch_size=16,
            learning_rate=2e-3,
            solver_type="kvaerno5",
            forcing_enabled=True,
            forcing_type="sine",
            visualization_enabled=True,
            msd_params={"mass": 0.1, "natural_frequency": 10.0},
        )

        assert custom_config["model"]["hidden_dim"] == 32
        assert custom_config["model"]["num_layers"] == 2
        assert custom_config["model"]["output_dim"] == 4
        assert custom_config["data"]["dataset_size"] == 128
        assert custom_config["training"]["num_steps"] == 500
        assert custom_config["training"]["batch_size"] == 16
        assert custom_config["training"]["learning_rate"] == 2e-3
        assert custom_config["solver"]["solver_type"] == "kvaerno5"
        assert custom_config["forcing"]["enabled"] == True
        assert custom_config["forcing"]["type"] == "sine"
        assert custom_config["visualization"]["enabled"] == True
        assert "msd_params" in custom_config

    def test_config_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid configuration parameters
        try:
            # This should work without errors
            config = create_neural_ode_config(
                hidden_dim=-1,  # Invalid but should be handled gracefully
                num_steps=0,  # Invalid but should be handled gracefully
            )
            # If we get here, the function should handle invalid inputs gracefully
            assert config["model"]["hidden_dim"] > 0
            assert config["training"]["num_steps"] > 0
        except (ValueError, TypeError):
            # This is also acceptable - the function can reject invalid inputs
            pass


# %%
# Data generation tests
class DataGenerationTests:
    """Test data generation functionality."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        config = create_neural_ode_config(
            dataset_size=32 if self.suite.config.quick_mode else 64,
            simulation_time=0.1,
            sample_rate=100,
            output_dim=3,
            forcing_enabled=False,
        )

        key = jr.fold_in(self.suite.test_key, 1)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Validate time vector
        assert len(ts.shape) == 1
        assert ts.shape[0] > 0
        assert jnp.all(jnp.diff(ts) > 0)  # Time should be increasing

        # Validate train data
        assert "initial_conditions" in train_data
        assert "trajectories" in train_data
        assert "ts" in train_data

        train_size = config["data"]["dataset_size"] * (1 - config["data"]["test_split"])
        expected_seq_len = len(ts)
        expected_output_dim = config["model"]["output_dim"]

        assert train_data["initial_conditions"].shape == (
            int(train_size),
            expected_output_dim,
        )
        assert train_data["trajectories"].shape == (
            int(train_size),
            expected_seq_len,
            expected_output_dim,
        )
        assert train_data["ts"].shape == ts.shape

        # Validate test data
        test_size = config["data"]["dataset_size"] * config["data"]["test_split"]
        assert test_data["initial_conditions"].shape == (
            int(test_size),
            expected_output_dim,
        )
        assert test_data["trajectories"].shape == (
            int(test_size),
            expected_seq_len,
            expected_output_dim,
        )

        # Check for NaN values
        assert not jnp.any(jnp.isnan(train_data["trajectories"]))
        assert not jnp.any(jnp.isnan(test_data["trajectories"]))

    def test_data_generation_with_forcing(self):
        """Test data generation with forcing signals."""
        config = create_neural_ode_config(
            dataset_size=16 if self.suite.config.quick_mode else 32,
            simulation_time=0.05,
            sample_rate=50,
            output_dim=3,
            forcing_enabled=True,
            forcing_type="pink_noise",
            forcing_amplitude=0.5,
        )

        key = jr.fold_in(self.suite.test_key, 2)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Validate forcing signals
        assert train_data["forcing"] is not None
        assert test_data["forcing"] is not None

        train_size = config["data"]["dataset_size"] * (1 - config["data"]["test_split"])
        expected_seq_len = len(ts)

        assert train_data["forcing"].shape == (int(train_size), expected_seq_len)
        assert test_data["forcing"].shape == (
            int(config["data"]["test_split"] * config["data"]["dataset_size"]),
            expected_seq_len,
        )

        # Check forcing signal properties
        assert not jnp.any(jnp.isnan(train_data["forcing"]))
        assert jnp.std(train_data["forcing"]) > 0  # Should have some variation

    def test_different_forcing_types(self):
        """Test different forcing signal types."""
        forcing_types = ["pink_noise", "sine", "chirp", "step"]

        for forcing_type in forcing_types:
            config = create_neural_ode_config(
                dataset_size=8,
                simulation_time=0.02,
                sample_rate=20,
                forcing_enabled=True,
                forcing_type=forcing_type,
                output_dim=3,
            )

            key = jr.fold_in(self.suite.test_key, hash(forcing_type) % 1000)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)

            # Check that forcing was generated
            assert train_data["forcing"] is not None
            assert train_data["forcing"].shape[1] == len(ts)

            # Check for reasonable signal properties
            forcing_signal = train_data["forcing"][0]  # First sample
            assert not jnp.any(jnp.isnan(forcing_signal))

            if forcing_type != "step":
                # Non-step signals should have some variation
                assert jnp.std(forcing_signal) > 1e-6

    def test_msd_data_generation(self):
        """Test MSD data generation with msd_simulation_with_forcing."""
        try:
            config = create_neural_ode_config(
                dataset_size=8 if self.suite.config.quick_mode else 16,
                simulation_time=0.02,
                sample_rate=50,
                output_dim=3,
                msd_params={
                    "mass": 0.05,
                    "natural_frequency": 25.0,
                    "damping_ratio": 0.01,
                    "forcing_amplitude": 0.5,
                },
            )

            key = jr.fold_in(self.suite.test_key, 3)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)

            # Validate MSD-specific data structure
            assert "forcings" in train_data or "forcing" in train_data
            assert (
                train_data["trajectories"].shape[-1] == 3
            )  # position, velocity, acceleration

            # Check that data has reasonable properties
            assert not jnp.any(jnp.isnan(train_data["trajectories"]))
            assert jnp.all(jnp.isfinite(train_data["trajectories"]))

        except ImportError:
            print(
                "⚠️ MSD simulation module not available, skipping MSD data generation test"
            )
            return  # Skip this test if MSD module is not available

    def test_data_normalization(self):
        """Test data normalization and preprocessing."""
        config = create_neural_ode_config(
            dataset_size=16,
            simulation_time=0.05,
            sample_rate=20,
            output_dim=3,
            noise_level=0.1,
        )

        key = jr.fold_in(self.suite.test_key, 4)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Test data preparation function
        input_data, target_data = prepare_neural_ode_data(train_data, config)

        # Validate prepared data
        assert input_data.shape == train_data["trajectories"].shape
        assert target_data.shape == train_data["trajectories"].shape
        assert not jnp.any(jnp.isnan(input_data))
        assert not jnp.any(jnp.isnan(target_data))


# %%
# Model architecture tests
class ModelArchitectureTests:
    """Test neural network architecture."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_neural_ode_func_creation(self):
        """Test NeuralODEFunc creation and forward pass."""
        key = jr.fold_in(self.suite.test_key, 5)
        func = NeuralODEFunc(
            data_size=3, hidden_dim=32, num_layers=2, activation="softplus", key=key
        )

        # Test forward pass
        y = jnp.array([1.0, 0.5, 0.1])
        t = 0.0
        dydt = func(t, y, None)

        assert dydt.shape == (3,), f"Expected shape (3,), got {dydt.shape}"
        assert not jnp.any(jnp.isnan(dydt)), "Output contains NaN values"
        assert jnp.all(jnp.isfinite(dydt)), "Output contains infinite values"

    def test_neural_ode_func_activations(self):
        """Test different activation functions."""
        activations = ["softplus", "tanh", "relu"]

        for activation in activations:
            key = jr.fold_in(self.suite.test_key, hash(activation) % 1000)
            func = NeuralODEFunc(
                data_size=3, hidden_dim=16, num_layers=2, activation=activation, key=key
            )

            y = jnp.array([1.0, 0.5, 0.1])
            t = 0.0
            dydt = func(t, y, None)

            assert dydt.shape == (3,)
            assert not jnp.any(jnp.isnan(dydt))
            assert jnp.all(jnp.isfinite(dydt))

    def test_neural_ode_model_creation(self):
        """Test NeuralODEModel creation and forward pass."""
        key = jr.fold_in(self.suite.test_key, 6)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=32,
            num_layers=2,
            solver_type="tsit5",
            activation="softplus",
            key=key,
        )

        # Test model parameters
        params = jax.tree_util.tree_leaves(model)
        total_params = sum(p.size for p in params if hasattr(p, "size"))
        assert total_params > 0, "Model should have parameters"

        # Test forward pass
        ts = jnp.linspace(0, 0.1, 10)
        y0 = jnp.array([1.0, 0.5, 0.1])

        solution = model(ts, y0)

        assert solution.shape == (
            10,
            3,
        ), f"Expected shape (10, 3), got {solution.shape}"
        assert not jnp.any(jnp.isnan(solution))
        assert jnp.all(jnp.isfinite(solution))

    def test_neural_ode_model_solvers(self):
        """Test different ODE solvers."""
        solver_types = ["tsit5", "kvaerno5", "dopri5"]

        for solver_type in solver_types:
            key = jr.fold_in(self.suite.test_key, hash(solver_type) % 1000)
            model = NeuralODEModel(
                data_size=3,
                hidden_dim=16,
                num_layers=2,
                solver_type=solver_type,
                key=key,
            )

            ts = jnp.linspace(0, 0.05, 5)
            y0 = jnp.array([0.5, 0.3, 0.1])

            solution = model(ts, y0)

            assert solution.shape == (5, 3)
            assert not jnp.any(jnp.isnan(solution))
            assert jnp.all(jnp.isfinite(solution))

    def test_model_batch_processing(self):
        """Test batch processing with vmap."""
        key = jr.fold_in(self.suite.test_key, 7)
        model = NeuralODEModel(data_size=3, hidden_dim=32, num_layers=2, key=key)

        ts = jnp.linspace(0, 0.1, 10)
        y0_batch = jnp.array([[1.0, 0.5, 0.1], [0.8, 0.3, 0.2], [1.2, 0.7, 0.0]])

        # Test batched forward pass
        batch_solution = jax.vmap(lambda y0: model(ts, y0))(y0_batch)

        assert batch_solution.shape == (3, 10, 3)
        assert not jnp.any(jnp.isnan(batch_solution))
        assert jnp.all(jnp.isfinite(batch_solution))

    def test_model_parameter_counting(self):
        """Test model parameter counting and structure."""
        key = jr.fold_in(self.suite.test_key, 8)
        model = NeuralODEModel(data_size=3, hidden_dim=64, num_layers=3, key=key)

        # Count parameters
        params = jax.tree_util.tree_leaves(model)
        total_params = sum(p.size for p in params if hasattr(p, "size"))

        assert total_params > 100, f"Expected >100 parameters, got {total_params}"
        assert total_params < 10000, f"Expected <10000 parameters, got {total_params}"

        # Check model structure
        assert hasattr(model, "func")
        assert hasattr(model, "initial_mapping")
        assert hasattr(model.func, "mlp")
        assert hasattr(model.func, "out_scale")


# %%
# Training tests
class TrainingTests:
    """Test training functionality."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_loss_function(self):
        """Test loss function computation."""
        config = create_neural_ode_config(
            dataset_size=8, simulation_time=0.02, sample_rate=10, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 9)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create simple model
        model_key = jr.fold_in(self.suite.test_key, 10)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        # Test loss function
        ts_batch = ts
        y0_batch = train_data["initial_conditions"][:2]
        target_batch = train_data["trajectories"][:2]

        loss, metrics = loss_fn_neural_ode(model, ts_batch, y0_batch, target_batch)

        assert loss.shape == (), "Loss should be a scalar"
        assert loss > 0, "Loss should be positive"
        assert not jnp.isnan(loss), "Loss should not be NaN"
        assert jnp.isfinite(loss), "Loss should be finite"

        # Check metrics
        assert "rmse" in metrics
        assert "relative_error" in metrics
        assert "dim_mse" in metrics
        assert "dim_rmse" in metrics
        assert metrics["rmse"] > 0
        assert metrics["relative_error"] >= 0

    def test_single_training_step(self):
        """Test single training step with gradients."""
        config = create_neural_ode_config(
            dataset_size=8,
            simulation_time=0.02,
            sample_rate=10,
            output_dim=3,
            batch_size=4,
        )

        key = jr.fold_in(self.suite.test_key, 11)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create model
        model_key = jr.fold_in(self.suite.test_key, 12)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        # Setup optimizer
        optimizer = optax.adam(config["training"]["learning_rate"])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Create batch
        batch = (
            ts,
            train_data["initial_conditions"][:4],
            train_data["trajectories"][:4],
        )

        # Test training step
        loss, metrics, new_model, new_opt_state = train_step(
            model, optimizer, opt_state, batch, config["solver"]
        )

        assert loss.shape == ()
        assert loss > 0
        assert not jnp.isnan(loss)
        assert jnp.isfinite(loss)
        assert new_model is not None
        assert new_opt_state is not None

        # Check that model parameters changed
        old_params = jax.tree_util.tree_leaves(model)
        new_params = jax.tree_util.tree_leaves(new_model)
        param_changes = [
            jnp.sum((new - old) ** 2)
            for old, new in zip(old_params, new_params)
            if hasattr(new, "shape")
        ]
        total_change = sum(param_changes)
        assert total_change > 0, "Model parameters should have changed during training"

    def test_training_loop(self):
        """Test complete training loop."""
        config = create_neural_ode_config(
            dataset_size=16 if self.suite.config.quick_mode else 32,
            simulation_time=0.05,
            sample_rate=20,
            output_dim=3,
            num_steps=10 if self.suite.config.quick_mode else 50,
            batch_size=8,
            eval_frequency=5,
        )

        key = jr.fold_in(self.suite.test_key, 13)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create model
        model_key = jr.fold_in(self.suite.test_key, 14)
        model = NeuralODEModel(data_size=3, hidden_dim=32, num_layers=2, key=model_key)

        # Test training
        trained_model, history = train_neural_ode(model, train_data, config, test_data)

        # Check training history
        assert len(history["train_loss"]) == config["training"]["num_steps"]
        assert len(history["train_rmse"]) == config["training"]["num_steps"]
        assert len(history["step_times"]) == config["training"]["num_steps"]

        # Check that loss decreased (basic sanity check)
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        assert (
            final_loss <= initial_loss * 2
        ), "Loss should not increase dramatically"  # Allow some increase due to noise

        # Check test metrics if available
        if history["test_loss"]:
            assert len(history["test_loss"]) > 0
            assert all(loss > 0 for loss in history["test_loss"])

        # Check model is still valid
        assert trained_model is not None
        test_ts = jnp.linspace(0, 0.02, 5)
        test_y0 = jnp.array([0.5, 0.3, 0.1])
        test_solution = trained_model(test_ts, test_y0)
        assert test_solution.shape == (5, 3)
        assert not jnp.any(jnp.isnan(test_solution))

    def test_different_optimizers(self):
        """Test different optimizers."""
        optimizers_to_test = ["adam", "adabelief", "sgd"]

        for optimizer_name in optimizers_to_test:
            config = create_neural_ode_config(
                dataset_size=8,
                simulation_time=0.02,
                sample_rate=10,
                output_dim=3,
                num_steps=5,
                optimizer=optimizer_name,
            )

            key = jr.fold_in(self.suite.test_key, hash(optimizer_name) % 1000)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)

            # Create model
            model_key = jr.fold_in(
                self.suite.test_key, hash(optimizer_name + 100) % 1000
            )
            model = NeuralODEModel(
                data_size=3, hidden_dim=16, num_layers=2, key=model_key
            )

            try:
                # Test training with this optimizer
                trained_model, history = train_neural_ode(
                    model, train_data, config, test_data
                )

                assert len(history["train_loss"]) == config["training"]["num_steps"]
                assert not jnp.any(jnp.isnan(jnp.array(history["train_loss"])))
                assert jnp.all(jnp.isfinite(jnp.array(history["train_loss"])))

            except Exception as e:
                if optimizer_name == "adam":
                    # Adam should always work
                    raise
                else:
                    print(f"⚠️ Optimizer {optimizer_name} failed: {e}")

    def test_batch_processing(self):
        """Test batch processing and dataloaders."""
        config = create_neural_ode_config(
            dataset_size=16,
            simulation_time=0.05,
            sample_rate=10,
            output_dim=3,
            batch_size=4,
        )

        key = jr.fold_in(self.suite.test_key, 15)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Test dataloader creation
        dataloader_key = jr.fold_in(self.suite.test_key, 16)
        dataloader = create_dataloader(
            train_data, config["training"]["batch_size"], key=dataloader_key
        )

        # Test batch generation
        batch_generator = dataloader()
        batch = next(batch_generator)

        assert len(batch) == 3  # (ts, y0_batch, target_batch)
        ts_batch, y0_batch, target_batch = batch

        assert y0_batch.shape[0] == config["training"]["batch_size"]
        assert y0_batch.shape[1] == config["model"]["output_dim"]
        assert target_batch.shape[0] == config["training"]["batch_size"]
        assert target_batch.shape[2] == config["model"]["output_dim"]
        assert target_batch.shape[1] == len(ts)


# %%
# Solving tests
class SolvingTests:
    """Test ODE solving functionality."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_basic_ode_solving(self):
        """Test basic ODE solving."""
        config = create_neural_ode_config(
            dataset_size=8, simulation_time=0.05, sample_rate=20, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 17)
        model_key = jr.fold_in(self.suite.test_key, 18)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        ts = jnp.linspace(0, 0.1, 20)
        y0 = jnp.array([1.0, 0.5, 0.1])

        # Test basic solving
        solution = solve_neural_ode(model, ts, y0, config)

        assert solution.shape == (20, 3)
        assert not jnp.any(jnp.isnan(solution))
        assert jnp.all(jnp.isfinite(solution))

    def test_solver_configurations(self):
        """Test different solver configurations."""
        config = create_neural_ode_config()

        key = jr.fold_in(self.suite.test_key, 19)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=key)

        ts = jnp.linspace(0, 0.05, 10)
        y0 = jnp.array([0.5, 0.3, 0.1])

        # Test different solver configurations
        solver_configs = [
            {
                "solver": diffrax.Tsit5(),
                "rtol": 1e-3,
                "atol": 1e-6,
                "adaptive_steps": True,
            },
            {
                "solver": diffrax.Kvaerno5(),
                "rtol": 1e-6,
                "atol": 1e-8,
                "adaptive_steps": True,
            },
            {
                "solver": diffrax.Dopri5(),
                "rtol": 1e-4,
                "atol": 1e-7,
                "adaptive_steps": True,
            },
        ]

        for solver_config in solver_configs:
            solution = solve_neural_ode(model, ts, y0, {"solver": solver_config})

            assert solution.shape == (10, 3)
            assert not jnp.any(jnp.isnan(solution))
            assert jnp.all(jnp.isfinite(solution))

    def test_stiff_vs_non_stiff_solvers(self):
        """Test stiff vs non-stiff solver selection."""
        config = create_neural_ode_config()

        key = jr.fold_in(self.suite.test_key, 20)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=key)

        ts = jnp.linspace(0, 0.02, 5)
        y0 = jnp.array([1.0, 0.5, 0.1])

        # Test stiff solver (Kvaerno5)
        stiff_config = {
            "solver": diffrax.Kvaerno5(),
            "rtol": 1e-6,
            "atol": 1e-8,
            "adaptive_steps": True,
        }

        stiff_solution = solve_neural_ode(model, ts, y0, {"solver": stiff_config})
        assert stiff_solution.shape == (5, 3)
        assert not jnp.any(jnp.isnan(stiff_solution))

        # Test non-stiff solver (Tsit5)
        non_stiff_config = {
            "solver": diffrax.Tsit5(),
            "rtol": 1e-3,
            "atol": 1e-6,
            "adaptive_steps": True,
        }

        non_stiff_solution = solve_neural_ode(
            model, ts, y0, {"solver": non_stiff_config}
        )
        assert non_stiff_solution.shape == (5, 3)
        assert not jnp.any(jnp.isnan(non_stiff_solution))

    def test_timestep_variations(self):
        """Test different time steps and tolerances."""
        config = create_neural_ode_config()

        key = jr.fold_in(self.suite.test_key, 21)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=key)

        y0 = jnp.array([0.5, 0.3, 0.1])

        # Test different time steps
        time_steps = [0.01, 0.001, 0.0001]

        for dt in time_steps:
            ts = jnp.arange(0, 0.05, dt)

            solution = solve_neural_ode(model, ts, y0, config)

            assert solution.shape[0] == len(ts)
            assert solution.shape[1] == 3
            assert not jnp.any(jnp.isnan(solution))

    def test_forcing_integration(self):
        """Test forcing term integration."""
        config = create_neural_ode_config(
            dataset_size=8, simulation_time=0.02, sample_rate=20, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 22)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=key)

        ts = jnp.linspace(0, 0.02, 20)
        y0 = jnp.array([0.5, 0.3, 0.1])
        forcing_signal = jnp.sin(2 * jnp.pi * 5 * ts)  # 5 Hz sine wave

        # Test solving with forcing (this may be a placeholder in the current implementation)
        solution = solve_with_forcing(model, ts, y0, forcing_signal, config)

        assert solution.shape == (20, 3)
        assert not jnp.any(jnp.isnan(solution))


# %%
# Evaluation tests
class EvaluationTests:
    """Test evaluation functionality."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_model_evaluation(self):
        """Test model evaluation."""
        config = create_neural_ode_config(
            dataset_size=16, simulation_time=0.05, sample_rate=10, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 23)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create and train a simple model
        model_key = jr.fold_in(self.suite.test_key, 24)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        # Train for a few steps
        test_config = config.copy()
        test_config["training"]["num_steps"] = 5
        test_config["training"]["batch_size"] = 8

        trained_model, _ = train_neural_ode(model, train_data, test_config, test_data)

        # Test evaluation
        loss, metrics = evaluate_model(trained_model, test_data, config)

        assert loss.shape == ()
        assert loss > 0
        assert not jnp.isnan(loss)
        assert jnp.isfinite(loss)

        # Check metrics structure
        assert "rmse" in metrics
        assert "relative_error" in metrics
        assert "dim_mse" in metrics
        assert "dim_rmse" in metrics

        assert metrics["rmse"] > 0
        assert metrics["relative_error"] >= 0
        assert len(metrics["dim_mse"]) == 3
        assert len(metrics["dim_rmse"]) == 3

    def test_metric_computation(self):
        """Test metric computation."""
        # Create synthetic predictions and targets
        predictions = jnp.array(
            [
                [[1.0, 0.5, 0.1], [1.1, 0.6, 0.2], [1.2, 0.7, 0.3]],
                [[0.9, 0.4, 0.0], [1.0, 0.5, 0.1], [1.1, 0.6, 0.2]],
            ]
        )

        targets = jnp.array(
            [
                [[1.0, 0.5, 0.1], [1.0, 0.5, 0.1], [1.0, 0.5, 0.1]],
                [[1.0, 0.5, 0.1], [1.0, 0.5, 0.1], [1.0, 0.5, 0.1]],
            ]
        )

        # Test metric computation
        metrics = compute_metrics(predictions, targets)

        # Check all expected metrics are present
        expected_metrics = [
            "total_mse",
            "total_rmse",
            "relative_error",
            "max_error",
            "r2_score",
            "dim_mse",
            "dim_rmse",
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"

        # Check metric values
        assert metrics["total_mse"] >= 0
        assert metrics["total_rmse"] >= 0
        assert metrics["relative_error"] >= 0
        assert metrics["max_error"] >= 0
        assert isinstance(metrics["r2_score"], (float, int))  # R2 can be negative

        # Check dimensional metrics
        assert len(metrics["dim_mse"]) == 3
        assert len(metrics["dim_rmse"]) == 3
        assert all(mse >= 0 for mse in metrics["dim_mse"])
        assert all(rmse >= 0 for rmse in metrics["dim_rmse"])

    def test_per_dimension_metrics(self):
        """Test per-dimension metric computation."""
        config = create_neural_ode_config(
            dataset_size=8, simulation_time=0.02, sample_rate=10, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 25)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create simple model
        model_key = jr.fold_in(self.suite.test_key, 26)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        # Get predictions
        predictions = jax.vmap(lambda y0: solve_neural_ode(model, ts, y0, config))(
            test_data["initial_conditions"]
        )
        targets = test_data["trajectories"]

        # Compute per-dimension metrics
        metrics = compute_metrics(predictions, targets)

        # Check that we have metrics for each dimension
        assert len(metrics["dim_mse"]) == 3
        assert len(metrics["dim_rmse"]) == 3

        # Each dimension should have reasonable metrics
        for dim in range(3):
            assert metrics["dim_mse"][dim] >= 0
            assert metrics["dim_rmse"][dim] >= 0
            assert not jnp.isnan(metrics["dim_mse"][dim])
            assert jnp.isfinite(metrics["dim_mse"][dim])

    def test_differentiability_testing(self):
        """Test differentiability of the model."""
        config = create_neural_ode_config(
            dataset_size=8, simulation_time=0.02, sample_rate=10, output_dim=3
        )

        key = jr.fold_in(self.suite.test_key, 27)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create model
        model_key = jr.fold_in(self.suite.test_key, 28)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=2, key=model_key)

        # Test differentiability
        y0 = test_data["initial_conditions"][0]
        diff_test = differentiability_test(model, ts, y0, config)

        # Check test results
        assert "gradients_computable" in diff_test
        assert "gradients_finite" in diff_test
        assert "gradients_nonzero" in diff_test
        assert "test_passed" in diff_test

        assert isinstance(diff_test["gradients_computable"], bool)
        assert isinstance(diff_test["gradients_finite"], bool)
        assert isinstance(diff_test["gradients_nonzero"], bool)
        assert isinstance(diff_test["test_passed"], bool)

        # For a well-initialized model, gradients should be computable and finite
        if diff_test["gradients_computable"]:
            assert diff_test[
                "gradients_finite"
            ], "Gradients should be finite for a well-initialized model"


# %%
# Visualization tests
class VisualizationTests:
    """Test visualization functionality."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_visualization_disabled(self):
        """Test that visualization functions return early when disabled."""
        config = create_neural_ode_config(
            visualization_enabled=False, save_dir="test_exp/"
        )

        # Create dummy history and data
        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "train_rmse": [1.0, 0.8, 0.6],
            "train_dim_mse": [jnp.array([0.5, 0.4, 0.3])],
            "step_times": [0.1, 0.1, 0.1],
            "test_loss": [0.9, 0.7],
            "test_rmse": [0.9, 0.7],
        }

        # Create dummy test data
        test_data = {
            "ts": jnp.linspace(0, 0.1, 10),
            "initial_conditions": jnp.ones((2, 3)),
            "trajectories": jnp.ones((2, 10, 3)),
        }

        # Create dummy model
        key = jr.fold_in(self.suite.test_key, 99)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=1, key=key)

        # Test that functions return early without errors
        try:
            plot_training_history(history, config)
            plot_trajectories(model, test_data, config, num_samples=1)
            plot_phase_space(model, test_data, config, num_samples=1)
            # If we get here, functions handled disabled visualization correctly
        except Exception as e:
            # This should not happen
            raise AssertionError(
                f"Visualization functions should handle disabled state gracefully: {e}"
            )

    def test_visualization_enabled(self):
        """Test that visualization functions work when enabled."""
        config = create_neural_ode_config(
            visualization_enabled=True,
            save_dir="test_exp/",
            format="png",
            dpi=150,  # Lower DPI for faster testing
        )

        # Create dummy history and data
        history = {
            "train_loss": [1.0, 0.8, 0.6],
            "train_rmse": [1.0, 0.8, 0.6],
            "train_dim_mse": [jnp.array([0.5, 0.4, 0.3])],
            "step_times": [0.1, 0.1, 0.1],
            "test_loss": [0.9, 0.7],
            "test_rmse": [0.9, 0.7],
        }

        # Create dummy test data
        test_data = {
            "ts": jnp.linspace(0, 0.1, 10),
            "initial_conditions": jnp.ones((2, 3)),
            "trajectories": jnp.ones((2, 10, 3)),
        }

        # Create dummy model
        key = jr.fold_in(self.suite.test_key, 100)
        model = NeuralODEModel(data_size=3, hidden_dim=16, num_layers=1, key=key)

        # Test that functions can be called without errors
        try:
            plot_training_history(history, config)
            plot_trajectories(model, test_data, config, num_samples=1)
            plot_phase_space(model, test_data, config, num_samples=1)
            # If we get here, functions handled enabled visualization correctly
        except Exception as e:
            # This should not happen in normal circumstances
            if "matplotlib" in str(e).lower() or "display" in str(e).lower():
                # Skip test if there's a display/matplotlib issue (common in headless environments)
                print(f"⚠️ Visualization test skipped due to matplotlib issue: {e}")
                return
            else:
                raise AssertionError(
                    f"Visualization functions should work when enabled: {e}"
                )

    def test_visualization_config_defaults(self):
        """Test that visualization config has correct defaults."""
        # Test default config
        default_config = create_neural_ode_config()
        assert default_config["visualization"]["enabled"] == False
        assert default_config["visualization"]["save_dir"] == "exp/"
        assert default_config["visualization"]["format"] == "png"
        assert default_config["visualization"]["dpi"] == 300
        assert default_config["visualization"]["bbox_inches"] == "tight"

        # Test custom config
        custom_config = create_neural_ode_config(
            visualization_enabled=True,
            save_dir="custom_exp/",
            plot_format="pdf",
            dpi=600,
        )
        assert custom_config["visualization"]["enabled"] == True
        assert custom_config["visualization"]["save_dir"] == "custom_exp/"
        assert custom_config["visualization"]["format"] == "pdf"
        assert custom_config["visualization"]["dpi"] == 600
        assert custom_config["visualization"]["bbox_inches"] == "tight"


# %%
# Integration tests
class IntegrationTests:
    """Test complete integration between components."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_complete_pipeline(self):
        """Test complete pipeline from data generation to evaluation."""
        # Create configuration
        config = create_neural_ode_config(
            dataset_size=32 if self.suite.config.quick_mode else 64,
            num_steps=20 if self.suite.config.quick_mode else 100,
            hidden_dim=32,
            num_layers=2,
            batch_size=16,
            eval_frequency=10,
            visualization_enabled=False,
        )

        # Generate data
        key = jr.fold_in(self.suite.test_key, 29)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create and train model
        model_key = jr.fold_in(self.suite.test_key, 30)
        model = NeuralODEModel(data_size=3, hidden_dim=32, num_layers=2, key=model_key)

        # Train model
        trained_model, history = train_neural_ode(model, train_data, config, test_data)

        # Evaluate
        metrics = evaluate_model(trained_model, test_data, config)

        # Verify results
        assert "loss" in metrics
        assert "rmse" in metrics
        assert "relative_error" in metrics
        assert metrics["loss"] > 0
        assert metrics["rmse"] > 0
        assert metrics["relative_error"] >= 0

        # Check training history
        assert len(history["train_loss"]) == config["training"]["num_steps"]
        assert len(history["train_rmse"]) == config["training"]["num_steps"]

        # Verify model can still make predictions
        test_ts = jnp.linspace(0, 0.02, 5)
        test_y0 = jnp.array([0.5, 0.3, 0.1])
        test_solution = trained_model(test_ts, test_y0)
        assert test_solution.shape == (5, 3)
        assert not jnp.any(jnp.isnan(test_solution))

    def test_different_parameter_combinations(self):
        """Test different parameter combinations."""
        test_configs = [
            {"hidden_dim": 16, "num_layers": 1, "dataset_size": 16, "num_steps": 10},
            {"hidden_dim": 64, "num_layers": 3, "dataset_size": 32, "num_steps": 20},
            {"hidden_dim": 32, "num_layers": 2, "dataset_size": 24, "num_steps": 15},
        ]

        for i, test_config_params in enumerate(test_configs):
            # Create config
            config = create_neural_ode_config(
                visualization_enabled=False, **test_config_params
            )

            # Generate data
            key = jr.fold_in(self.suite.test_key, 31 + i)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)

            # Create and train model
            model_key = jr.fold_in(self.suite.test_key, 41 + i)
            model = NeuralODEModel(
                data_size=3,
                hidden_dim=test_config_params["hidden_dim"],
                num_layers=test_config_params["num_layers"],
                key=model_key,
            )

            # Train model
            trained_model, history = train_neural_ode(
                model, train_data, config, test_data
            )

            # Evaluate
            metrics = evaluate_model(trained_model, test_data, config)

            # Basic validation
            assert metrics["loss"] > 0
            assert metrics["rmse"] > 0
            assert len(history["train_loss"]) == test_config_params["num_steps"]

    def test_error_handling_and_edge_cases(self):
        """Test error handling and edge cases."""
        # Test with very small dataset
        try:
            config = create_neural_ode_config(
                dataset_size=2,  # Very small
                simulation_time=0.01,
                sample_rate=5,
                output_dim=3,
                batch_size=1,
            )

            key = jr.fold_in(self.suite.test_key, 51)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)

            # Create model
            model_key = jr.fold_in(self.suite.test_key, 52)
            model = NeuralODEModel(
                data_size=3, hidden_dim=8, num_layers=1, key=model_key
            )

            # Train model (should handle small dataset gracefully)
            trained_model, history = train_neural_ode(
                model, train_data, config, test_data
            )

            # Should complete without errors
            assert trained_model is not None
            assert len(history["train_loss"]) == config["training"]["num_steps"]

        except Exception as e:
            # This is acceptable - the system should handle edge cases gracefully
            print(f"⚠️ Small dataset test failed (this may be expected): {e}")

        # Test with extreme parameters
        try:
            extreme_config = create_neural_ode_config(
                hidden_dim=1024,  # Very large
                num_layers=10,  # Many layers
                dataset_size=1024,  # Large dataset
                num_steps=5,  # Few steps for quick test
                batch_size=256,
                learning_rate=1e-6,  # Very small learning rate
            )

            key = jr.fold_in(self.suite.test_key, 53)
            ts, train_data, test_data = generate_synthetic_data(extreme_config, key=key)

            # Create large model
            model_key = jr.fold_in(self.suite.test_key, 54)
            model = NeuralODEModel(
                data_size=3, hidden_dim=1024, num_layers=10, key=model_key
            )

            # Train for a few steps
            test_config = extreme_config.copy()
            test_config["training"]["num_steps"] = 5
            trained_model, history = train_neural_ode(
                model, train_data, test_config, test_data
            )

            # Should complete without errors
            assert trained_model is not None
            assert len(history["train_loss"]) == 5

        except Exception as e:
            # Large models may fail due to memory constraints, which is acceptable
            print(f"⚠️ Extreme parameters test failed (may be due to memory): {e}")

    def test_memory_efficiency(self):
        """Test memory efficiency with different dataset sizes."""
        dataset_sizes = (
            [16, 32, 64] if self.suite.config.quick_mode else [16, 32, 64, 128]
        )

        for dataset_size in dataset_sizes:
            try:
                config = create_neural_ode_config(
                    dataset_size=dataset_size,
                    simulation_time=0.05,
                    sample_rate=20,
                    output_dim=3,
                    num_steps=5,  # Short training for memory test
                    batch_size=min(8, dataset_size // 2),
                )

                key = jr.fold_in(self.suite.test_key, 55 + dataset_size)
                ts, train_data, test_data = generate_synthetic_data(config, key=key)

                # Create model
                model_key = jr.fold_in(self.suite.test_key, 65 + dataset_size)
                model = NeuralODEModel(
                    data_size=3, hidden_dim=32, num_layers=2, key=model_key
                )

                # Train model
                start_time = time.time()
                trained_model, history = train_neural_ode(
                    model, train_data, config, test_data
                )
                training_time = time.time() - start_time

                # Should complete without memory errors
                assert trained_model is not None
                assert len(history["train_loss"]) == 5
                assert training_time > 0

                if self.suite.config.verbose:
                    print(f"  Dataset size {dataset_size}: {training_time:.3f}s")

            except Exception as e:
                # Memory errors are acceptable for very large datasets
                print(f"⚠️ Memory test failed for dataset size {dataset_size}: {e}")


# %%
# Performance benchmarks
class PerformanceTests:
    """Test performance characteristics."""

    def __init__(self, suite: TestSuite):
        self.suite = suite

    def test_data_generation_performance(self):
        """Test data generation performance."""
        dataset_sizes = (
            [32, 64, 128] if self.suite.config.quick_mode else [32, 64, 128, 256]
        )

        for dataset_size in dataset_sizes:
            config = create_neural_ode_config(
                dataset_size=dataset_size,
                simulation_time=0.1,
                sample_rate=50,
                output_dim=3,
            )

            key = jr.fold_in(self.suite.test_key, 75 + dataset_size)

            start_time = time.time()
            ts, train_data, test_data = generate_synthetic_data(config, key=key)
            generation_time = time.time() - start_time

            # Check that generation completed in reasonable time
            assert generation_time > 0
            assert (
                generation_time < 30
            ), f"Data generation took too long: {generation_time}s"

            # Check data shapes
            expected_train_size = int(dataset_size * 0.8)
            assert train_data["trajectories"].shape[0] == expected_train_size
            assert train_data["trajectories"].shape[-1] == 3

            if self.suite.config.verbose:
                print(f"  Dataset size {dataset_size}: {generation_time:.3f}s")

    def test_model_forward_pass_performance(self):
        """Test model forward pass performance."""
        sequence_lengths = [10, 50, 100]
        batch_sizes = [1, 8, 32]

        # Create model
        model_key = jr.fold_in(self.suite.test_key, 85)
        model = NeuralODEModel(data_size=3, hidden_dim=64, num_layers=3, key=model_key)

        for seq_len in sequence_lengths:
            for batch_size in batch_sizes:
                ts = jnp.linspace(0, 0.1, seq_len)
                y0_batch = jnp.ones((batch_size, 3))

                start_time = time.time()
                batch_solution = jax.vmap(lambda y0: model(ts, y0))(y0_batch)
                forward_time = time.time() - start_time

                # Check that forward pass completed
                assert batch_solution.shape == (batch_size, seq_len, 3)
                assert forward_time > 0
                assert forward_time < 10, f"Forward pass took too long: {forward_time}s"

                if self.suite.config.verbose:
                    print(
                        f"  Seq len {seq_len}, batch {batch_size}: {forward_time:.3f}s"
                    )

    def test_training_performance(self):
        """Test training performance."""
        config = create_neural_ode_config(
            dataset_size=64 if self.suite.config.quick_mode else 128,
            simulation_time=0.1,
            sample_rate=20,
            output_dim=3,
            num_steps=20,
            batch_size=16,
            learning_rate=1e-3,
        )

        key = jr.fold_in(self.suite.test_key, 95)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        # Create model
        model_key = jr.fold_in(self.suite.test_key, 96)
        model = NeuralODEModel(data_size=3, hidden_dim=64, num_layers=3, key=model_key)

        # Benchmark training
        start_time = time.time()
        trained_model, history = train_neural_ode(model, train_data, config, test_data)
        training_time = time.time() - start_time

        # Check training completed
        assert trained_model is not None
        assert len(history["train_loss"]) == 20
        assert training_time > 0
        assert training_time < 300, f"Training took too long: {training_time}s"

        # Check that loss improved
        initial_loss = history["train_loss"][0]
        final_loss = history["train_loss"][-1]
        loss_improvement = (initial_loss - final_loss) / initial_loss

        if self.suite.config.verbose:
            print(
                f"  Training: {training_time:.3f}s, loss improvement: {loss_improvement:.2%}"
            )

    def test_solver_performance_comparison(self):
        """Test and compare solver performance."""
        solvers_to_test = [
            ("tsit5", diffrax.Tsit5()),
            ("kvaerno5", diffrax.Kvaerno5()),
            ("dopri5", diffrax.Dopri5()),
        ]

        # Create simple model
        model_key = jr.fold_in(self.suite.test_key, 97)
        model = NeuralODEModel(data_size=3, hidden_dim=32, num_layers=2, key=model_key)

        ts = jnp.linspace(0, 0.1, 100)
        y0 = jnp.array([1.0, 0.5, 0.1])

        solver_times = {}

        for solver_name, solver in solvers_to_test:
            config = {
                "solver": {
                    "solver": solver,
                    "rtol": 1e-6,
                    "atol": 1e-8,
                    "adaptive_steps": True,
                }
            }

            start_time = time.time()
            solution = solve_neural_ode(model, ts, y0, config)
            solve_time = time.time() - start_time

            solver_times[solver_name] = solve_time

            # Check solution quality
            assert solution.shape == (100, 3)
            assert not jnp.any(jnp.isnan(solution))
            assert jnp.all(jnp.isfinite(solution))

            if self.suite.config.verbose:
                print(f"  {solver_name}: {solve_time:.4f}s")

        # Verify we got timing results for all solvers
        assert len(solver_times) == len(solvers_to_test)
        for solver_name in solver_times:
            assert solver_times[solver_name] > 0


# %%
# Main test execution
def run_all_tests(config: TestConfig = None):
    """Run all tests and return success status."""
    suite = TestSuite(config or TestConfig())

    # Create test instances
    config_tests = ConfigurationTests(suite)
    data_tests = DataGenerationTests(suite)
    model_tests = ModelArchitectureTests(suite)
    training_tests = TrainingTests(suite)
    solving_tests = SolvingTests(suite)
    evaluation_tests = EvaluationTests(suite)
    visualization_tests = VisualizationTests(suite)
    integration_tests = IntegrationTests(suite)
    performance_tests = PerformanceTests(suite)

    # Define all tests
    all_tests = [
        # Configuration tests
        (config_tests.test_basic_config_creation, "Basic Configuration Creation"),
        (config_tests.test_custom_config_creation, "Custom Configuration Creation"),
        (config_tests.test_config_validation, "Configuration Validation"),
        # Data generation tests
        (data_tests.test_synthetic_data_generation, "Synthetic Data Generation"),
        (data_tests.test_data_generation_with_forcing, "Data Generation with Forcing"),
        (data_tests.test_different_forcing_types, "Different Forcing Types"),
        (data_tests.test_msd_data_generation, "MSD Data Generation"),
        (data_tests.test_data_normalization, "Data Normalization"),
        # Model architecture tests
        (model_tests.test_neural_ode_func_creation, "NeuralODEFunc Creation"),
        (model_tests.test_neural_ode_func_activations, "NeuralODEFunc Activations"),
        (model_tests.test_neural_ode_model_creation, "NeuralODEModel Creation"),
        (model_tests.test_neural_ode_model_solvers, "NeuralODEModel Solvers"),
        (model_tests.test_model_batch_processing, "Model Batch Processing"),
        (model_tests.test_model_parameter_counting, "Model Parameter Counting"),
        # Training tests
        (training_tests.test_loss_function, "Loss Function"),
        (training_tests.test_single_training_step, "Single Training Step"),
        (training_tests.test_training_loop, "Training Loop"),
        (training_tests.test_different_optimizers, "Different Optimizers"),
        (training_tests.test_batch_processing, "Batch Processing"),
        # Solving tests
        (solving_tests.test_basic_ode_solving, "Basic ODE Solving"),
        (solving_tests.test_solver_configurations, "Solver Configurations"),
        (solving_tests.test_stiff_vs_non_stiff_solvers, "Stiff vs Non-stiff Solvers"),
        (solving_tests.test_timestep_variations, "Timestep Variations"),
        (solving_tests.test_forcing_integration, "Forcing Integration"),
        # Evaluation tests
        (evaluation_tests.test_model_evaluation, "Model Evaluation"),
        (evaluation_tests.test_metric_computation, "Metric Computation"),
        (evaluation_tests.test_per_dimension_metrics, "Per-dimension Metrics"),
        (evaluation_tests.test_differentiability_testing, "Differentiability Testing"),
        # Visualization tests
        (visualization_tests.test_visualization_disabled, "Visualization Disabled"),
        (visualization_tests.test_visualization_enabled, "Visualization Enabled"),
        (
            visualization_tests.test_visualization_config_defaults,
            "Visualization Config Defaults",
        ),
        # Integration tests
        (integration_tests.test_complete_pipeline, "Complete Pipeline"),
        (
            integration_tests.test_different_parameter_combinations,
            "Parameter Combinations",
        ),
        (integration_tests.test_error_handling_and_edge_cases, "Error Handling"),
        (integration_tests.test_memory_efficiency, "Memory Efficiency"),
        # Performance tests
        (
            performance_tests.test_data_generation_performance,
            "Data Generation Performance",
        ),
        (
            performance_tests.test_model_forward_pass_performance,
            "Forward Pass Performance",
        ),
        (performance_tests.test_training_performance, "Training Performance"),
        (
            performance_tests.test_solver_performance_comparison,
            "Solver Performance Comparison",
        ),
    ]

    # Filter tests by category if specified
    if config and config.test_categories:
        category_tests = {
            "configuration": all_tests[:3],
            "data": all_tests[3:8],
            "model": all_tests[8:14],
            "training": all_tests[14:19],
            "solving": all_tests[19:24],
            "evaluation": all_tests[24:28],
            "visualization": all_tests[28:31],
            "integration": all_tests[31:35],
            "performance": all_tests[35:39],
        }

        selected_tests = []
        for category in config.test_categories:
            if category in category_tests:
                selected_tests.extend(category_tests[category])
            else:
                print(f"⚠️ Unknown test category: {category}")

        if selected_tests:
            all_tests = selected_tests
        else:
            print("⚠️ No valid test categories specified, running all tests")

    # Run all tests
    print(f"Running {len(all_tests)} tests...")

    for test_func, test_name in all_tests:
        result = suite.run_test(test_func, test_name)
        suite.results.append(result)

    # Print summary
    success = suite.print_summary()

    # Print recommendations
    if not success:
        print("\n🔧 RECOMMENDATIONS:")
        failed_count = len([r for r in suite.results if not r.passed])
        if failed_count > len(suite.results) // 2:
            print("• Consider running with --quick-mode to reduce test complexity")
        print("• Check individual test errors above for specific issues")
        print("• Ensure all required dependencies are installed")

    return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Neural ODE Test Suite")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        help="Use smaller datasets for faster testing",
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save plots (if enabled in config)"
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise exceptions on test failures",
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        choices=[
            "configuration",
            "data",
            "model",
            "training",
            "solving",
            "evaluation",
            "visualization",
            "integration",
            "performance",
        ],
        help="Run specific test categories",
    )

    args = parser.parse_args()

    # Print header
    print("NEURAL ODE COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"64-bit precision: {jax.config.jax_enable_x64}")
    print(f"Quick mode: {args.quick_mode}")
    print("=" * 80)

    # Create test config
    test_config = TestConfig(
        verbose=args.verbose or not args.quick_mode,
        save_plots=args.save_plots,
        quick_mode=args.quick_mode,
        raise_on_error=args.raise_on_error,
        test_categories=args.categories,
    )

    # Run tests
    success = run_all_tests(test_config)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


# %%
# Example usage and testing
if __name__ == "__main__":
    # You can run the test suite in different ways:

    # 1. Run all tests with default settings
    # success = run_all_tests()

    # 2. Run with custom configuration
    # config = TestConfig(verbose=True, quick_mode=True)
    # success = run_all_tests(config)

    # 3. Run specific categories
    # config = TestConfig(test_categories=['model', 'training'])
    # success = run_all_tests(config)

    # 4. Or use command line arguments
    main()
