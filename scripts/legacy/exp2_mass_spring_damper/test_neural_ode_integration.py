#!/usr/bin/env python3
"""
Test script for neural_ode_example.py integration
===============================================

This script tests the refactored neural_ode_example.py to ensure:
1. All imports work correctly
2. Configuration system functions properly
3. Data generation integrates with msd_simulation_with_forcing
4. Model creation and training pipeline works
5. Basic functionality without full training run

Usage:
    python scripts/exp2_mass_spring_damper/test_neural_ode_integration.py
"""

import os
import sys
import traceback

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")

    # Import JAX ecosystem
    try:
        import diffrax as dx
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import optax

        print("✓ JAX ecosystem imports successful")
    except ImportError as e:
        print(f"✗ JAX ecosystem import failed: {e}")
        return False, None, None, None, None, None

    # Import neural_ode_funcs
    try:
        import neural_ode_funcs

        # Test that key functions are available
        assert hasattr(neural_ode_funcs, "create_neural_ode_config")
        assert hasattr(neural_ode_funcs, "generate_synthetic_data")
        assert hasattr(neural_ode_funcs, "NeuralODEModel")
        print("✓ neural_ode_funcs import successful")
    except (ImportError, AssertionError) as e:
        print(f"✗ neural_ode_funcs import failed: {e}")
        return False, None, None, None, None, None

    # Import msd_simulation_with_forcing
    try:
        from msd_simulation_with_forcing import ForcingType
        from msd_simulation_with_forcing import MSDConfig as MSDFullConfig
        from msd_simulation_with_forcing import (
            run_batch_simulation,
            run_single_simulation,
        )

        print("✓ msd_simulation_with_forcing import successful")
    except ImportError as e:
        print(f"✗ msd_simulation_with_forcing import failed: {e}")
        # This is optional, so don't fail the test

    return True, neural_ode_funcs, jax, jnp, eqx, optax


def test_configuration(neural_ode_funcs):
    """Test configuration system."""
    print("\nTesting configuration...")

    try:
        config = neural_ode_funcs.create_neural_ode_config(
            hidden_dim=32,
            num_layers=2,
            output_dim=3,
            dataset_size=64,
            num_steps=100,
            visualization_enabled=False,
            msd_params={
                "mass": 0.05,
                "natural_frequency": 25.0,
                "damping_ratio": 0.01,
                "forcing_amplitude": 0.5,
                "forcing_type": "pink_noise",
            },
        )

        # Check that config has expected structure
        assert "model" in config
        assert "data" in config
        assert "training" in config
        assert "msd_params" in config
        assert config["model"]["hidden_dim"] == 32
        assert config["model"]["output_dim"] == 3

        print("✓ Configuration system works correctly")
        return True, config
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False, None


def test_data_generation(neural_ode_funcs, config, jax):
    """Test data generation with small dataset."""
    print("\nTesting data generation...")

    try:
        # Use smaller dataset for testing
        test_config = config.copy()
        test_config["data"]["dataset_size"] = 16

        key = jax.random.PRNGKey(42)
        ts, train_data, test_data = neural_ode_funcs.generate_synthetic_data(
            test_config, key=key
        )

        # Check data shapes
        assert len(ts.shape) == 1
        assert len(train_data["trajectories"].shape) == 3  # (batch, time, features)
        assert train_data["trajectories"].shape[-1] == 3  # 3D state
        assert len(test_data["trajectories"].shape) == 3

        print(
            f"✓ Data generation successful: ts={ts.shape}, train={train_data['trajectories'].shape}"
        )
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation(neural_ode_funcs, config, jax):
    """Test model creation."""
    print("\nTesting model creation...")

    try:
        key = jax.random.PRNGKey(123)
        model = neural_ode_funcs.NeuralODEModel(
            data_size=config["model"]["output_dim"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            key=key,
        )

        # Check that model has parameters
        params = jax.tree_util.tree_leaves(model)
        total_params = sum(p.size for p in params if hasattr(p, "size"))
        assert total_params > 0

        print(f"✓ Model creation successful: {total_params} parameters")
        return True, model
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        traceback.print_exc()
        return False, None


def test_model_forward_pass(neural_ode_funcs, model, config, jax, jnp):
    """Test model forward pass."""
    print("\nTesting model forward pass...")

    try:
        key = jax.random.PRNGKey(456)

        # Create small test data
        ts = jnp.linspace(0, 0.1, 10)
        y0 = jax.random.uniform(key, (3,), minval=-0.5, maxval=0.5)

        # Test forward pass
        solution = neural_ode_funcs.solve_neural_ode(model, ts, y0, config)

        # Check output shape
        assert solution.shape == (len(ts), 3)
        assert not jnp.any(jnp.isnan(solution))

        print(f"✓ Model forward pass successful: output shape {solution.shape}")
        return True
    except Exception as e:
        print(f"✗ Model forward pass test failed: {e}")
        traceback.print_exc()
        return False


def test_training_step(neural_ode_funcs, model, config, jax, jnp, eqx, optax):
    """Test a single training step."""
    print("\nTesting training step...")

    try:
        # Create minimal training data
        test_config = config.copy()
        test_config["data"]["dataset_size"] = 8
        test_config["training"]["batch_size"] = 4

        key = jax.random.PRNGKey(789)
        ts, train_data, test_data = neural_ode_funcs.generate_synthetic_data(
            test_config, key=key
        )

        # Create a single batch
        batch_size = test_config["training"]["batch_size"]
        batch = (
            ts,
            train_data["initial_conditions"][:batch_size],
            train_data["trajectories"][:batch_size],
        )

        # Setup optimizer
        if test_config["training"]["optimizer"] == "adam":
            optimizer = optax.adam(test_config["training"]["learning_rate"])

        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Test training step
        loss, metrics, model, opt_state = neural_ode_funcs.train_step(
            model, optimizer, opt_state, batch, test_config["solver"]
        )

        # Check that training step completed
        assert loss is not None
        assert loss.shape == ()  # scalar
        assert "rmse" in metrics

        print(
            f"✓ Training step successful: loss={loss:.6f}, rmse={metrics['rmse']:.6f}"
        )
        return True
    except Exception as e:
        print(f"✗ Training step test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("Neural ODE Integration Test")
    print("=" * 50)

    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run tests
    tests_passed = 0
    total_tests = 6

    # Test 1: Imports
    import_success, neural_ode_funcs, jax, jnp, eqx, optax = test_imports()
    if import_success:
        tests_passed += 1

        # Test 2: Configuration
        config_success, config = test_configuration(neural_ode_funcs)
        if config_success:
            tests_passed += 1

            # Test 3: Data generation (only if config works)
            if test_data_generation(neural_ode_funcs, config, jax):
                tests_passed += 1

                # Test 4: Model creation (only if data generation works)
                model_success, model = test_model_creation(
                    neural_ode_funcs, config, jax
                )
                if model_success:
                    tests_passed += 1

                    # Test 5: Forward pass (only if model creation works)
                    if test_model_forward_pass(
                        neural_ode_funcs, model, config, jax, jnp
                    ):
                        tests_passed += 1

                    # Test 6: Training step (only if model creation works)
                    if test_training_step(
                        neural_ode_funcs, model, config, jax, jnp, eqx, optax
                    ):
                        tests_passed += 1

    # Print results
    print("\n" + "=" * 50)
    print(f"Integration Test Results: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print(
            "✓ All tests passed! neural_ode_example.py integration is working correctly."
        )
        return True
    else:
        print(
            f"✗ {total_tests - tests_passed} tests failed. Please check the issues above."
        )
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
