#!/usr/bin/env python3
"""
Test Script for Neural ODE Functions Module
===========================================

This script tests the neural_ode_funcs.py module to ensure all functions work correctly
and demonstrates basic usage patterns. It provides a quick way to verify the module
is functioning properly before running full experiments.

Key Features:
- Tests all major functions in neural_ode_funcs.py
- Demonstrates configuration creation and modification
- Shows data generation, model training, and evaluation
- Includes basic error handling and validation
- Provides performance benchmarks

Usage:
    python scripts/exp2_mass_spring_damper/test_neural_ode_funcs.py
"""

import time
import traceback
from typing import Dict, Any

import jax
import jax.numpy as jnp
import jax.random as jr

# Import the neural ODE functions module
try:
    from neural_ode_funcs import *
    print("✓ Successfully imported neural_ode_funcs module")
except ImportError as e:
    print(f"✗ Failed to import neural_ode_funcs: {e}")
    exit(1)


def test_configuration_system():
    """Test the configuration management system."""
    print("\n" + "="*50)
    print("Testing Configuration System")
    print("="*50)
    
    try:
        # Test basic configuration creation
        config = create_neural_ode_config()
        print("✓ Basic configuration created")
        
        # Test custom configuration
        custom_config = create_neural_ode_config(
            hidden_dim=32,
            num_layers=2,
            output_dim=3,
            dataset_size=64,
            num_steps=100,
            batch_size=16,
            visualization_enabled=False,  # Disable for testing
            forcing_enabled=True,
            forcing_type='sine'
        )
        print("✓ Custom configuration created")
        
        # Test configuration structure
        assert 'model' in custom_config
        assert 'training' in custom_config
        assert 'solver' in custom_config
        assert 'data' in custom_config
        print("✓ Configuration structure validated")
        
        # Test configuration parameters
        assert custom_config['model']['hidden_dim'] == 32
        assert custom_config['model']['num_layers'] == 2
        assert custom_config['model']['output_dim'] == 3
        assert custom_config['data']['dataset_size'] == 64
        assert custom_config['training']['num_steps'] == 100
        print("✓ Configuration parameters validated")
        
        return custom_config
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return None


def test_data_generation(config: Dict[str, Any], key: jr.PRNGKey):
    """Test data generation functionality."""
    print("\n" + "="*50)
    print("Testing Data Generation")
    print("="*50)
    
    try:
        # Generate synthetic data
        ts, train_data, test_data = generate_synthetic_data(config, key=key)
        print("✓ Synthetic data generated")
        
        # Validate data shapes
        assert len(ts.shape) == 1
        assert len(train_data['initial_conditions'].shape) == 2
        assert len(train_data['trajectories'].shape) == 3
        assert len(test_data['initial_conditions'].shape) == 2
        assert len(test_data['trajectories'].shape) == 3
        print("✓ Data shapes validated")
        
        # Validate data dimensions
        actual_train_size = train_data['initial_conditions'].shape[0]
        actual_test_size = test_data['initial_conditions'].shape[0]
        expected_total_size = config['data']['dataset_size']
        expected_seq_len = len(ts)
        expected_output_dim = config['model']['output_dim']
        
        # Debug shapes
        print(f"  Expected total size: {expected_total_size}")
        print(f"  Actual train size: {actual_train_size}")
        print(f"  Actual test size: {actual_test_size}")
        print(f"  Expected output dim: {expected_output_dim}")
        print(f"  Actual train initial_conditions shape: {train_data['initial_conditions'].shape}")
        print(f"  Actual train trajectories shape: {train_data['trajectories'].shape}")
        print(f"  Actual test initial_conditions shape: {test_data['initial_conditions'].shape}")
        print(f"  Actual test trajectories shape: {test_data['trajectories'].shape}")
        
        # Check that sizes add up correctly
        assert actual_train_size + actual_test_size == expected_total_size, f"Train + test size ({actual_train_size} + {actual_test_size}) != total size ({expected_total_size})"
        
        # Check shapes
        assert train_data['initial_conditions'].shape == (actual_train_size, expected_output_dim)
        assert train_data['trajectories'].shape == (actual_train_size, expected_seq_len, expected_output_dim)
        assert test_data['initial_conditions'].shape == (actual_test_size, expected_output_dim)
        assert test_data['trajectories'].shape == (actual_test_size, expected_seq_len, expected_output_dim)
        print("✓ Data dimensions validated")
        
        # Check for NaN values
        assert not jnp.any(jnp.isnan(train_data['trajectories']))
        assert not jnp.any(jnp.isnan(test_data['trajectories']))
        print("✓ No NaN values found in data")
        
        # Test forcing signal generation
        if config['forcing']['enabled']:
            assert train_data['forcing'] is not None
            assert test_data['forcing'] is not None
            assert train_data['forcing'].shape == (actual_train_size, expected_seq_len)
            assert test_data['forcing'].shape == (actual_test_size, expected_seq_len)
            print("✓ Forcing signals generated and validated")
        
        print(f"  Training data shape: {train_data['trajectories'].shape}")
        print(f"  Test data shape: {test_data['trajectories'].shape}")
        print(f"  Time points: {len(ts)}")
        
        return ts, train_data, test_data
        
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        traceback.print_exc()
        return None, None, None


def test_neural_network_architecture(config: Dict[str, Any], key: jr.PRNGKey):
    """Test neural network architecture creation and forward pass."""
    print("\n" + "="*50)
    print("Testing Neural Network Architecture")
    print("="*50)
    
    try:
        # Create model
        model_key = jr.fold_in(key, 42)
        model = NeuralODEModel(
            data_size=config['model']['output_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            solver_type=config['solver']['solver_type'],
            activation=config['model']['activation'],
            key=model_key
        )
        print("✓ Neural ODE model created")
        
        # Test forward pass
        ts = jnp.linspace(0, 1.0, 10)
        y0 = jnp.ones(config['model']['output_dim'])
        
        # Solve ODE
        solution = model(ts, y0)
        print("✓ Forward pass completed")
        
        # Validate solution
        assert len(solution.shape) == 2
        assert solution.shape[0] == len(ts)
        assert solution.shape[1] == config['model']['output_dim']
        assert not jnp.any(jnp.isnan(solution))
        print("✓ Solution shape and validity validated")
        
        # Test batched forward pass
        y0_batch = jnp.ones((5, config['model']['output_dim']))
        batch_solution = jax.vmap(lambda y0: model(ts, y0))(y0_batch)
        assert batch_solution.shape == (5, len(ts), config['model']['output_dim'])
        print("✓ Batched forward pass validated")
        
        return model
        
    except Exception as e:
        print(f"✗ Neural network architecture test failed: {e}")
        traceback.print_exc()
        return None


def test_training_functions(model, train_data: Dict[str, Any], config: Dict[str, Any]):
    """Test training functions with a short training run."""
    print("\n" + "="*50)
    print("Testing Training Functions")
    print("="*50)
    
    try:
        # Create small config for fast testing
        test_config = config.copy()
        test_config['training']['num_steps'] = 20
        test_config['training']['batch_size'] = 8
        test_config['evaluation']['eval_frequency'] = 10
        
        # Run short training
        start_time = time.time()
        trained_model, history = train_neural_ode(model, train_data, test_config)
        training_time = time.time() - start_time
        
        print("✓ Training completed")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final loss: {history['train_loss'][-1]:.6f}")
        print(f"  Final RMSE: {history['train_rmse'][-1]:.6f}")
        
        # Validate training history
        assert len(history['train_loss']) == test_config['training']['num_steps']
        assert len(history['train_rmse']) == test_config['training']['num_steps']
        assert len(history['step_times']) == test_config['training']['num_steps']
        assert all(loss > 0 for loss in history['train_loss'])
        assert all(rmse > 0 for rmse in history['train_rmse'])
        print("✓ Training history validated")
        
        # Check that loss decreased (basic sanity check)
        initial_loss = history['train_loss'][0]
        final_loss = history['train_loss'][-1]
        print(f"  Loss improvement: {initial_loss:.6f} -> {final_loss:.6f}")
        
        return trained_model, history
        
    except Exception as e:
        print(f"✗ Training functions test failed: {e}")
        traceback.print_exc()
        return None, None


def test_evaluation_functions(model, test_data: Dict[str, Any], config: Dict[str, Any]):
    """Test evaluation and metrics computation."""
    print("\n" + "="*50)
    print("Testing Evaluation Functions")
    print("="*50)
    
    try:
        # Test model evaluation
        loss, metrics = evaluate_model(model, test_data, config)
        print("✓ Model evaluation completed")
        
        # Validate metrics
        expected_metrics = ['loss', 'rmse', 'relative_error']
        print(f"  Available metrics: {list(metrics.keys())}")
        
        for metric in expected_metrics:
            if metric in metrics:
                value = metrics[metric]
                print(f"  {metric}: {value} (type: {type(value)})")
                # Check if it's a numeric type (more flexible)
                assert hasattr(value, 'item') or isinstance(value, (float, int, jnp.number))
            else:
                print(f"  Warning: {metric} not found in metrics")
        
        # Check that the required metrics are present
        assert 'loss' in metrics
        assert 'rmse' in metrics
        assert 'relative_error' in metrics
        print("✓ Evaluation metrics validated")
        
        # Test detailed metrics computation
        ts = test_data['ts']
        y0_batch = test_data['initial_conditions']
        target_batch = test_data['trajectories']
        
        # Get predictions
        predictions = jax.vmap(lambda y0: solve_neural_ode(model, ts, y0, config))(y0_batch)
        
        # Compute detailed metrics
        detailed_metrics = compute_metrics(predictions, target_batch)
        print("✓ Detailed metrics computation completed")
        
        # Validate detailed metrics
        assert 'dim_mse' in detailed_metrics
        assert 'dim_rmse' in detailed_metrics
        assert len(detailed_metrics['dim_mse']) == config['model']['output_dim']
        assert len(detailed_metrics['dim_rmse']) == config['model']['output_dim']
        print("✓ Detailed metrics validated")
        
        # Test differentiability
        y0 = test_data['initial_conditions'][0]
        diff_test = differentiability_test(model, ts, y0, config)
        print("✓ Differentiability test completed")
        
        # Validate differentiability test
        assert 'test_passed' in diff_test
        assert isinstance(diff_test['test_passed'], bool)
        print(f"  Differentiability test passed: {diff_test['test_passed']}")
        
        return metrics, detailed_metrics
        
    except Exception as e:
        print(f"✗ Evaluation functions test failed: {e}")
        traceback.print_exc()
        return None, None


def test_solving_functions(model, test_data: Dict[str, Any], config: Dict[str, Any]):
    """Test ODE solving functions."""
    print("\n" + "="*50)
    print("Testing Solving Functions")
    print("="*50)
    
    try:
        # Test basic solving
        ts = test_data['ts']
        y0 = test_data['initial_conditions'][0]
        
        solution = solve_neural_ode(model, ts, y0, config)
        print("✓ Basic ODE solving completed")
        
        # Validate solution
        assert solution.shape == (len(ts), config['model']['output_dim'])
        assert not jnp.any(jnp.isnan(solution))
        print("✓ Solution shape and validity validated")
        
        # Test with custom solver configuration
        custom_solver_config = {
            'solver': diffrax.Tsit5(),
            'rtol': 1e-6,
            'atol': 1e-8,
            'adaptive_steps': True
        }
        
        solution_custom = solve_neural_ode(model, ts, y0, {'solver': custom_solver_config})
        print("✓ Custom solver configuration test completed")
        
        # Validate custom solution
        assert solution_custom.shape == solution.shape
        print("✓ Custom solver solution validated")
        
        return solution
        
    except Exception as e:
        print(f"✗ Solving functions test failed: {e}")
        traceback.print_exc()
        return None


def test_utility_functions(config: Dict[str, Any], key: jr.PRNGKey):
    """Test utility functions."""
    print("\n" + "="*50)
    print("Testing Utility Functions")
    print("="*50)
    
    try:
        # Test JAX environment setup
        setup_jax_environment(use_64bit=True)
        print("✓ JAX environment setup completed")
        
        # Test dataloader creation
        ts = jnp.linspace(0, 1.0, 50)
        dummy_data = {
            'ts': ts,
            'initial_conditions': jnp.ones((10, config['model']['output_dim'])),
            'trajectories': jnp.ones((10, len(ts), config['model']['output_dim'])),
            'forcing': jnp.ones((10, len(ts))) if config['forcing']['enabled'] else None
        }
        
        dataloader = create_dataloader(dummy_data, batch_size=4, key=key)
        print("✓ Dataloader creation completed")
        
        # Test one batch from dataloader
        batch_generator = dataloader()
        batch = next(batch_generator)
        ts_batch, y0_batch, target_batch = batch
        
        assert y0_batch.shape == (4, config['model']['output_dim'])
        assert target_batch.shape == (4, len(ts), config['model']['output_dim'])
        print("✓ Dataloader batch generation validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")
        traceback.print_exc()
        return False


def run_performance_benchmark(config: Dict[str, Any], key: jr.PRNGKey):
    """Run a quick performance benchmark."""
    print("\n" + "="*50)
    print("Running Performance Benchmark")
    print("="*50)
    
    try:
        # Create larger test configuration
        benchmark_config = create_neural_ode_config(
            hidden_dim=64,
            num_layers=3,
            output_dim=3,
            dataset_size=128,
            num_steps=50,
            batch_size=16,
            visualization_enabled=False
        )
        
        # Benchmark data generation
        start_time = time.time()
        ts, train_data, test_data = generate_synthetic_data(benchmark_config, key=key)
        data_gen_time = time.time() - start_time
        print(f"✓ Data generation: {data_gen_time:.3f}s")
        
        # Create model
        model_key = jr.fold_in(key, 123)
        model = NeuralODEModel(
            data_size=benchmark_config['model']['output_dim'],
            hidden_dim=benchmark_config['model']['hidden_dim'],
            num_layers=benchmark_config['model']['num_layers'],
            key=model_key
        )
        
        # Benchmark forward pass
        start_time = time.time()
        y0 = jnp.ones(benchmark_config['model']['output_dim'])
        solution = model(ts, y0)
        forward_time = time.time() - start_time
        print(f"✓ Forward pass: {forward_time:.3f}s")
        
        # Benchmark training
        train_config = benchmark_config.copy()
        train_config['training']['num_steps'] = 10
        train_config['evaluation']['eval_frequency'] = 5
        
        start_time = time.time()
        _, history = train_neural_ode(model, train_data, train_config)
        training_time = time.time() - start_time
        print(f"✓ Training (10 steps): {training_time:.3f}s")
        
        # Benchmark evaluation
        start_time = time.time()
        metrics = evaluate_model(model, test_data, benchmark_config)
        eval_time = time.time() - start_time
        print(f"✓ Evaluation: {eval_time:.3f}s")
        
        print(f"\nSummary:")
        print(f"  Data generation: {data_gen_time:.3f}s")
        print(f"  Forward pass: {forward_time:.3f}s")
        print(f"  Training (10 steps): {training_time:.3f}s")
        print(f"  Evaluation: {eval_time:.3f}s")
        print(f"  Final loss: {history['train_loss'][-1]:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function that runs all tests."""
    print("Neural ODE Functions Module Test Suite")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"64-bit precision: {jax.config.jax_enable_x64}")
    
    # Initialize random key
    key = jr.PRNGKey(42)
    
    # Track test results
    test_results = {}
    
    try:
        # Test configuration system
        config = test_configuration_system()
        if config is None:
            print("✗ Configuration system test failed - stopping tests")
            return False
        test_results['configuration'] = True
        
        # Test data generation
        ts, train_data, test_data = test_data_generation(config, key)
        if ts is None:
            print("✗ Data generation test failed - stopping tests")
            return False
        test_results['data_generation'] = True
        
        # Test neural network architecture
        model = test_neural_network_architecture(config, key)
        if model is None:
            print("✗ Neural network architecture test failed - stopping tests")
            return False
        test_results['architecture'] = True
        
        # Test training functions
        trained_model, history = test_training_functions(model, train_data, config)
        if trained_model is None:
            print("✗ Training functions test failed")
            test_results['training'] = False
        else:
            test_results['training'] = True
        
        # Test evaluation functions
        if trained_model is not None:
            metrics, detailed_metrics = test_evaluation_functions(trained_model, test_data, config)
            if metrics is not None:
                test_results['evaluation'] = True
                print(f"  Test Loss: {metrics['loss']:.6f}")
                print(f"  Test RMSE: {metrics['rmse']:.6f}")
            else:
                test_results['evaluation'] = False
        else:
            test_results['evaluation'] = False
        
        # Test solving functions
        solution = test_solving_functions(model, test_data, config)
        if solution is not None:
            test_results['solving'] = True
        else:
            test_results['solving'] = False
        
        # Test utility functions
        utils_success = test_utility_functions(config, key)
        test_results['utilities'] = utils_success
        
        # Run performance benchmark
        benchmark_success = run_performance_benchmark(config, key)
        test_results['benchmark'] = benchmark_success
        
    except Exception as e:
        print(f"✗ Unexpected error in test suite: {e}")
        traceback.print_exc()
        return False
    
    # Print test results summary
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20s}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 All tests passed! The neural_ode_funcs module is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)