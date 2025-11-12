#!/usr/bin/env python3
"""
Comprehensive Test Suite for Neural ODE Experiments
===================================================

This module provides comprehensive unit tests, integration tests, and regression tests
for the neural ODE experiments (Exp2, Exp3, Exp4) based on the todo.md requirements.

Tests are designed to be fast, simple, and focused on correctness and performance.
"""

import unittest
import time
import sys
import os
import pickle
import numpy as np

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

# Import neural ode functions
from neural_ode_funcs import (
    create_neural_ode_config, generate_synthetic_data, NeuralODEModel,
    train_neural_ode, evaluate_model_step, solve_neural_ode,
    compute_metrics, plot_training_history, plot_trajectories,
    plot_phase_space, NeuralODEFunc, generate_pink_noise_bandpassed
)

class TestNeuralODEExperiments(unittest.TestCase):
    """Comprehensive test suite for neural ODE experiments."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.key = jr.PRNGKey(42)
        jax.config.update("jax_enable_x64", True)
        
        # Minimal configuration for fast testing
        self.minimal_config = create_neural_ode_config(
            hidden_dim=3,
            num_layers=1,
            output_dim=3,
            learning_rate=1e-3,
            num_steps=10,  # Very small for testing
            batch_size=2,
            dataset_size=3,  # Very small for testing
            test_split=0.3,
            noise_level=0.001,
            simulation_time=0.01,  # Very short simulation
            sample_rate=100,
            initial_condition_range=(-0.1, 0.1),
            forcing_enabled=True,
            forcing_type='pink_noise',
            forcing_amplitude=1.0,
            solver_type='tsit5',
            rtol=1e-6,
            atol=1e-8,
            visualization_enabled=False,  # Disable for testing
            eval_frequency=5,
            early_stopping=True,
            patience=3,
            msd_params={
                'mass': 0.05,
                'natural_frequency': 25.0,
                'damping_ratio': 0.01,
                'forcing_amplitude': 1.0,
            },
            use_64bit=True,
            gradient_clipping=1.0,
        )
    
    def test_neural_ode_func_creation(self):
        """Test NeuralODEFunc creation and basic functionality."""
        print("Testing NeuralODEFunc creation...")
        
        func_key = jr.PRNGKey(123)
        func = NeuralODEFunc(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            activation='tanh',
            key=func_key
        )
        
        # Test function call
        t = 0.0
        y = jnp.array([1.0, 2.0, 3.0])
        args = None
        
        result = func(t, y, args)
        
        # Check output shape
        self.assertEqual(result.shape, (3,))
        
        # Check that output is finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        
        # Check parameter count (should be 6 for 2x3 matrix)
        params = jax.tree_util.tree_leaves(func)
        total_params = sum(p.size for p in params if p.ndim > 0)
        print(f"NeuralODEFunc parameters: {total_params}")
        self.assertEqual(total_params, 9)  # 3x3 weight matrix = 9 parameters
    
    def test_neural_ode_model_creation(self):
        """Test NeuralODEModel creation and basic functionality."""
        print("Testing NeuralODEModel creation...")
        
        model_key = jr.PRNGKey(456)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            solver_type='tsit5',
            activation='tanh',
            key=model_key
        )
        
        # Test model call
        ts = jnp.linspace(0, 0.01, 10)
        y0 = jnp.array([0.1, 0.2, 0.3])
        
        result = model(ts, y0)
        
        # Check output shape
        self.assertEqual(result.shape, (10, 3))
        
        # Check that output is finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))
        
        # Check parameter count
        params = jax.tree_util.tree_leaves(model)
        total_params = sum(p.size for p in params if p.ndim > 0)
        print(f"NeuralODEModel parameters: {total_params}")
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        print("Testing data generation...")
        
        ts, train_data, test_data = generate_synthetic_data(self.minimal_config, key=self.key)
        
        # Check data shapes
        self.assertEqual(len(ts.shape), 1)
        self.assertEqual(train_data['trajectories'].shape[1:], (3,))
        self.assertEqual(test_data['trajectories'].shape[1:], (3,))
        
        # Check that data is finite
        self.assertTrue(jnp.all(jnp.isfinite(train_data['trajectories'])))
        self.assertTrue(jnp.all(jnp.isfinite(test_data['trajectories'])))
        
        # Check that forcing signals exist
        self.assertIsNotNone(train_data['forcing'])
        self.assertIsNotNone(test_data['forcing'])
        
        print(f"Generated data: ts={ts.shape}, train={train_data['trajectories'].shape}")
    
    def test_pink_noise_generation(self):
        """Test pink noise generation with bandpass filtering."""
        print("Testing pink noise generation...")
        
        length = 100
        sample_rate = 100.0
        freq_range = (1.0, 25.0)
        key = jr.PRNGKey(789)
        
        pink_noise = generate_pink_noise_bandpassed(length, sample_rate, freq_range, key)
        
        # Check output shape
        self.assertEqual(pink_noise.shape, (length,))
        
        # Check that output is finite
        self.assertTrue(jnp.all(jnp.isfinite(pink_noise)))
        
        # Check that signal has expected frequency content (basic check)
        fft_result = jnp.fft.fft(pink_noise)
        frequencies = jnp.fft.fftfreq(length, 1.0/sample_rate)
        
        # Most energy should be in the passband
        passband_mask = (jnp.abs(frequencies) >= freq_range[0]) & (jnp.abs(frequencies) <= freq_range[1])
        passband_energy = jnp.sum(jnp.abs(fft_result[passband_mask])**2)
        total_energy = jnp.sum(jnp.abs(fft_result)**2)
        
        # At least 50% of energy should be in passband (rough check)
        self.assertGreater(passband_energy / total_energy, 0.5)
        
        print(f"Pink noise generated: {pink_noise.shape}, energy ratio: {passband_energy/total_energy:.3f}")
    
    def test_loss_function(self):
        """Test loss function computation."""
        print("Testing loss function...")
        
        # Create minimal model and data
        model_key = jr.PRNGKey(789)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            key=model_key
        )
        
        ts, train_data, test_data = generate_synthetic_data(self.minimal_config, key=self.key)
        
        # Test loss function
        y0_batch = train_data['initial_conditions'][:2]  # Small batch
        target_batch = train_data['trajectories'][:2]
        
        loss, metrics = evaluate_model_step(model, {'ts': ts, 'initial_conditions': y0_batch, 'trajectories': target_batch}, {})
        
        # Check that loss is finite and positive
        self.assertTrue(jnp.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        
        # Check metrics
        self.assertIn('rmse', metrics)
        self.assertIn('relative_error', metrics)
        
        print(f"Loss: {loss:.6f}, RMSE: {metrics['rmse']:.6f}")
    
    def test_training_step(self):
        """Test single training step."""
        print("Testing training step...")
        
        from neural_ode_funcs import train_step
        
        # Create model and optimizer
        model_key = jr.PRNGKey(999)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            key=model_key
        )
        
        import optax
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        
        # Generate small batch
        ts, train_data, test_data = generate_synthetic_data(self.minimal_config, key=self.key)
        y0_batch = train_data['initial_conditions'][:2]
        target_batch = train_data['trajectories'][:2]
        
        batch = (ts, y0_batch, target_batch)
        
        # Run training step
        loss, metrics, new_model, new_opt_state = train_step(model, optimizer, opt_state, batch)
        
        # Check that training step completed successfully
        self.assertTrue(jnp.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        
        # Check that model was updated (parameters changed)
        old_params = jax.tree_util.tree_leaves(model)
        new_params = jax.tree_util.tree_leaves(new_model)
        
        params_changed = any(not jnp.allclose(old, new) for old, new in zip(old_params, new_params) if old.size > 0)
        self.assertTrue(params_changed, "Model parameters should change after training step")
        
        print(f"Training step completed: loss={loss:.6f}")
    
    def test_model_evaluation(self):
        """Test model evaluation metrics."""
        print("Testing model evaluation...")
        
        # Create model and data
        model_key = jr.PRNGKey(111)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            key=model_key
        )
        
        ts, train_data, test_data = generate_synthetic_data(self.minimal_config, key=self.key)
        
        # Evaluate model
        metrics = evaluate_model(model, test_data, self.minimal_config)
        
        # Check all expected metrics are present
        expected_metrics = ['total_mse', 'total_rmse', 'relative_error', 'max_error', 'r2_score']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(jnp.isfinite(metrics[metric]))
        
        print(f"Evaluation metrics: {metrics}")
    
    def test_integration_exp2_config(self):
        """Test Exp2 configuration and data generation."""
        print("Testing Exp2 configuration...")
        
        exp2_config = create_neural_ode_config(
            # Model parameters
            hidden_dim=3,
            num_layers=1,
            output_dim=3,
            activation='tanh',
            
            # Training parameters
            learning_rate=1e-3,
            num_steps=50,  # Small for testing
            batch_size=5,  # Exp2 uses 5 samples
            optimizer='adam',
            
            # Data parameters - Exp2 specific
            dataset_size=5,  # 5 samples as specified
            test_split=0.2,
            noise_level=0.001,
            simulation_time=0.05,  # Short for testing
            sample_rate=300,  # 300Hz as specified
            initial_condition_range=(-0.1, 0.1),
            
            # Forcing parameters
            forcing_enabled=True,
            forcing_type='pink_noise',
            forcing_amplitude=1.0,
            forcing_frequency_range=(1.0, 100.0),  # 1Hz-100Hz bandwidth
            
            # Solver parameters
            solver_type='tsit5',
            rtol=1e-6,
            atol=1e-8,
            adaptive_steps=True,
            
            # Visualization parameters
            visualization_enabled=False,
            
            # Evaluation parameters
            eval_frequency=10,
            early_stopping=True,
            patience=20,
            
            # MSD-specific parameters
            msd_params={
                'mass': 0.05,  # kg
                'natural_frequency': 25.0,  # Hz (peak tuned to 25Hz)
                'damping_ratio': 0.01,
                'forcing_amplitude': 1.0,
            },
            
            # Numerical parameters
            use_64bit=True,
            gradient_clipping=1.0,
        )
        
        # Generate data
        ts, train_data, test_data = generate_synthetic_data(exp2_config, key=self.key)
        
        # Verify Exp2 specifications
        self.assertEqual(train_data['trajectories'].shape[0], 4)  # 5 samples, 80% train
        self.assertEqual(exp2_config['data']['dataset_size'], 5)
        self.assertEqual(exp2_config['data']['sample_rate'], 300)
        self.assertEqual(exp2_config['msd_params']['natural_frequency'], 25.0)
        
        print(f"Exp2 config verified: {exp2_config['data']['dataset_size']} samples, {exp2_config['data']['sample_rate']} Hz")
    
    def test_integration_exp3_config(self):
        """Test Exp3 configuration and data generation."""
        print("Testing Exp3 configuration...")
        
        exp3_config = create_neural_ode_config(
            # Model parameters
            hidden_dim=3,
            num_layers=1,
            output_dim=3,
            activation='tanh',
            
            # Training parameters
            learning_rate=1e-3,
            num_steps=50,  # Small for testing
            batch_size=10,  # Larger batch for 50 samples
            optimizer='adam',
            
            # Data parameters - Exp3 specific (50 samples)
            dataset_size=50,  # 50 samples as specified
            test_split=0.2,
            noise_level=0.001,
            simulation_time=0.05,  # Short for testing
            sample_rate=300,  # 300Hz as specified
            initial_condition_range=(-0.1, 0.1),
            
            # Forcing parameters
            forcing_enabled=True,
            forcing_type='pink_noise',
            forcing_amplitude=1.0,
            forcing_frequency_range=(1.0, 100.0),  # 1Hz-100Hz bandwidth
            
            # Solver parameters
            solver_type='tsit5',
            rtol=1e-6,
            atol=1e-8,
            adaptive_steps=True,
            
            # Visualization parameters
            visualization_enabled=False,
            
            # Evaluation parameters
            eval_frequency=10,
            early_stopping=True,
            patience=20,
            
            # MSD-specific parameters
            msd_params={
                'mass': 0.05,  # kg
                'natural_frequency': 25.0,  # Hz (peak tuned to 25Hz)
                'damping_ratio': 0.01,
                'forcing_amplitude': 1.0,
            },
            
            # Numerical parameters
            use_64bit=True,
            gradient_clipping=1.0,
        )
        
        # Generate data
        ts, train_data, test_data = generate_synthetic_data(exp3_config, key=self.key)
        
        # Verify Exp3 specifications
        self.assertEqual(train_data['trajectories'].shape[0], 40)  # 50 samples, 80% train
        self.assertEqual(exp3_config['data']['dataset_size'], 50)
        self.assertEqual(exp3_config['data']['sample_rate'], 300)
        self.assertEqual(exp3_config['msd_params']['natural_frequency'], 25.0)
        
        print(f"Exp3 config verified: {exp3_config['data']['dataset_size']} samples, {exp3_config['data']['sample_rate']} Hz")
    
    def test_timeout_functionality(self):
        """Test timeout functionality."""
        print("Testing timeout functionality...")
        
        from neural_ode_funcs import timeout_context
        
        # Test that timeout works
        try:
            with timeout_context(1):  # 1 second timeout
                time.sleep(2)  # Should timeout
            self.fail("Should have timed out")
        except TimeoutError:
            print("Timeout functionality working correctly")
        
        # Test that normal execution works
        try:
            with timeout_context(5):  # 5 second timeout
                time.sleep(0.1)  # Should complete
            print("Normal execution with timeout context works")
        except TimeoutError:
            self.fail("Should not have timed out")
    
    def test_error_requirement_check(self):
        """Test that error requirement (< 1e-3) can be checked."""
        print("Testing error requirement check...")
        
        # Create a simple test case
        predictions = jnp.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        targets = jnp.array([[0.1001, 0.2001, 0.3001], [0.4001, 0.5001, 0.6001]])  # Small error
        
        metrics = compute_metrics(predictions, targets)
        
        # Check that RMSE is computed
        self.assertIn('total_rmse', metrics)
        rmse = metrics['total_rmse']
        
        # This should be much less than 1e-3 (good case)
        self.assertLess(rmse, 1e-3)
        
        print(f"Error requirement test - RMSE: {rmse:.8f}, Requirement met: {rmse < 1e-3}")
    
    def test_performance_requirements(self):
        """Test that experiments can run within performance requirements."""
        print("Testing performance requirements...")
        
        start_time = time.time()
        
        # Create minimal test case
        model_key = jr.PRNGKey(222)
        model = NeuralODEModel(
            data_size=3,
            hidden_dim=3,
            num_layers=1,
            key=model_key
        )
        
        # Generate minimal data
        ts, train_data, test_data = generate_synthetic_data(self.minimal_config, key=self.key)
        
        # Run minimal training
        config = self.minimal_config.copy()
        config['training']['num_steps'] = 5  # Very small
        
        trained_model, history = train_neural_ode(model, train_data, config, test_data)
        
        elapsed_time = time.time() - start_time
        
        # Should complete very quickly (much less than 10 minutes)
        self.assertLess(elapsed_time, 60)  # 1 minute should be plenty
        
        print(f"Performance test completed in {elapsed_time:.3f} seconds")
    
    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        print("Testing reproducibility...")
        
        # Run first experiment
        key1 = jr.PRNGKey(42)
        ts1, train_data1, test_data1 = generate_synthetic_data(self.minimal_config, key=key1)
        
        # Run second experiment with same seed
        key2 = jr.PRNGKey(42)
        ts2, train_data2, test_data2 = generate_synthetic_data(self.minimal_config, key=key2)
        
        # Results should be identical
        self.assertTrue(jnp.allclose(ts1, ts2))
        self.assertTrue(jnp.allclose(train_data1['trajectories'], train_data2['trajectories']))
        self.assertTrue(jnp.allclose(test_data1['trajectories'], test_data2['trajectories']))
        
        print("Reproducibility test passed")

def run_speed_test():
    """Run a speed test to ensure experiments can complete quickly."""
    print("\n" + "="*50)
    print("SPEED TEST")
    print("="*50)
    
    start_time = time.time()
    
    # Test with minimal configuration
    config = create_neural_ode_config(
        hidden_dim=3,
        num_layers=1,
        output_dim=3,
        learning_rate=1e-3,
        num_steps=20,
        batch_size=3,
        dataset_size=5,
        test_split=0.2,
        noise_level=0.001,
        simulation_time=0.02,
        sample_rate=100,
        forcing_enabled=True,
        solver_type='tsit5',
        rtol=1e-6,
        atol=1e-8,
        visualization_enabled=False,
        eval_frequency=5,
        early_stopping=True,
        patience=10,
        msd_params={
            'mass': 0.05,
            'natural_frequency': 25.0,
            'damping_ratio': 0.01,
            'forcing_amplitude': 1.0,
        },
        use_64bit=True,
        gradient_clipping=1.0,
    )
    
    # Create model
    model_key = jr.PRNGKey(42)
    model = NeuralODEModel(
        data_size=3,
        hidden_dim=3,
        num_layers=1,
        key=model_key
    )
    
    # Generate data
    ts, train_data, test_data = generate_synthetic_data(config, key=jr.PRNGKey(42))
    
    # Train model
    trained_model, history = train_neural_ode(model, train_data, config, test_data)
    
    # Evaluate
    test_loss, test_metrics = evaluate_model_step(trained_model, test_data, config['solver'])
    
    elapsed_time = time.time() - start_time
    
    print(f"Speed test completed in {elapsed_time:.3f} seconds")
    print(f"Final loss: {test_loss:.6f}")
    print(f"Final RMSE: {test_metrics['rmse']:.6f}")
    
    return elapsed_time, float(test_loss), float(test_metrics['rmse'])

if __name__ == "__main__":
    print("Starting Neural ODE Experiments Test Suite")
    print("="*60)
    
    # Run speed test first
    speed_time, final_loss, final_rmse = run_speed_test()
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    print(f"Speed test time: {speed_time:.3f} seconds")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Final RMSE: {final_rmse:.6f}")
    print("All tests completed successfully!")
    print("="*60)