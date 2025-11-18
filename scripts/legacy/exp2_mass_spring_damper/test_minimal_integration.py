#!/usr/bin/env python3
"""
Minimal test script for neural_ode_example.py integration
========================================================

This script tests the core functionality without visualization to avoid Qt issues.
"""

import sys
import os

# Add the current directory to Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_core_functionality():
    """Test core neural ODE functionality."""
    print("Testing core neural ODE functionality...")
    
    try:
        # Import JAX ecosystem
        import jax
        import jax.numpy as jnp
        import equinox as eqx
        import diffrax as dx
        import optax
        print("✓ JAX ecosystem imports successful")
        
        # Import neural_ode_funcs
        import neural_ode_funcs
        print("✓ neural_ode_funcs import successful")
        
        # Test configuration creation
        config = neural_ode_funcs.create_neural_ode_config(
            hidden_dim=16,  # Small for testing
            num_layers=1,
            output_dim=3,
            dataset_size=8,
            num_steps=10,
            visualization_enabled=False,
            msd_params={
                'mass': 0.05,
                'natural_frequency': 25.0,
                'damping_ratio': 0.01,
                'forcing_amplitude': 0.5,
                'forcing_type': 'pink_noise'
            }
        )
        print("✓ Configuration creation successful")
        print(f"  Model: hidden_dim={config['model']['hidden_dim']}, output_dim={config['model']['output_dim']}")
        print(f"  MSD params: {config['msd_params']}")
        
        # Test data generation
        key = jax.random.PRNGKey(42)
        ts, train_data, test_data = neural_ode_funcs.generate_synthetic_data(config, key=key)
        print(f"✓ Data generation successful: ts={ts.shape}, train={train_data['trajectories'].shape}")
        
        # Test model creation
        model_key = jax.random.PRNGKey(123)
        model = neural_ode_funcs.NeuralODEModel(
            data_size=config['model']['output_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            key=model_key
        )
        print("✓ Model creation successful")
        
        # Test forward pass
        y0 = jax.random.uniform(jax.random.PRNGKey(456), (3,), minval=-0.5, maxval=0.5)
        solution = neural_ode_funcs.solve_neural_ode(model, ts[:10], y0, config)
        print(f"✓ Forward pass successful: solution shape {solution.shape}")
        
        # Test loss function with matching dimensions
        # Use only first 10 time points for both predictions and targets
        ts_short = ts[:10]
        predictions = jax.vmap(lambda y0: model(ts_short, y0))(train_data['initial_conditions'][:2])
        targets = train_data['trajectories'][:2, :10, :]
        
        loss, metrics = neural_ode_funcs.loss_fn_neural_ode(
            model, ts_short, train_data['initial_conditions'][:2], targets
        )
        print(f"✓ Loss function successful: loss={loss:.6f}, rmse={metrics['rmse']:.6f}")
        
        print("\n✓ All core functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Minimal Neural ODE Integration Test")
    print("=" * 40)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    success = test_core_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ Integration test PASSED")
        sys.exit(0)
    else:
        print("✗ Integration test FAILED")
        sys.exit(1)