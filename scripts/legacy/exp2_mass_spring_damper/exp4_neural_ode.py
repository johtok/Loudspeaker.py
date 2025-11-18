#!/usr/bin/env python3
"""
Exp4: Neural ODE Mass-Spring-Damper Experiment (Real params + noise)
==================================================================

This experiment is identical to Exp3 but with parameters initialized using 
real physical parameters with different levels of noise perturbations.
All other parameters remain the same:
- Pink noise bandwidth 1Hz-100Hz
- Timeseries 50 samples 
- Sampling frequency 300Hz
- Pink noise bandpassed to 1-50Hz
- Mass spring damper peak tuned to 25Hz
- Neural network 6 parameters in 2x3 matrix without bias
- Parameters initialized using real params with noise perturbations

Based on the todo.md requirements for neural ODE experiments.
"""

import time
import signal
import numpy as np
from contextlib import contextmanager
from typing import Dict, Any

# Timeout context manager
@contextmanager
def timeout_context(seconds: int = 600):
    """Context manager for timeout functionality (10 minutes default)."""
    
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

class RealParamsNeuralODEFunc:
    """
    Neural ODE function that initializes parameters based on real physical parameters.
    """
    
    def __init__(self, data_size: int, key: jax.random.PRNGKey, 
                 msd_params: Dict[str, float], noise_levels: list = [0.1, 0.01, 0.001]):
        """
        Initialize with real physical parameters plus noise perturbations.
        
        Args:
            data_size: Size of the state vector (3 for position, velocity, acceleration)
            key: JAX random key
            msd_params: Dictionary with mass, stiffness, damping parameters
            noise_levels: List of noise levels to try
        """
        self.data_size = data_size
        self.msd_params = msd_params
        self.noise_levels = noise_levels
        
        # Create multiple parameter sets with different noise levels
        self.real_weight_matrices = []
        self.noisy_weight_matrices = []
        
        # Generate real physical weight matrix based on MSD parameters
        real_weight_matrix = self._create_real_weight_matrix()
        self.real_weight_matrices.append(real_weight_matrix)
        
        # Generate noisy versions
        for noise_level in noise_levels:
            noisy_matrix = self._add_noise_to_matrix(real_weight_matrix, noise_level, key)
            self.noisy_weight_matrices.append(noisy_matrix)
    
    def _create_real_weight_matrix(self) -> np.ndarray:
        """
        Create a weight matrix based on real mass-spring-damper physics.
        
        For a 3D state [position, velocity, acceleration], the dynamics are:
        - d/dt(position) = velocity
        - d/dt(velocity) = acceleration  
        - d/dt(acceleration) = -(k/m)*position - (c/m)*velocity + forcing/m
        
        This gives us a system matrix that can be used as initialization.
        """
        m = self.msd_params['mass']
        k = self.msd_params['stiffness'] 
        c = self.msd_params['damping_coefficient']
        
        # System matrix for linear MSD system
        # dx/dt = A * x + B * u
        # where x = [position, velocity, acceleration]
        # For a second-order system, we typically have:
        # d²x/dt² + (c/m)*dx/dt + (k/m)*x = F/m
        
        # For a 3D state [x, v, a], we can construct a reasonable system matrix
        # This is a simplified approach - in reality acceleration dynamics would be more complex
        system_matrix = np.array([
            [0.0, 1.0, 0.0],           # dx/dt = v
            [-k/m, -c/m, 0.0],         # dv/dt = -(k/m)*x - (c/m)*v
            [0.0, -k/m, -c/m]          # da/dt = -(k/m)*v - (c/m)*a (simplified)
        ])
        
        return system_matrix
    
    def _add_noise_to_matrix(self, matrix: np.ndarray, noise_level: float, key) -> np.ndarray:
        """Add noise perturbation to the real weight matrix."""
        noise_key, _ = jax.random.split(key)
        noise = jax.random.normal(noise_key, matrix.shape) * noise_level
        return matrix + noise

def run_exp4():
    """Run Exp4 neural ODE experiment with real params initialization."""
    
    with timeout_context(600):  # 10 minute timeout
        print("Starting Exp4: Neural ODE with Real Params + Noise Perturbations")
        print("=" * 60)
        
        # Import required modules
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        import equinox as eqx
        
        # Import neural ode functions
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from neural_ode_funcs import (
            create_neural_ode_config, generate_synthetic_data, NeuralODEModel,
            train_neural_ode, evaluate_model_step, solve_neural_ode,
            compute_metrics, plot_training_history, plot_trajectories,
            plot_phase_space
        )
        
        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)
        print(f"JAX devices: {jax.devices()}")
        print(f"64-bit precision: {jax.config.jax_enable_x64}")
        
        # Physical parameters for real initialization
        mass = 0.05  # kg
        natural_frequency = 25.0  # Hz
        damping_ratio = 0.01
        stiffness = mass * (2 * jnp.pi * natural_frequency)**2
        damping_coefficient = 2 * damping_ratio * mass * (2 * jnp.pi * natural_frequency)
        
        msd_params = {
            'mass': mass,
            'stiffness': stiffness,
            'damping_coefficient': damping_coefficient,
            'natural_frequency': natural_frequency,
            'damping_ratio': damping_ratio
        }
        
        print("Real Physical Parameters:")
        print(f"  Mass: {mass} kg")
        print(f"  Stiffness: {stiffness:.2f} N/m")
        print(f"  Damping: {damping_coefficient:.4f} Ns/m")
        print(f"  Natural Frequency: {natural_frequency} Hz")
        
        # Test different noise levels
        noise_levels = [0.1, 0.01, 0.001, 0.0001]
        results_by_noise = {}
        
        for noise_level in noise_levels:
            print(f"\n{'='*50}")
            print(f"Testing noise level: {noise_level}")
            print(f"{'='*50}")
            
            # Create real params neural ODE function
            key = jr.PRNGKey(42)
            real_func = RealParamsNeuralODEFunc(
                data_size=3,  # position, velocity, acceleration
                key=key,
                msd_params=msd_params,
                noise_levels=[noise_level]
            )
            
            # Get the noisy weight matrix
            noisy_weight_matrix = real_func.noisy_weight_matrices[0]
            print(f"Weight matrix range: [{noisy_weight_matrix.min():.4f}, {noisy_weight_matrix.max():.4f}]")
            
            # Exp4 Configuration - same as Exp3
            config = create_neural_ode_config(
                # Model parameters - will be overridden with real params
                hidden_dim=3,  # Will create 2x3 matrix
                num_layers=1,
                output_dim=3,  # position, velocity, acceleration
                activation='tanh',
                
                # Training parameters
                learning_rate=1e-3,
                num_steps=2000,
                batch_size=10,
                weight_decay=1e-4,
                optimizer='adam',
                
                # Data parameters - 50 samples like Exp3
                dataset_size=50,
                test_split=0.2,
                noise_level=0.001,
                simulation_time=0.1,
                sample_rate=300,
                initial_condition_range=(-0.1, 0.1),
                
                # Forcing parameters
                forcing_enabled=True,
                forcing_type='pink_noise',
                forcing_amplitude=1.0,
                forcing_frequency_range=(1.0, 100.0),
                
                # Solver parameters
                dt=1e-3,
                solver_type='tsit5',
                rtol=1e-6,
                atol=1e-8,
                adaptive_steps=True,
                
                # Visualization parameters
                visualization_enabled=True,
                save_dir=f'exp4_results_noise_{noise_level}/',
                plot_format='png',
                dpi=300,
                
                # Evaluation parameters
                eval_frequency=100,
                early_stopping=True,
                patience=100,
                
                # MSD-specific parameters
                msd_params={
                    'mass': mass,
                    'natural_frequency': natural_frequency,
                    'damping_ratio': damping_ratio,
                    'forcing_amplitude': 1.0,
                },
                
                # Numerical parameters
                use_64bit=True,
                gradient_clipping=1.0,
            )
            
            # Create save directory
            import os
            os.makedirs(config['visualization']['save_dir'], exist_ok=True)
            
            # Generate synthetic data
            print("Generating synthetic data...")
            key = jr.PRNGKey(42)
            ts, train_data, test_data = generate_synthetic_data(config, key=key)
            
            # Create neural ODE model with real params initialization
            print("Creating neural ODE model with real params initialization...")
            model_key = jr.PRNGKey(123)
            
            # Custom model class that uses real params initialization
            class RealParamsNeuralODEModel(NeuralODEModel):
                def __init__(self, data_size: int, hidden_dim: int, num_layers: int, 
                            solver_type: str = 'tsit5', activation: str = 'tanh', 
                            weight_matrix: jnp.ndarray = None,
                            *, key: jax.random.PRNGKey, **kwargs):
                    super().__init__(data_size, hidden_dim, num_layers, solver_type, 
                                   activation, key=key, **kwargs)
                    
                    # Override the weight matrix with real params
                    if weight_matrix is not None:
                        self.func.weight_matrix = weight_matrix
            
            model = RealParamsNeuralODEModel(
                data_size=config['model']['output_dim'],
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                solver_type=config['solver']['solver_type'],
                activation=config['model']['activation'],
                weight_matrix=noisy_weight_matrix,
                key=model_key
            )
            
            # Count parameters
            total_params = sum(p.size for p in jax.tree_util.tree_leaves(model) if p.ndim > 0)
            print(f"Model parameters: {total_params}")
            print(f"Weight matrix shape: {model.func.weight_matrix.shape}")
            print(f"Initial weight matrix norm: {jnp.linalg.norm(model.func.weight_matrix):.6f}")
            
            # Train the model
            print("Training neural ODE model...")
            start_time = time.time()
            
            try:
                trained_model, training_history = train_neural_ode(
                    model, train_data, config, test_data
                )
                
                training_time = time.time() - start_time
                print(f"Training completed in {training_time:.2f} seconds")
                
            except Exception as e:
                print(f"Training failed: {e}")
                continue
            
            # Evaluate the model
            print("Evaluating trained model...")
            test_loss, test_metrics = evaluate_model_step(trained_model, test_data, config['solver'])
            
            # Compute additional metrics
            predictions = jax.vmap(lambda y0: solve_neural_ode(trained_model, ts, y0, config))(
                test_data['initial_conditions']
            )
            additional_metrics = compute_metrics(predictions, test_data['trajectories'])
            
            print("Evaluation Results:")
            print(f"  Test Loss (MSE): {test_loss:.8f}")
            print(f"  Test RMSE: {test_metrics['rmse']:.8f}")
            print(f"  Relative Error: {test_metrics['relative_error']:.8f}")
            print(f"  R² Score: {additional_metrics['r2_score']:.6f}")
            print(f"  Max Error: {additional_metrics['max_error']:.8f}")
            
            # Check if error requirement is met
            error_requirement_met = test_metrics['rmse'] < 1e-3
            print(f"Error requirement (< 1e-3): {'✓ MET' if error_requirement_met else '✗ NOT MET'}")
            
            # Generate visualizations
            print("Generating visualizations...")
            plot_training_history(training_history, config)
            plot_trajectories(trained_model, test_data, config, num_samples=min(5, len(test_data['initial_conditions'])))
            plot_phase_space(trained_model, test_data, config, num_samples=min(3, len(test_data['initial_conditions'])))
            
            # Save results
            import pickle
            results = {
                'config': config,
                'model': trained_model,
                'history': training_history,
                'metrics': {
                    'test_loss': float(test_loss),
                    'test_rmse': float(test_metrics['rmse']),
                    'test_rel_error': float(test_metrics['relative_error']),
                    'additional_metrics': additional_metrics,
                    'error_requirement_met': error_requirement_met
                },
                'predictions': predictions,
                'ground_truth': test_data['trajectories'],
                'time_points': ts,
                'training_time': training_time,
                'noise_level': noise_level,
                'initial_weight_matrix': noisy_weight_matrix
            }
            
            with open(f"{config['visualization']['save_dir']}/exp4_results.pkl", 'wb') as f:
                pickle.dump(results, f)
            
            results_by_noise[noise_level] = results
            
            print(f"Results saved to {config['visualization']['save_dir']}")
        
        # Summary across all noise levels
        print("\n" + "=" * 60)
        print("EXP4 SUMMARY: RESULTS ACROSS NOISE LEVELS")
        print("=" * 60)
        
        best_noise_level = None
        best_rmse = float('inf')
        
        for noise_level, results in results_by_noise.items():
            rmse = results['metrics']['test_rmse']
            met_requirement = results['metrics']['error_requirement_met']
            
            print(f"Noise Level {noise_level:6f}: RMSE={rmse:.8f}, Requirement={'✓' if met_requirement else '✗'}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_noise_level = noise_level
        
        print(f"\nBest noise level: {best_noise_level}")
        print(f"Best RMSE: {best_rmse:.8f}")
        
        if best_rmse < 1e-3:
            print("🎉 Exp4 SUCCESS: Error requirement achieved with real params initialization!")
        else:
            print("⚠️  Exp4 PARTIAL: Best result achieved but error requirement not met")
        
        print("=" * 60)
        
        return results_by_noise

if __name__ == "__main__":
    try:
        results = run_exp4()
        
        # Check if any noise level achieved the requirement
        success = any(r['metrics']['error_requirement_met'] for r in results.values())
        
        if success:
            print("\n🎉 Exp4 SUCCESS: Error requirement achieved with real params initialization!")
        else:
            print("\n⚠️  Exp4 PARTIAL: Training completed but error requirement not met")
            
    except TimeoutError as e:
        print(f"\n⏰ Exp4 TIMEOUT: {e}")
    except Exception as e:
        print(f"\n❌ Exp4 ERROR: {e}")