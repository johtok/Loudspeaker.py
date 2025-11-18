#!/usr/bin/env python3
"""
Exp2: Neural ODE Mass-Spring-Damper Experiment
===============================================

This experiment implements neural ODE fitting with specific parameters:
- Pink noise bandwidth 1Hz-100Hz
- Timeseries 5 samples
- Sampling frequency 300Hz
- Pink noise bandpassed to 1-50Hz
- Mass spring damper peak tuned to 25Hz
- Neural network 6 parameters in 2x3 matrix without bias

Based on the todo.md requirements for neural ODE experiments.
"""

import signal
import time
from contextlib import contextmanager


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


def run_exp2():
    """Run Exp2 neural ODE experiment with timeout protection."""

    with timeout_context(600):  # 10 minute timeout
        print("Starting Exp2: Neural ODE Mass-Spring-Damper Experiment")
        print("=" * 60)

        # Import required modules
        import os

        # Import neural ode functions
        import sys

        import jax
        import jax.random as jr

        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        from neural_ode_funcs import (
            NeuralODEModel,
            compute_metrics,
            create_neural_ode_config,
            evaluate_model_step,
            generate_synthetic_data,
            plot_phase_space,
            plot_training_history,
            plot_trajectories,
            solve_neural_ode,
            train_neural_ode,
        )

        # Enable 64-bit precision
        jax.config.update("jax_enable_x64", True)
        print(f"JAX devices: {jax.devices()}")
        print(f"64-bit precision: {jax.config.jax_enable_x64}")

        # Exp2 Configuration
        config = create_neural_ode_config(
            # Model parameters - 6 parameters in 2x3 matrix
            hidden_dim=3,  # Will create 2x3 matrix
            num_layers=1,
            output_dim=3,  # position, velocity, acceleration
            activation="tanh",  # Simple activation for linear-ish behavior
            # Training parameters
            learning_rate=1e-3,
            num_steps=2000,  # Increased steps for better convergence
            batch_size=5,  # Small batch size for 5 samples
            weight_decay=1e-4,
            optimizer="adam",
            # Data parameters - Exp2 specific
            dataset_size=5,  # 5 samples as specified
            test_split=0.2,
            noise_level=0.001,  # Low noise for better fitting
            simulation_time=0.1,  # 100ms simulation
            sample_rate=300,  # 300Hz sampling as specified
            initial_condition_range=(-0.1, 0.1),
            # Forcing parameters
            forcing_enabled=True,
            forcing_type="pink_noise",
            forcing_amplitude=1.0,
            forcing_frequency_range=(1.0, 100.0),  # 1Hz-100Hz bandwidth
            # Solver parameters
            dt=1e-3,
            solver_type="tsit5",
            rtol=1e-6,  # High precision
            atol=1e-8,
            adaptive_steps=True,
            # Visualization parameters - export only, no display
            visualization_enabled=True,
            save_dir="exp2_results/",
            plot_format="png",
            dpi=300,
            # Evaluation parameters
            eval_frequency=100,
            early_stopping=True,
            patience=100,
            # MSD-specific parameters
            msd_params={
                "mass": 0.05,  # kg
                "natural_frequency": 25.0,  # Hz (peak tuned to 25Hz)
                "damping_ratio": 0.01,
                "forcing_amplitude": 1.0,
            },
            # Numerical parameters
            use_64bit=True,
            gradient_clipping=1.0,
        )

        print("Exp2 Configuration:")
        print(f"  Dataset size: {config['data']['dataset_size']} samples")
        print(f"  Sampling rate: {config['data']['sample_rate']} Hz")
        print(f"  Simulation time: {config['data']['simulation_time']} s")
        print(
            f"  MSD natural frequency: {config['msd_params']['natural_frequency']} Hz"
        )
        print(f"  Forcing bandwidth: {config['forcing']['forcing_frequency_range']} Hz")
        print("  Model parameters: 6 (2x3 matrix without bias)")

        # Create save directory
        import os

        os.makedirs(config["visualization"]["save_dir"], exist_ok=True)

        # Generate synthetic data
        print("\nGenerating synthetic data with MSD parameters...")
        key = jr.PRNGKey(42)
        ts, train_data, test_data = generate_synthetic_data(config, key=key)

        print("Data shapes:")
        print(f"  Time points: {ts.shape}")
        print(f"  Train trajectories: {train_data['trajectories'].shape}")
        print(f"  Test trajectories: {test_data['trajectories'].shape}")
        print(f"  Forcing signals: {train_data['forcing'].shape}")

        # Create neural ODE model with 6-parameter architecture
        print("\nCreating neural ODE model with 2x3 parameter matrix...")
        model_key = jr.PRNGKey(123)
        model = NeuralODEModel(
            data_size=config["model"]["output_dim"],
            hidden_dim=config["model"]["hidden_dim"],  # Creates 2x3 matrix
            num_layers=config["model"]["num_layers"],
            solver_type=config["solver"]["solver_type"],
            activation=config["model"]["activation"],
            key=model_key,
        )

        # Count parameters
        total_params = sum(
            p.size for p in jax.tree_util.tree_leaves(model) if p.ndim > 0
        )
        print(f"Model parameters: {total_params}")
        print(f"Weight matrix shape: {model.func.weight_matrix.shape}")

        # Train the model with timeout protection
        print("\nTraining neural ODE model...")
        start_time = time.time()

        try:
            trained_model, training_history = train_neural_ode(
                model, train_data, config, test_data
            )

            training_time = time.time() - start_time
            print(f"Training completed in {training_time:.2f} seconds")

        except Exception as e:
            print(f"Training failed: {e}")
            return None, None, None

        # Evaluate the model
        print("\nEvaluating trained model...")
        test_loss, test_metrics = evaluate_model_step(
            trained_model, test_data, config["solver"]
        )

        # Compute additional metrics
        predictions = jax.vmap(
            lambda y0: solve_neural_ode(trained_model, ts, y0, config)
        )(test_data["initial_conditions"])
        additional_metrics = compute_metrics(predictions, test_data["trajectories"])

        print("Evaluation Results:")
        print(f"  Test Loss (MSE): {test_loss:.8f}")
        print(f"  Test RMSE: {test_metrics['rmse']:.8f}")
        print(f"  Relative Error: {test_metrics['relative_error']:.8f}")
        print(f"  R² Score: {additional_metrics['r2_score']:.6f}")
        print(f"  Max Error: {additional_metrics['max_error']:.8f}")

        # Check if error requirement is met
        error_requirement_met = test_metrics["rmse"] < 1e-3
        print(
            f"\nError requirement (< 1e-3): {'✓ MET' if error_requirement_met else '✗ NOT MET'}"
        )

        # Generate visualizations (export only, no display)
        print("\nGenerating visualizations...")
        plot_training_history(training_history, config)
        plot_trajectories(
            trained_model,
            test_data,
            config,
            num_samples=min(3, len(test_data["initial_conditions"])),
        )
        plot_phase_space(
            trained_model,
            test_data,
            config,
            num_samples=min(2, len(test_data["initial_conditions"])),
        )

        # Save model weights and configuration
        print("\nSaving results...")
        import pickle

        results = {
            "config": config,
            "model": trained_model,
            "history": training_history,
            "metrics": {
                "test_loss": float(test_loss),
                "test_rmse": float(test_metrics["rmse"]),
                "test_rel_error": float(test_metrics["relative_error"]),
                "additional_metrics": additional_metrics,
                "error_requirement_met": error_requirement_met,
            },
            "predictions": predictions,
            "ground_truth": test_data["trajectories"],
            "time_points": ts,
            "training_time": training_time,
        }

        # Save results
        with open(f"{config['visualization']['save_dir']}/exp2_results.pkl", "wb") as f:
            pickle.dump(results, f)

        print(f"Results saved to {config['visualization']['save_dir']}")

        # Final summary
        print("\n" + "=" * 60)
        print("EXP2 NEURAL ODE EXPERIMENT COMPLETED")
        print("=" * 60)
        print(f"Final Test RMSE: {test_metrics['rmse']:.8f}")
        print(f"Error Requirement Met: {'Yes' if error_requirement_met else 'No'}")
        print(f"Training Steps: {len(training_history['train_loss'])}")
        print(f"Training Time: {training_time:.2f} seconds")
        print(f"Model Parameters: {total_params} (6-parameter architecture)")
        print(f"Dataset Size: {config['data']['dataset_size']} samples")
        print(f"Sampling Rate: {config['data']['sample_rate']} Hz")
        print("=" * 60)

        return trained_model, training_history, results


if __name__ == "__main__":
    try:
        model, history, results = run_exp2()
        if results and results["metrics"]["error_requirement_met"]:
            print("\n🎉 Exp2 SUCCESS: Error requirement achieved!")
        else:
            print("\n⚠️  Exp2 PARTIAL: Training completed but error requirement not met")
    except TimeoutError as e:
        print(f"\n⏰ Exp2 TIMEOUT: {e}")
    except Exception as e:
        print(f"\n❌ Exp2 ERROR: {e}")
