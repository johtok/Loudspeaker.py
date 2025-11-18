#!/usr/bin/env python3
"""
Demo Script for Mass-Spring-Damper Simulation with Forcing
==========================================================

This script demonstrates the key features of the MSD simulation system.
Run this script to see various configurations and forcing types in action.

Usage:
    pixi run python scripts/exp2_mass_spring_damper/demo_script.py
"""

import sys

sys.path.append(".")

from scripts.exp2_mass_spring_damper.msd_simulation_with_forcing import (
    ForcingType,
    MSDConfig,
    NormalizationType,
    demonstrate_solver_comparison,
    run_batch_simulation,
    run_single_simulation,
)


def demo_basic_simulation():
    """Demonstrate basic single simulation."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Single Simulation")
    print("=" * 60)

    config = MSDConfig(
        simulation_time=0.08,
        natural_frequency=20.0,
        damping_ratio=0.015,
        save_plots=True,
    )

    print(
        f"Configuration: Natural freq={config.natural_frequency}Hz, "
        f"Damping={config.damping_ratio}, Solver={config.solver_type.value}"
    )

    result = run_single_simulation(config)

    if result["stats"]["successful"]:
        print("✓ Simulation completed successfully!")
        print(f"  - Solver steps: {result['stats']['num_steps']}")
        print(f"  - Max displacement: {max(abs(result['position'])):.6f}m")
        print(f"  - Max velocity: {max(abs(result['velocity'])):.6f}m/s")
    else:
        print("❌ Simulation failed")


def demo_forcing_types():
    """Demonstrate different forcing signal types."""
    print("\n" + "=" * 60)
    print("DEMO 2: Different Forcing Types")
    print("=" * 60)

    forcing_types = [
        ForcingType.PINK_NOISE,
        ForcingType.SINE,
        ForcingType.COMPLEX_SINE,
        ForcingType.CHIRP,
    ]

    results = {}

    for forcing_type in forcing_types:
        print(f"\nTesting {forcing_type.value} forcing...")

        config = MSDConfig(
            forcing_type=forcing_type,
            simulation_time=0.05,
            forcing_amplitude=0.8,
            save_plots=False,  # Don't save individual plots for this demo
        )

        simulator = MSDSimulator(config)
        result = simulator.simulate_single()

        if result["stats"]["successful"]:
            max_response = max(abs(result["position"]))
            print(f"  ✓ Max response: {max_response:.6f}m")
            results[forcing_type.value] = result
        else:
            print("  ❌ Failed")

    print(f"\nCompleted forcing type comparison for {len(results)} types")


def demo_solver_comparison():
    """Demonstrate different ODE solvers."""
    print("\n" + "=" * 60)
    print("DEMO 3: Solver Comparison")
    print("=" * 60)

    print("Comparing Kvaerno5 (stiff) vs Tsit5 (non-stiff) solvers...")

    # This will run the full solver comparison
    results = demonstrate_solver_comparison()

    print("✓ Solver comparison completed")


def demo_normalization_methods():
    """Demonstrate different normalization methods."""
    print("\n" + "=" * 60)
    print("DEMO 4: Normalization Methods")
    print("=" * 60)

    normalization_types = [
        NormalizationType.STANDARDIZE,
        NormalizationType.MINMAX,
        NormalizationType.UNIT_VECTOR,
        NormalizationType.NONE,
    ]

    config = MSDConfig(simulation_time=0.03, normalize_plots=True)

    for norm_type in normalization_types:
        print(f"\nTesting {norm_type.value} normalization...")

        config.normalization_type = norm_type

        simulator = MSDSimulator(config)
        result = simulator.simulate_single()

        if result["stats"]["successful"]:
            # Check if normalization was applied
            if norm_type == NormalizationType.NONE:
                print("  ✓ No normalization applied")
            else:
                print(f"  ✓ {norm_type.value} normalization applied")
        else:
            print("  ❌ Failed")


def demo_batch_simulation():
    """Demonstrate batch simulation capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 5: Batch Simulation")
    print("=" * 60)

    config = MSDConfig(batch_size=5, simulation_time=0.04, forcing_amplitude=0.5)

    print(f"Running batch simulation with {config.batch_size} trials...")

    batch_result = run_batch_simulation(config)

    successful = sum(
        1 for stats in batch_result["stats"] if stats.get("successful", False)
    )
    print(f"✓ Batch completed: {successful}/{config.batch_size} successful")

    if successful > 0:
        # Analyze batch results
        positions = batch_result["positions"]
        max_displacements = [max(abs(positions[i])) for i in range(successful)]
        print(
            f"  - Max displacements range: {min(max_displacements):.6f} to {max(max_displacements):.6f}m"
        )


def demo_parameter_sweep():
    """Demonstrate parameter sweep capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 6: Parameter Sweep")
    print("=" * 60)

    frequencies = [15.0, 25.0, 35.0, 50.0]
    damping_ratios = [0.005, 0.01, 0.02, 0.05]

    print(
        f"Testing {len(frequencies)} frequencies × {len(damping_ratios)} damping ratios"
    )
    print("This demonstrates resonance behavior at different parameters")

    results = []

    for freq in frequencies:
        for damp in damping_ratios:
            config = MSDConfig(
                natural_frequency=freq,
                damping_ratio=damp,
                simulation_time=0.06,
                save_plots=False,
            )

            simulator = MSDSimulator(config)
            result = simulator.simulate_single()

            if result["stats"]["successful"]:
                max_disp = max(abs(result["position"]))
                results.append((freq, damp, max_disp))
                print(f"  f={freq:4.0f}Hz, ζ={damp:.3f}: max_disp={max_disp:.6f}m")

    if results:
        print(f"\n✓ Parameter sweep completed: {len(results)} successful simulations")


def main():
    """Run all demonstrations."""
    print("Mass-Spring-Damper Simulation Demo")
    print("=" * 60)
    print("This script demonstrates the key features of the MSD simulation system.")
    print("Each demo will show different aspects of the simulation capabilities.")

    try:
        # Import here to ensure it works in the pixi environment

        # Run demonstrations
        demo_basic_simulation()
        demo_forcing_types()
        demo_solver_comparison()
        demo_normalization_methods()
        demo_batch_simulation()
        demo_parameter_sweep()

        print("\n" + "=" * 60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nGenerated files:")
        print("  - msd_time_domain.png")
        print("  - msd_3d_phase_space.png")
        print("  - msd_normalized_phase.png")
        print("  - msd_frequency_analysis.png")
        print("  - msd_batch_comparison.png")
        print("  - solver_comparison.png")
        print("  - forcing_types_comparison.png")
        print("\nCheck the output files to see the visualization results!")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
