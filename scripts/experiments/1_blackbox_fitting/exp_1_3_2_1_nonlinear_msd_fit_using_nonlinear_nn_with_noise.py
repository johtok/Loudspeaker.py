#!/usr/bin/env python3
"""Nonlinear MSD fit with additive noise sweeps (taxonomy 1.3.2.1)."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / "1_blackbox_fitting"
    / "exp_1_2_2_1_nonlinear_msd_fit_using_nonlinear_nn.py"
)


def _noise_levels() -> tuple[float, ...]:
    # Cover a few orders of magnitude around the nominal scale of the data.
    return (0.005, 0.01, 0.02)


def main() -> None:
    spec = spec_from_file_location(
        "exp_1_2_2_1_nonlinear_msd_fit_using_nonlinear_nn", TARGET
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    for noise_std in _noise_levels():
        print(f"Running {Path(__file__).name} with noise_std={noise_std:.3f}")
        module.main(noise_std=noise_std)


if __name__ == "__main__":
    main()
