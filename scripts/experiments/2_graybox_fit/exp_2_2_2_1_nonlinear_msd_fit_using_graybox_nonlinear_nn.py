#!/usr/bin/env python3
"""Entry point for taxonomy 2.2.2.1: noisy graybox nonlinear MSD fit."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

TARGET = (
    Path(__file__).resolve().parents[1]
    / "1_blackbox_fitting"
    / "exp_1_3_2_1_nonlinear_msd_fit_using_nonlinear_nn_with_noise.py"
)


def main() -> None:
    spec = spec_from_file_location(
        "exp_1_3_2_1_nonlinear_msd_fit_using_nonlinear_nn_with_noise", TARGET
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
