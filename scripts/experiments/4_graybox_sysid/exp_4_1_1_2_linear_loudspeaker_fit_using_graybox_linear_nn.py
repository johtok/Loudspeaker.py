#!/usr/bin/env python3
"""Entry point for taxonomy 4.1.1.2: Linear loudspeaker fit using graybox linear NN."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

TARGET = (
    Path(__file__).parents[1]
    / "1_blackbox_fitting"
    / "exp_1_1_1_2_linear_loudspeaker_fit_using_linear_nn.py"
)


def main() -> None:
    spec = spec_from_file_location(
        "exp_1_1_1_2_linear_loudspeaker_fit_using_linear_nn", TARGET
    )
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
