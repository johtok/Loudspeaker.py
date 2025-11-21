Project development guidelines (advanced, project-specific)

Overview
- This repository is managed with Pixi (conda-forge based) and exposes all common dev flows as Pixi tasks in pyproject.toml under [tool.pixi.tasks]. Prefer pixi run <task> over ad-hoc invocations.
- Primary stack: JAX/diffrax/equinox/optax on Python >=3.11 with linux-64 target. Plotting with matplotlib/seaborn/plotly. Testing with pytest + hypothesis + pytest-benchmark. Static analysis with ruff, mypy, pyright, bandit, radon, vulture.

Environment and build/config
- One-time setup
  - Install Pixi: https://pixi.sh (required).
  - Create/resolve the env and install the package in editable mode via Pixi tasks:
    - pixi run dev-setup  # runs: pre-commit install && python -m pip install -e .
    - Optional Jupyter kernel registration: pixi run init-kernel
- Notes on JAX:
  - Versions are pinned to jax/jaxlib >=0.7.2,<0.8. CPU-only works out of the box in Pixi. For GPU, jaxlib may need pip-install with appropriate CUDA/ROCm wheels outside of Pixi; keep versions consistent with jax.
- Editable install
  - The package is named loudspeaker, with sources in src/loudspeaker. Pixi config exposes a pypi-dependency mapping to editable path: [tool.pixi.pypi-dependencies].
- Cleaning artifacts
  - pixi run clean  # removes build/, dist/, *.egg-info/, .pytest_cache/, htmlcov/, .coverage

Running tests
- Layout and discovery
  - pytest is configured via [tool.pytest.ini_options] to search in: src, tests, scripts. The default test paths/tasks assume the main suite lives in tests/test_loudspeaker and tests/benchmarks.
- Common invocations (prefer Pixi tasks)
  - Run core suite: pixi run test          # maps to: pytest tests/test_loudspeaker
  - All tests incl. benchmarks dir: pixi run test-all
  - Coverage for src/, with HTML report to htmlcov/: pixi run test-cov
  - Only benchmarks (no correctness tests): pixi run bench or pixi run benchmark
  - Selective tests/examples:
    - Single file: pixi run pytest tests/test_loudspeaker/test_plotting.py -q
    - By keyword: pixi run pytest -k msd_sim -q
    - With Hypothesis health checks silenced: pixi run pytest -q -o hypothesis_show_locals=false
- Adding tests
  - Place new tests under tests/test_<area>/ or extend existing modules. Tests under src/ are also discovered, but keep product code separated.
  - If a test depends on JAX/diffrax numerics, seed deterministically when asserting bitwise equality; otherwise assert approximate properties (e.g., numpy.allclose with rtol/atol) to avoid spurious failures across CPU/GPU/BLAS impls.
  - For Hypothesis, cap example counts in slow numeric spaces and use strategies to bound array shapes and dtypes.

Type checking and static analysis
- Type checking
  - mypy (primary, permissive missing imports): pixi run type-check
    - Config: [tool.mypy] python_version = 3.11, ignore_missing_imports = true.
  - Pyright (secondary): pixi run type-check-pyright
    - Pyright config: pyrightconfig.json at repo root.
- Lint/format
  - Ruff (lint): pixi run lint  # checks src tests scripts; ignores in scripts/experiments/**/*.py: E402.
  - Ruff (format + import sort): pixi run format  # runs ruff format and ruff check --select I --fix
  - Lint targets are tuned for py311; line length 99, indent 4; plugins enabled: E4,E7,E9,F,I,UP. See [tool.ruff] in pyproject.
- Security and quality
  - Bandit: pixi run security
  - Complexity and maintainability (radon): pixi run complexity and pixi run maintainability
  - Dead code scan (vulture): pixi run dead-code

Benchmarks
- Use pytest-benchmark-based suites in tests/benchmarks via pixi run bench or benchmark. The log examples in logs/benchmark.log show stable runs; avoid running on CI unless required to keep timing variance low.

Experiment scripts and tracking
- Hydrated configs are under configs/, run orchestration via pixi run run-experiments (scripts/run_all_experiments.py). Hydra artifacts go to outputs/<timestamp>/.
- TensorBoard logs under out/tensorboard; launch via pixi run tensorboard.

Code style and conventions
- Source lives in src/loudspeaker; tests under tests/ and benchmark suites under tests/benchmarks.
- Match Ruff’s formatting (line length 99, 4-space indent). Use explicit imports; run pixi run format to enforce import order and formatting.
- Prefer immutable functional style for JAX code paths; avoid Python-side mutation in functions that will be transformed (jit, grad, vmap). Where stateful behavior is needed, isolate outside jit regions.
- Numerical tests: prefer property-based tests and tolerance-based assertions. Avoid relying on exact solver trajectories across solver backends; diffrax tolerances and step sizes can cause small divergences.

Troubleshooting notes (project-specific)
- If you see missing jax/jaxlib/numpy imports in IDE analysis under .pixi/, ensure you open the workspace with the Pixi interpreter and that the environment is resolved for linux-64.
- Some tests reference logs/test.log and logs/benchmark.log as prior runs; they are not part of pass/fail.
- When running on machines without AVX2 or with older CPUs, ensure the numpy binary in Pixi matches the platform capabilities (conda-forge handles this by default).

Reproducible local flow (suggested)
1) Resolve env and install hooks/package:
   - pixi install
   - pixi run dev-setup
2) Quick hygiene before commit:
   - pixi run format && pixi run lint && pixi run type-check
3) Run core tests (faster loop):
   - pixi run test
4) For in-depth coverage/bench:
   - pixi run test-cov
   - pixi run bench

Verified example: creating and running a simple test
- To demonstrate test wiring, during guideline preparation we created a temporary test and ran:
  - pixi run pytest tests/_demo/test_smoke.py -q
  - The test asserted True and passed, verifying the configuration.
  - The temporary file has been removed to keep the tree clean. To reproduce locally:
    - Create tests/_demo/test_smoke.py with:
      """
      def test_smoke():
          assert True
      """
    - Run: pixi run pytest tests/_demo/test_smoke.py -q
    - Delete the file afterwards.
