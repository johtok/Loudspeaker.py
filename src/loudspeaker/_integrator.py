from __future__ import annotations

from typing import Any, Callable, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve

ScalarLike: TypeAlias = bool | int | float | jax.Array | np.ndarray
VectorField = Callable[[ScalarLike, jnp.ndarray, Any], jnp.ndarray]


def _prepare_time_grid(config: Any, ts: jnp.ndarray | None) -> jnp.ndarray:
    if ts is None:
        return jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    return jnp.asarray(ts, dtype=jnp.float32)


def integrate_system(
    config: Any,
    vector_field: VectorField,
    *,
    solver: Tsit5 | None = None,
    ts: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    solver = solver or Tsit5()
    ts_values = _prepare_time_grid(config, ts)
    term = ODETerm(vector_field)

    @eqx.filter_jit
    def _solve(initial_state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        solution = diffeqsolve(
            term,
            solver,
            t0=ts_values[0],
            t1=ts_values[-1],
            dt0=config.dt,
            y0=initial_state,
            saveat=SaveAt(ts=ts_values),
        )
        if solution.ts is None or solution.ys is None:
            raise RuntimeError("Solver returned no trajectory.")
        return solution.ts, solution.ys

    return _solve(config.initial_state)
