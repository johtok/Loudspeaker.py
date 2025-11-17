from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from loudspeaker.metrics import mae, mse


def _paired_arrays():
    """Return pairs of matching-shaped numpy arrays for metric tests."""

    shape_strategy = hnp.array_shapes(min_dims=1, max_dims=2, min_side=1, max_side=4)
    element_strategy = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
    )
    return shape_strategy.flatmap(
        lambda shape: st.tuples(
            hnp.arrays(dtype=np.float64, shape=shape, elements=element_strategy),
            hnp.arrays(dtype=np.float64, shape=shape, elements=element_strategy),
        )
    )


@given(_paired_arrays())
def test_mse_matches_numpy(arrays):
    ref_np, pred_np = arrays
    ref = jnp.asarray(ref_np)
    pred = jnp.asarray(pred_np)
    expected = np.mean((ref_np - pred_np) ** 2)
    chex.assert_trees_all_close(
        mse(ref, pred),
        jnp.asarray(expected, dtype=ref.dtype),
        atol=1e-6,
        rtol=1e-6,
    )


@given(_paired_arrays())
def test_mae_matches_numpy(arrays):
    ref_np, pred_np = arrays
    ref = jnp.asarray(ref_np)
    pred = jnp.asarray(pred_np)
    expected = np.mean(np.abs(ref_np - pred_np))
    chex.assert_trees_all_close(
        mae(ref, pred),
        jnp.asarray(expected, dtype=ref.dtype),
        atol=1e-6,
        rtol=1e-6,
    )
