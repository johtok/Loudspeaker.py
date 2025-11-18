from __future__ import annotations

import chex
import jax.numpy as jnp
import numpy as np
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hnp

from loudspeaker.metrics import mae, mse, norm_mse, nrmse


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


def test_nrmse_matches_manual_computation():
    target = jnp.array([0.0, 1.0, 2.0], dtype=jnp.float32)
    pred = target + jnp.array([0.5, -0.5, 1.0], dtype=jnp.float32)
    sigma = jnp.ones_like(target) * 0.5
    manual = np.sqrt(np.mean(((np.asarray(pred) - np.asarray(target)) / 0.5) ** 2))
    chex.assert_trees_all_close(nrmse(pred, target, sigma), jnp.float32(manual))


def test_norm_mse_matches_squared_nrmse():
    target = jnp.array([[0.0, 1.0], [2.0, 3.0]], dtype=jnp.float32)
    pred = target + 0.25
    normalizer = jnp.std(target, axis=0, keepdims=True) + 1e-8
    expected = nrmse(pred, target, normalizer) ** 2
    chex.assert_trees_all_close(norm_mse(pred, target), expected)
