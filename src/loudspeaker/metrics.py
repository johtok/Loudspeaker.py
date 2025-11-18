from typing import TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float

FloatArray: TypeAlias = Float[jnp.ndarray, "..."]


@eqx.filter_jit
def mse(pred: FloatArray, target: FloatArray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


@eqx.filter_jit
def mae(pred: FloatArray, target: FloatArray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - target))


@eqx.filter_jit
def nrmse(pred: FloatArray, target: FloatArray, normalizer: FloatArray) -> jnp.ndarray:
    """Compute normalized RMSE using a provided scale (e.g., σ)."""

    scaled_error = (pred - target) / normalizer
    return jnp.sqrt(jnp.mean(scaled_error**2))


@eqx.filter_jit
def norm_mse(pred: FloatArray, target: FloatArray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute normalized MSE via the squared normalized RMSE."""

    normalizer = jnp.std(target, axis=0, keepdims=True) + eps
    return nrmse(pred, target, normalizer) ** 2
