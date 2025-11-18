import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


@eqx.filter_jit
def mae(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - target))


@eqx.filter_jit
def nrmse(pred: jnp.ndarray, target: jnp.ndarray, normalizer: jnp.ndarray) -> jnp.ndarray:
    """Compute normalized RMSE using a provided scale (e.g., σ)."""

    scaled_error = (pred - target) / normalizer
    return jnp.sqrt(jnp.mean(scaled_error**2))


@eqx.filter_jit
def norm_mse(pred: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Compute normalized MSE via the squared normalized RMSE."""

    normalizer = jnp.std(target, axis=0, keepdims=True) + eps
    return nrmse(pred, target, normalizer) ** 2
