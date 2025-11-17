import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


@eqx.filter_jit
def mae(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(pred - target))
