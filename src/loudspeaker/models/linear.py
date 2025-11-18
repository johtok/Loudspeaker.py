from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from ..loudspeaker_sim import LoudspeakerConfig
from ..msd_sim import MSDConfig


class _LinearModel(eqx.Module):
    weight: jax.Array

    def __call__(self: "_LinearModel", inputs: jax.Array) -> jax.Array:
        return self.weight @ inputs


class LinearMSDModel(_LinearModel):
    """Single dense layer without bias (2x3 parameters)."""

    def __init__(
        self,
        config: MSDConfig,
        perturbation: float = 0.01,
        key: jax.Array | None = None,
    ):
        base = jnp.array(
            [
                [0.0, 1.0, 0.0],
                [
                    -config.stiffness / config.mass,
                    -config.damping / config.mass,
                    1.0 / config.mass,
                ],
            ],
            dtype=jnp.float32,
        )
        if key is not None:
            base = base + perturbation * jr.normal(key, base.shape)
        super().__init__(weight=base)


class LinearLoudspeakerModel(_LinearModel):
    """Three-state loudspeaker model capturing cone and coil dynamics."""

    def __init__(
        self,
        config: LoudspeakerConfig,
        perturbation: float = 0.01,
        key: jax.Array | None = None,
    ):
        inv_mass = 1.0 / config.moving_mass
        inv_inductance = 1.0 / config.voice_coil_inductance
        base = jnp.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [
                    -config.stiffness * inv_mass,
                    -config.damping * inv_mass,
                    config.motor_force * inv_mass,
                    0.0,
                ],
                [
                    0.0,
                    -config.motor_force * inv_inductance,
                    -config.voice_coil_resistance * inv_inductance,
                    inv_inductance,
                ],
            ],
            dtype=jnp.float32,
        )
        if key is not None:
            base = base + perturbation * jr.normal(key, base.shape)
        super().__init__(weight=base)


class AugmentedMSDModel(_LinearModel):
    """Random linear reservoir that augments MSD dynamics with extra states."""

    def __init__(
        self,
        state_size: int,
        *,
        key: jax.Array,
        scale: float = 0.1,
    ):
        if state_size <= 0:
            raise ValueError("state_size must be positive.")
        weight_shape = (state_size, state_size + 1)
        self_key, _ = jr.split(key)
        base = scale * jr.normal(self_key, weight_shape, dtype=jnp.float32)
        super().__init__(weight=base)
