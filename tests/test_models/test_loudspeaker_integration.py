from __future__ import annotations

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from loudspeaker.loudspeaker_sim import LoudspeakerConfig
from loudspeaker.models import (
    AugmentedMSDModel,
    LinearLoudspeakerModel,
    LinearMSDModel,
    LoudspeakerSimulationModel,
)
from loudspeaker.msd_sim import MSDConfig
from loudspeaker.testsignals import build_control_signal


def test_models_package_exports_linear_primitives():
    loudspeaker_config = LoudspeakerConfig()
    loud_model = LinearLoudspeakerModel(loudspeaker_config)
    assert loud_model.weight.shape == (3, 4)

    msd_model = LinearMSDModel(MSDConfig())
    assert msd_model.weight.shape == (2, 3)

    reservoir = AugmentedMSDModel(state_size=4, key=jr.PRNGKey(0))
    assert reservoir.weight.shape == (4, 5)


def test_augmented_model_requires_positive_state_size():
    with pytest.raises(ValueError):
        AugmentedMSDModel(state_size=0, key=jr.PRNGKey(0))


def test_loudspeaker_simulation_model_wraps_simulator(monkeypatch):
    config = LoudspeakerConfig(num_samples=4)
    sim_model = LoudspeakerSimulationModel(config)
    assert isinstance(sim_model.linear_model(), LinearLoudspeakerModel)

    ts = jnp.linspace(0.0, config.duration, config.num_samples, dtype=jnp.float32)
    control = build_control_signal(ts, jnp.ones_like(ts))

    captured: dict[str, object] = {}

    def _fake_simulate(cfg, ctrl, capture_details):
        captured["config"] = cfg
        captured["control"] = ctrl
        captured["capture_details"] = capture_details
        return "result"

    monkeypatch.setattr(
        "loudspeaker.models.simulation.simulate_loudspeaker_system", _fake_simulate
    )

    result = sim_model.simulate(control, capture_details=True)
    assert result == "result"
    assert captured["config"] is config
    chex.assert_trees_all_close(captured["control"].values, control.values)
    assert captured["capture_details"] is True
