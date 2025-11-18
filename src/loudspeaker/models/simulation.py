from __future__ import annotations

from dataclasses import dataclass

from ..loudspeaker_sim import (
    LoudspeakerConfig,
    LoudspeakerSimulationResult,
    simulate_loudspeaker_system,
)
from ..testsignals import ControlSignal
from .linear import LinearLoudspeakerModel


@dataclass
class LoudspeakerSimulationModel:
    """High-level façade exposing the linear loudspeaker plant.

    The model bundles together the physical configuration, a helper for
    instantiating the analytical linear state-space approximation, and a
    convenience wrapper around the time-domain simulator.  It keeps scripts
    focused on experiment logic instead of repeatedly wiring the same plumbing.
    """

    config: LoudspeakerConfig

    def linear_model(self, **kwargs) -> LinearLoudspeakerModel:
        """Return the deterministic linearized loudspeaker model."""

        return LinearLoudspeakerModel(self.config, **kwargs)

    def simulate(
        self,
        control: ControlSignal,
        *,
        capture_details: bool = False,
    ) -> LoudspeakerSimulationResult:
        """Forward to :func:`simulate_loudspeaker_system` with this config."""

        return simulate_loudspeaker_system(
            self.config,
            control,
            capture_details=capture_details,
        )
