from __future__ import annotations

from dataclasses import dataclass

from loudspeaker.loudspeaker_sim import (
    LoudspeakerConfig,
    LoudspeakerSimulationResult,
    simulate_loudspeaker_system,
)
from loudspeaker.models import LinearLoudspeakerModel
from loudspeaker.testsignals import ControlSignal


@dataclass
class LoudspeakerSimulationModel:
    """High-level entry point for the linear loudspeaker plant."""

    config: LoudspeakerConfig

    def linear_model(self, **kwargs) -> LinearLoudspeakerModel:
        return LinearLoudspeakerModel(self.config, **kwargs)

    def simulate(
        self,
        control: ControlSignal,
        *,
        capture_details: bool = False,
    ) -> LoudspeakerSimulationResult:
        return simulate_loudspeaker_system(
            self.config,
            control,
            capture_details=capture_details,
        )
