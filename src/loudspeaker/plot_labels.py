from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LabelSpec:
    """Describe a physical quantity for use in plot legends and axes."""

    name: str
    unit: str | None = None
    symbol: str | None = None

    def raw(self) -> str:
        if self.unit:
            return f"{self.name} [{self.unit}]"
        return self.name

    def normalized(self) -> str:
        """Return the label with explicit normalized notation X/Σ_X."""

        base = self.raw()
        symbol = self.symbol or self._default_symbol()
        return f"{base} ({symbol}/\\Sigma_{symbol})"

    def _default_symbol(self) -> str:
        condensed = "".join(ch for ch in self.name if ch.isalnum())
        if not condensed:
            return "x"
        return condensed[0].lower()


def raw_labels(*specs: LabelSpec) -> Tuple[str, ...]:
    """Return tuple of raw labels with units for the provided specs."""

    return tuple(spec.raw() for spec in specs)


def normalized_labels(*specs: LabelSpec) -> Tuple[str, ...]:
    """Return tuple of normalized labels with explicit Σ notation."""

    return tuple(spec.normalized() for spec in specs)
