"""Scientific and corporate colour palettes for the Pairing Analysis UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigurePalette:
    """Two-series colours plus the density colourscale used by the multiplot."""

    reference: str
    comparison: str
    density_scale: str

    @property
    def series(self) -> tuple[str, str]:
        return self.reference, self.comparison


FIGURE_PALETTES: dict[str, FigurePalette] = {
    "Corporate": FigurePalette("#03547C", "#A39161", "Viridis"),
    "Viridis": FigurePalette("#440154", "#FDE725", "Viridis"),
    "Cividis": FigurePalette("#00224E", "#FDE737", "Cividis"),
    "Plasma": FigurePalette("#0D0887", "#F0F921", "Plasma"),
    "Turbo": FigurePalette("#4669E8", "#E65A2E", "Turbo"),
    "Spectral": FigurePalette("#3288BD", "#D53E4F", "Spectral"),
}

DEFAULT_FIGURE_PALETTE = "Corporate"
