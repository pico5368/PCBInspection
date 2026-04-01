"""Inspector protocol — common interface for all inspection modules."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pcb_inspection.common.types import InspectionResult


class Inspector(Protocol):
    """Protocol that all inspection modules must implement."""

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        """Run inspection on a single ROI image.

        Args:
            roi_image: Cropped component image from aligned board.
            reference: Golden/reference image for comparison (may be None).
            config: Inspection-specific parameters (thresholds, etc.)

        Returns:
            InspectionResult with severity, score, and details.
        """
        ...

    def load(self, model_path: str | None = None) -> None:
        """Load model or resources needed for inspection."""
        ...
