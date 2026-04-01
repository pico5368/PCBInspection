"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from pcb_inspection.common.types import ComponentROI, InspectionType


@pytest.fixture
def sample_image() -> np.ndarray:
    """A simple 640x480 BGR test image with some features."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Add some rectangles to simulate components
    img[100:150, 200:260] = [60, 60, 200]   # brownish component
    img[200:230, 300:350] = [50, 50, 50]     # dark component
    img[300:340, 100:180] = [180, 180, 180]  # bright component
    return img


@pytest.fixture
def sample_reference() -> np.ndarray:
    """Golden reference image matching sample_image."""
    img = np.full((480, 640, 3), 128, dtype=np.uint8)
    img[100:150, 200:260] = [60, 60, 200]
    img[200:230, 300:350] = [50, 50, 50]
    img[300:340, 100:180] = [180, 180, 180]
    return img


@pytest.fixture
def sample_rois() -> list[ComponentROI]:
    """Sample ROIs matching features in sample_image."""
    return [
        ComponentROI(
            component_id="R1",
            component_type="0603",
            bbox=(195, 95, 70, 60),
            rotation=0,
            inspection_types=[InspectionType.REFERENCE, InspectionType.RULE_BASED],
        ),
        ComponentROI(
            component_id="U1",
            component_type="SOP8",
            bbox=(295, 195, 60, 40),
            rotation=0,
            inspection_types=[
                InspectionType.REFERENCE,
                InspectionType.RULE_BASED,
                InspectionType.ANOMALY,
            ],
        ),
        ComponentROI(
            component_id="C1",
            component_type="0805",
            bbox=(95, 295, 90, 50),
            rotation=0,
            inspection_types=[InspectionType.REFERENCE, InspectionType.BLOB],
        ),
    ]
