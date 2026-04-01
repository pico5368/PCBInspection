"""Tests for alignment module."""

from __future__ import annotations

import numpy as np

from pcb_inspection.alignment.fiducial import (
    FiducialConfig,
    detect_fiducials_by_circles,
)
from pcb_inspection.alignment.quality import check_alignment_quality
from pcb_inspection.common.types import AlignmentResult


def _make_image_with_circles(
    positions: list[tuple[int, int]], radius: int = 15
) -> np.ndarray:
    """Create a test image with white circles on dark background."""
    import cv2

    img = np.zeros((500, 700), dtype=np.uint8)
    for x, y in positions:
        cv2.circle(img, (x, y), radius, 255, -1)
    return img


class TestFiducialDetection:
    def test_detect_circles(self):
        positions = [(100, 100), (600, 400)]
        img = _make_image_with_circles(positions, radius=15)

        config = FiducialConfig(
            method="circle",
            min_diameter=0.3,
            max_diameter=5.0,
            pixels_per_mm=10.0,
        )
        matches = detect_fiducials_by_circles(img, config)

        assert len(matches) >= 2

    def test_no_circles_found(self):
        img = np.zeros((500, 700), dtype=np.uint8)
        config = FiducialConfig(method="circle", pixels_per_mm=10.0)
        matches = detect_fiducials_by_circles(img, config)
        assert len(matches) == 0


class TestAlignmentQuality:
    def test_good_quality_passes(self):
        result = AlignmentResult(
            transform_matrix=np.eye(3),
            aligned_image=np.zeros((100, 100, 3), dtype=np.uint8),
            quality_score=0.95,
            fiducial_positions=[(100, 100)],
            success=True,
        )
        assert check_alignment_quality(result) is True

    def test_low_quality_fails(self):
        result = AlignmentResult(
            transform_matrix=np.eye(3),
            aligned_image=np.zeros((100, 100, 3), dtype=np.uint8),
            quality_score=0.3,
            fiducial_positions=[],
            success=True,
        )
        assert check_alignment_quality(result) is False

    def test_failed_alignment_fails(self):
        result = AlignmentResult(
            transform_matrix=np.eye(3),
            aligned_image=np.zeros((100, 100, 3), dtype=np.uint8),
            quality_score=0.95,
            fiducial_positions=[],
            success=False,
        )
        assert check_alignment_quality(result) is False
