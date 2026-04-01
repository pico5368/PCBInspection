"""PCB image registration (alignment) using fiducial markers."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from pcb_inspection.alignment.fiducial import (
    FiducialConfig,
    FiducialMatch,
    detect_fiducials,
)
from pcb_inspection.common.geometry import compute_affine_transform, apply_transform
from pcb_inspection.common.types import AlignmentResult

logger = logging.getLogger(__name__)


def align_board(
    image: np.ndarray,
    reference_fiducials: list[tuple[float, float]],
    fiducial_config: FiducialConfig,
    fiducial_template: np.ndarray | None = None,
    output_size: tuple[int, int] | None = None,
) -> AlignmentResult:
    """Align a PCB image to reference coordinates using fiducial markers.

    Args:
        image: Input PCB image (BGR).
        reference_fiducials: Expected fiducial positions in output coordinates.
        fiducial_config: Fiducial detection parameters.
        fiducial_template: Template image for template matching method.
        output_size: (width, height) of output. Defaults to input size.

    Returns:
        AlignmentResult with aligned image and quality metrics.
    """
    expected_count = len(reference_fiducials)
    fiducial_config.expected_count = expected_count

    # Detect fiducials
    matches = detect_fiducials(image, fiducial_config, fiducial_template)

    if len(matches) < 2:
        logger.warning("Fiducial detection failed: found %d, need >= 2", len(matches))
        return AlignmentResult(
            transform_matrix=np.eye(3, dtype=np.float64),
            aligned_image=image,
            quality_score=0.0,
            fiducial_positions=[],
            success=False,
        )

    # Take best N matches
    best_matches = matches[:expected_count]
    detected_points = _order_fiducials(best_matches, reference_fiducials)

    src_pts = np.array(detected_points, dtype=np.float64)
    dst_pts = np.array(reference_fiducials, dtype=np.float64)

    # Compute transform
    matrix = compute_affine_transform(src_pts, dst_pts)

    # Apply transform
    aligned = apply_transform(image, matrix, output_size)

    # Compute quality score
    quality = _compute_alignment_quality(src_pts, dst_pts, matrix)

    logger.info(
        "Alignment: %d fiducials, quality=%.3f",
        len(best_matches),
        quality,
    )

    return AlignmentResult(
        transform_matrix=matrix,
        aligned_image=aligned,
        quality_score=quality,
        fiducial_positions=detected_points,
        success=True,
    )


def _order_fiducials(
    matches: list[FiducialMatch],
    reference: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Order detected fiducials to match reference order.

    Uses nearest-neighbor matching based on relative geometry.
    """
    if len(matches) <= 1:
        return [m.center for m in matches]

    detected = [m.center for m in matches]

    if len(detected) == len(reference):
        # Try all permutations for small count, nearest-neighbor for larger
        if len(detected) <= 4:
            from itertools import permutations

            best_perm = None
            best_dist = float("inf")
            for perm in permutations(range(len(detected))):
                total = sum(
                    np.hypot(
                        detected[perm[i]][0] - reference[i][0],
                        detected[perm[i]][1] - reference[i][1],
                    )
                    for i in range(len(reference))
                )
                if total < best_dist:
                    best_dist = total
                    best_perm = perm
            return [detected[i] for i in best_perm]

    # Fallback: sort by x then y (top-left first)
    return sorted(detected, key=lambda p: (p[1], p[0]))


def _compute_alignment_quality(
    src: np.ndarray,
    dst: np.ndarray,
    matrix: np.ndarray,
) -> float:
    """Compute alignment quality as reprojection error (0~1, higher is better)."""
    ones = np.ones((len(src), 1), dtype=np.float64)
    src_h = np.hstack([src, ones])  # (N, 3)
    projected = (matrix @ src_h.T).T[:, :2]

    errors = np.sqrt(np.sum((projected - dst) ** 2, axis=1))
    mean_error = float(np.mean(errors))

    # Convert pixel error to 0~1 score (0 error = 1.0, >10px error = 0.0)
    quality = max(0.0, 1.0 - mean_error / 10.0)
    return quality
