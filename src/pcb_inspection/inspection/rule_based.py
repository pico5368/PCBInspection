"""Rule-based inspection: position offset, rotation error."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from pcb_inspection.common.types import InspectionResult, InspectionType, Severity

logger = logging.getLogger(__name__)


class RuleBasedInspector:
    """Check position offset and rotation using template matching."""

    def load(self, model_path: str | None = None) -> None:
        pass  # No model needed

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        """Check position and rotation against reference.

        Config keys:
            component_id: str
            max_offset_px: float — maximum allowed center offset in pixels
            max_rotation_deg: float — maximum allowed rotation error
        """
        component_id = config.get("component_id", "unknown")
        max_offset = config.get("max_offset_px", 10.0)
        max_rotation = config.get("max_rotation_deg", 5.0)

        if reference is None:
            return InspectionResult(
                inspection_type=InspectionType.RULE_BASED,
                component_id=component_id,
                severity=Severity.WARNING,
                score=0.5,
                detail="No reference image available",
            )

        offset, angle, match_score = _measure_offset(roi_image, reference)

        # Evaluate
        offset_ratio = offset / max_offset if max_offset > 0 else 0
        angle_ratio = abs(angle) / max_rotation if max_rotation > 0 else 0
        worst_ratio = max(offset_ratio, angle_ratio)

        if worst_ratio > 1.0:
            severity = Severity.NG
        elif worst_ratio > 0.7:
            severity = Severity.WARNING
        else:
            severity = Severity.OK

        score = max(0.0, 1.0 - worst_ratio)

        return InspectionResult(
            inspection_type=InspectionType.RULE_BASED,
            component_id=component_id,
            severity=severity,
            score=score,
            detail=f"offset={offset:.1f}px, angle={angle:.1f}deg, match={match_score:.3f}",
            metadata={
                "offset_px": offset,
                "angle_deg": angle,
                "match_score": match_score,
            },
        )


def _measure_offset(
    image: np.ndarray, reference: np.ndarray
) -> tuple[float, float, float]:
    """Measure position offset and rotation between image and reference.

    Returns:
        (offset_pixels, rotation_degrees, match_confidence)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference

    # Resize reference to match image if needed
    if gray.shape != ref_gray.shape:
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))

    # Phase correlation for sub-pixel offset
    shift, response = cv2.phaseCorrelate(
        gray.astype(np.float64),
        ref_gray.astype(np.float64),
    )
    offset = np.hypot(shift[0], shift[1])

    # Template matching confidence
    result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    match_score = float(result.max()) if result.size > 0 else 0.0

    # Rotation estimation via feature matching
    angle = _estimate_rotation(gray, ref_gray)

    return float(offset), angle, match_score


def _estimate_rotation(image: np.ndarray, reference: np.ndarray) -> float:
    """Estimate rotation difference using ORB features."""
    orb = cv2.ORB_create(nfeatures=200)

    kp1, des1 = orb.detectAndCompute(reference, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) < 4:
        return 0.0

    matches = sorted(matches, key=lambda m: m.distance)[:20]

    # Extract matched points
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float64)
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float64)

    # Estimate affine to extract rotation
    mat, _ = cv2.estimateAffinePartial2D(pts1, pts2)
    if mat is None:
        return 0.0

    angle = np.degrees(np.arctan2(mat[1, 0], mat[0, 0]))
    return float(angle)
