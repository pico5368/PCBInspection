"""Reference comparison inspection: component presence, polarity, wrong part."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from pcb_inspection.common.types import InspectionResult, InspectionType, Severity

logger = logging.getLogger(__name__)


class ReferenceInspector:
    """Compare ROI against golden reference image."""

    def load(self, model_path: str | None = None) -> None:
        pass

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        """Compare ROI image with reference.

        Config keys:
            component_id: str
            similarity_threshold: float — below this = NG (default 0.8)
            missing_threshold: float — below this = missing component (default 0.3)
        """
        component_id = config.get("component_id", "unknown")
        sim_threshold = config.get("similarity_threshold", 0.8)
        missing_threshold = config.get("missing_threshold", 0.3)

        if reference is None:
            return InspectionResult(
                inspection_type=InspectionType.REFERENCE,
                component_id=component_id,
                severity=Severity.WARNING,
                score=0.5,
                detail="No reference image",
            )

        similarity = _compute_similarity(roi_image, reference)

        if similarity < missing_threshold:
            severity = Severity.NG
            detail = f"Component likely missing (similarity={similarity:.3f})"
        elif similarity < sim_threshold:
            severity = Severity.NG
            detail = f"Component mismatch (similarity={similarity:.3f})"
        else:
            severity = Severity.OK
            detail = f"Match OK (similarity={similarity:.3f})"

        return InspectionResult(
            inspection_type=InspectionType.REFERENCE,
            component_id=component_id,
            severity=severity,
            score=similarity,
            detail=detail,
            metadata={"similarity": similarity},
        )


def _compute_similarity(image: np.ndarray, reference: np.ndarray) -> float:
    """Compute structural similarity between two images.

    Uses histogram comparison + template matching as a simple baseline.
    Can be upgraded to DINOv2/Siamese features later.
    """
    # Ensure same size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference

    if gray.shape != ref_gray.shape:
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))

    # Histogram similarity
    hist1 = cv2.calcHist([gray], [0], None, [64], [0, 256])
    hist2 = cv2.calcHist([ref_gray], [0], None, [64], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    hist_sim = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))

    # Structural similarity (normalized cross-correlation)
    result = cv2.matchTemplate(gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    ncc = float(result.max()) if result.size > 0 else 0.0

    # Combined score (weighted average)
    similarity = 0.4 * max(0, hist_sim) + 0.6 * max(0, ncc)
    return float(np.clip(similarity, 0.0, 1.0))
