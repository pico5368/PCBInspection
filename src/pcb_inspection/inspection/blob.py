"""Blob detection for solder balls and foreign materials."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from pcb_inspection.common.types import InspectionResult, InspectionType, Severity

logger = logging.getLogger(__name__)


class BlobInspector:
    """Detect solder balls and foreign material using blob analysis."""

    def load(self, model_path: str | None = None) -> None:
        pass

    def inspect(
        self,
        roi_image: np.ndarray,
        reference: np.ndarray | None,
        config: dict[str, Any],
    ) -> InspectionResult:
        """Detect blobs (solder balls, foreign material) in ROI.

        Config keys:
            component_id: str
            min_blob_area: int — minimum blob area in pixels (default 10)
            max_blob_count: int — max allowed blobs before NG (default 0)
            diff_threshold: int — difference threshold for foreground (default 30)
        """
        component_id = config.get("component_id", "unknown")
        min_area = config.get("min_blob_area", 10)
        max_count = config.get("max_blob_count", 0)
        diff_threshold = config.get("diff_threshold", 30)

        if reference is None:
            # Without reference, use simple blob detection
            blobs = _detect_blobs_direct(roi_image, min_area)
        else:
            blobs = _detect_blobs_by_diff(roi_image, reference, min_area, diff_threshold)

        blob_count = len(blobs)

        if blob_count > max_count:
            severity = Severity.NG
        elif blob_count > 0:
            severity = Severity.WARNING
        else:
            severity = Severity.OK

        score = max(0.0, 1.0 - blob_count / max(1, max_count + 3))

        return InspectionResult(
            inspection_type=InspectionType.BLOB,
            component_id=component_id,
            severity=severity,
            score=score,
            detail=f"Detected {blob_count} blob(s)",
            metadata={"blob_count": blob_count, "blobs": blobs},
        )


def _detect_blobs_by_diff(
    image: np.ndarray,
    reference: np.ndarray,
    min_area: int,
    threshold: int,
) -> list[dict]:
    """Detect blobs by differencing with reference image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference

    if gray.shape != ref_gray.shape:
        ref_gray = cv2.resize(ref_gray, (gray.shape[1], gray.shape[0]))

    diff = cv2.absdiff(gray, ref_gray)
    _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append({"bbox": (x, y, w, h), "area": int(area)})

    return blobs


def _detect_blobs_direct(image: np.ndarray, min_area: int) -> list[dict]:
    """Detect bright/dark blobs without reference."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Adaptive threshold to find anomalous regions
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    diff = cv2.absdiff(gray, blurred)
    _, binary = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            blobs.append({"bbox": (x, y, w, h), "area": int(area)})

    return blobs
