"""Visualization utilities for debugging and PoC."""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from pcb_inspection.common.types import (
    AlignmentResult,
    BoardJudgment,
    ComponentROI,
    InspectionResult,
    Severity,
)

SEVERITY_COLORS = {
    Severity.OK: (0, 200, 0),       # green
    Severity.WARNING: (0, 200, 255), # yellow
    Severity.NG: (0, 0, 255),       # red
}


def draw_fiducials(
    image: np.ndarray,
    positions: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 255, 0),
    radius: int = 20,
) -> np.ndarray:
    """Draw fiducial markers on image."""
    vis = image.copy()
    for i, (x, y) in enumerate(positions):
        cx, cy = int(x), int(y)
        cv2.circle(vis, (cx, cy), radius, color, 2)
        cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, radius, 2)
        cv2.putText(vis, f"F{i}", (cx + radius + 5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


def draw_rois(
    image: np.ndarray,
    rois: list[ComponentROI],
    show_labels: bool = True,
) -> np.ndarray:
    """Draw ROI bounding boxes on image."""
    vis = image.copy()
    for roi in rois:
        x, y, w, h = roi.bbox
        color = (255, 200, 0)  # cyan
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        if show_labels:
            label = f"{roi.component_id} ({roi.component_type})"
            cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return vis


def draw_inspection_results(
    image: np.ndarray,
    rois: list[ComponentROI],
    judgment: BoardJudgment,
) -> np.ndarray:
    """Draw inspection results overlaid on image."""
    vis = image.copy()

    for roi in rois:
        x, y, w, h = roi.bbox
        results = judgment.component_results.get(roi.component_id, [])

        # Determine worst severity for this component
        worst = Severity.OK
        for r in results:
            if r.severity == Severity.NG:
                worst = Severity.NG
                break
            elif r.severity == Severity.WARNING:
                worst = Severity.WARNING

        color = SEVERITY_COLORS[worst]
        thickness = 3 if worst == Severity.NG else 2

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

        # Label
        label = f"{roi.component_id}: {worst.value}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Overall verdict
    overall_color = SEVERITY_COLORS[judgment.overall]
    cv2.putText(
        vis,
        f"BOARD: {judgment.overall.value.upper()}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        overall_color,
        3,
    )

    return vis


def draw_alignment_comparison(
    original: np.ndarray,
    aligned: np.ndarray,
    scale: float = 0.5,
) -> np.ndarray:
    """Side-by-side comparison of original and aligned images."""
    h, w = original.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)

    orig_small = cv2.resize(original, (new_w, new_h))
    aligned_small = cv2.resize(aligned, (new_w, new_h))

    # Add labels
    cv2.putText(orig_small, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(aligned_small, "Aligned", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return np.hstack([orig_small, aligned_small])
