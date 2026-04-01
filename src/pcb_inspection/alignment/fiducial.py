"""Fiducial marker detection for PCB alignment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FiducialConfig:
    """Configuration for fiducial detection."""

    method: str = "template"  # "template" | "circle" | "blob"
    template_path: str | None = None
    min_diameter: float = 0.5  # mm
    max_diameter: float = 3.0  # mm
    pixels_per_mm: float = 50.0
    match_threshold: float = 0.7
    expected_count: int = 2


@dataclass
class FiducialMatch:
    """A detected fiducial marker."""

    center: tuple[float, float]
    confidence: float
    radius: float


def detect_fiducials_by_template(
    image: np.ndarray,
    template: np.ndarray,
    config: FiducialConfig,
) -> list[FiducialMatch]:
    """Detect fiducials using template matching.

    Args:
        image: Grayscale input image.
        template: Grayscale fiducial template.
        config: Detection parameters.

    Returns:
        List of detected fiducials sorted by confidence.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape[:2]

    matches = []
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < config.match_threshold:
            break

        cx = max_loc[0] + tw / 2
        cy = max_loc[1] + th / 2
        matches.append(FiducialMatch(
            center=(cx, cy),
            confidence=float(max_val),
            radius=max(tw, th) / 2,
        ))

        # Suppress region around found match
        x1 = max(0, max_loc[0] - tw)
        y1 = max(0, max_loc[1] - th)
        x2 = min(result.shape[1], max_loc[0] + tw)
        y2 = min(result.shape[0], max_loc[1] + th)
        result[y1:y2, x1:x2] = 0

        if len(matches) >= config.expected_count * 2:
            break

    return sorted(matches, key=lambda m: m.confidence, reverse=True)


def detect_fiducials_by_circles(
    image: np.ndarray,
    config: FiducialConfig,
) -> list[FiducialMatch]:
    """Detect circular fiducials using HoughCircles.

    Args:
        image: Grayscale input image.
        config: Detection parameters.

    Returns:
        List of detected fiducials.
    """
    min_r = int(config.min_diameter * config.pixels_per_mm / 2)
    max_r = int(config.max_diameter * config.pixels_per_mm / 2)

    blurred = cv2.GaussianBlur(image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_r * 4,
        param1=100,
        param2=30,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if circles is None:
        return []

    matches = []
    for c in circles[0]:
        matches.append(FiducialMatch(
            center=(float(c[0]), float(c[1])),
            confidence=1.0,  # HoughCircles doesn't provide confidence
            radius=float(c[2]),
        ))

    return matches


def detect_fiducials(
    image: np.ndarray,
    config: FiducialConfig,
    template: np.ndarray | None = None,
) -> list[FiducialMatch]:
    """Detect fiducial markers using configured method.

    Args:
        image: Input image (BGR or grayscale).
        config: Detection configuration.
        template: Required if method is "template".

    Returns:
        List of detected fiducials, sorted by confidence.
    """
    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if config.method == "template" and template is not None:
        tpl_gray = template if len(template.shape) == 2 else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        return detect_fiducials_by_template(gray, tpl_gray, config)
    elif config.method == "circle":
        return detect_fiducials_by_circles(gray, config)
    else:
        # Fallback: try circle detection
        return detect_fiducials_by_circles(gray, config)
