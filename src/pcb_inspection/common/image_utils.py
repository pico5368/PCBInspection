"""Image utility functions: crop, resize, normalize."""

from __future__ import annotations

import cv2
import numpy as np


def crop_roi(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 0,
) -> np.ndarray:
    """Crop a region from an image with optional padding.

    Args:
        image: Source image.
        bbox: (x, y, w, h) region.
        padding: Extra pixels around the bbox.

    Returns:
        Cropped image. Returns empty array if bbox is out of bounds.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(w, x + bw + padding)
    y2 = min(h, y + bh + padding)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)

    return image[y1:y2, x1:x2].copy()


def resize_to(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize image to exact (width, height)."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Histogram equalization for brightness normalization."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.equalizeHist(image)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
