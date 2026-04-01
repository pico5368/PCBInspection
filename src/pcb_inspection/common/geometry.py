"""Geometry utilities: coordinate transforms, distance, angle calculations."""

from __future__ import annotations

import cv2
import numpy as np


def compute_affine_transform(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray:
    """Compute affine transform matrix from matched point pairs.

    Args:
        src_points: Source points (N, 2).
        dst_points: Destination points (N, 2).

    Returns:
        3x3 affine transform matrix.
    """
    if len(src_points) < 3:
        # Use partial affine (translation + rotation + scale)
        mat, _ = cv2.estimateAffinePartial2D(src_points, dst_points)
    else:
        mat, _ = cv2.estimateAffine2D(src_points, dst_points)

    if mat is None:
        return np.eye(3, dtype=np.float64)

    # Convert 2x3 to 3x3
    result = np.eye(3, dtype=np.float64)
    result[:2, :] = mat
    return result


def apply_transform(
    image: np.ndarray,
    matrix: np.ndarray,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Apply affine transform to an image.

    Args:
        image: Input image.
        matrix: 3x3 transform matrix.
        output_size: (width, height). Defaults to input size.

    Returns:
        Transformed image.
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)
    return cv2.warpAffine(image, matrix[:2, :], output_size)


def point_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def angle_between_points(
    p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Angle in degrees from p1 to p2 (relative to horizontal)."""
    return float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Get center point of a bbox (x, y, w, h)."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)
