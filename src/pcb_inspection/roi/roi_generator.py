"""Generate pixel-coordinate ROIs from CAD component data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pcb_inspection.common.types import ComponentROI, InspectionType
from pcb_inspection.roi.models import CADComponent

logger = logging.getLogger(__name__)

# Default ROI padding around component center (mm)
DEFAULT_PADDING_MM = 1.0

# Package type → inspection types mapping
PACKAGE_INSPECTION_MAP: dict[str, list[InspectionType]] = {
    "default": [
        InspectionType.REFERENCE,
        InspectionType.RULE_BASED,
        InspectionType.ANOMALY,
    ],
    "chip": [  # 0402, 0603, 0805, etc.
        InspectionType.REFERENCE,
        InspectionType.RULE_BASED,
        InspectionType.ANOMALY,
    ],
    "ic": [  # QFP, SOP, TSSOP, etc.
        InspectionType.REFERENCE,
        InspectionType.RULE_BASED,
        InspectionType.ANOMALY,
        InspectionType.BLOB,
    ],
    "connector": [
        InspectionType.REFERENCE,
        InspectionType.ANOMALY,
    ],
}

# Approximate package sizes (mm) — used when CAD doesn't provide dimensions
PACKAGE_SIZES: dict[str, tuple[float, float]] = {
    "0201": (0.6, 0.3),
    "0402": (1.0, 0.5),
    "0603": (1.6, 0.8),
    "0805": (2.0, 1.25),
    "1206": (3.2, 1.6),
    "1210": (3.2, 2.5),
    "SOT23": (3.0, 1.75),
    "SOT223": (6.5, 3.5),
    "SOP8": (5.0, 4.0),
    "TSSOP16": (5.0, 4.4),
    "QFP48": (9.0, 9.0),
    "QFP64": (12.0, 12.0),
    "QFP100": (14.0, 14.0),
    "QFN16": (4.0, 4.0),
    "QFN32": (5.0, 5.0),
}


def generate_rois(
    components: list[CADComponent],
    pixels_per_mm: float,
    image_size: tuple[int, int],
    origin_offset: tuple[float, float] = (0.0, 0.0),
    padding_mm: float = DEFAULT_PADDING_MM,
    package_sizes: dict[str, tuple[float, float]] | None = None,
) -> list[ComponentROI]:
    """Convert CAD components to pixel-coordinate ROIs.

    Args:
        components: Parsed CAD component list.
        pixels_per_mm: Image resolution.
        image_size: (width, height) of the aligned image.
        origin_offset: (x_mm, y_mm) offset of CAD origin from image origin.
        padding_mm: Extra padding around each component.
        package_sizes: Custom package size overrides.

    Returns:
        List of ComponentROI with pixel coordinates.
    """
    sizes = {**PACKAGE_SIZES, **(package_sizes or {})}
    img_w, img_h = image_size
    rois = []

    for comp in components:
        # Get package size
        pkg_w_mm, pkg_h_mm = _get_package_size(comp.package, sizes)
        pkg_w_mm += padding_mm * 2
        pkg_h_mm += padding_mm * 2

        # Convert mm to pixels
        cx_px = (comp.x_mm - origin_offset[0]) * pixels_per_mm
        cy_px = (comp.y_mm - origin_offset[1]) * pixels_per_mm
        w_px = int(pkg_w_mm * pixels_per_mm)
        h_px = int(pkg_h_mm * pixels_per_mm)

        # Bbox (x, y, w, h)
        x = int(cx_px - w_px / 2)
        y = int(cy_px - h_px / 2)

        # Clip to image bounds
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w_px = min(w_px, img_w - x)
        h_px = min(h_px, img_h - y)

        if w_px <= 0 or h_px <= 0:
            logger.warning("Component %s is outside image bounds", comp.designator)
            continue

        # Determine inspection types
        inspection_types = _get_inspection_types(comp.package)

        roi = ComponentROI(
            component_id=comp.designator,
            component_type=comp.package,
            bbox=(x, y, w_px, h_px),
            rotation=comp.rotation,
            inspection_types=inspection_types,
            metadata={
                "value": comp.value,
                "layer": comp.layer,
                "center_mm": (comp.x_mm, comp.y_mm),
            },
        )
        rois.append(roi)

    logger.info("Generated %d ROIs from %d components", len(rois), len(components))
    return rois


def _get_package_size(
    package: str, sizes: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Look up package dimensions. Returns (width_mm, height_mm)."""
    # Direct match
    pkg_upper = package.upper()
    if pkg_upper in sizes:
        return sizes[pkg_upper]

    # Try prefix match (e.g., "0603_RES" → "0603")
    for key in sizes:
        if pkg_upper.startswith(key):
            return sizes[key]

    # Default: 3x3mm
    logger.debug("Unknown package '%s', using default 3x3mm", package)
    return (3.0, 3.0)


def _get_inspection_types(package: str) -> list[InspectionType]:
    """Determine which inspections to run based on package type."""
    pkg_upper = package.upper()

    # Classify package category
    if any(pkg_upper.startswith(p) for p in ["0201", "0402", "0603", "0805", "1206", "1210"]):
        return list(PACKAGE_INSPECTION_MAP["chip"])
    elif any(pkg_upper.startswith(p) for p in ["QFP", "SOP", "TSSOP", "QFN", "BGA", "SOT"]):
        return list(PACKAGE_INSPECTION_MAP["ic"])
    elif any(kw in pkg_upper for kw in ["CONN", "HDR", "USB", "JACK"]):
        return list(PACKAGE_INSPECTION_MAP["connector"])

    return list(PACKAGE_INSPECTION_MAP["default"])
