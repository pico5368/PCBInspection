"""ROI-specific data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CADComponent:
    """Raw component data parsed from CAD file (CPL/ODB++/Gerber)."""

    designator: str  # e.g., "R1", "U3", "C15"
    package: str  # e.g., "0603", "QFP48", "SOT23"
    x_mm: float  # center X in mm
    y_mm: float  # center Y in mm
    rotation: float  # degrees
    layer: str = "top"  # "top" | "bottom"
    value: str = ""  # e.g., "10K", "100nF"
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class PadInfo:
    """Pad geometry for a component."""

    pad_id: str
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float
    shape: str = "rect"  # "rect" | "circle" | "oblong"
