"""Core data types shared across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Severity(Enum):
    OK = "ok"
    WARNING = "warning"
    NG = "ng"


class InspectionType(Enum):
    REFERENCE = "reference"
    RULE_BASED = "rule_based"
    ANOMALY = "anomaly"
    BLOB = "blob"


@dataclass
class ComponentROI:
    """Single component region of interest."""

    component_id: str
    component_type: str
    bbox: tuple[int, int, int, int]  # (x, y, w, h) in aligned image
    rotation: float = 0.0  # degrees
    inspection_types: list[InspectionType] = field(default_factory=list)
    pad_regions: list[tuple[int, int, int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Output from the alignment module."""

    transform_matrix: np.ndarray  # 3x3
    aligned_image: np.ndarray
    quality_score: float  # 0.0~1.0
    fiducial_positions: list[tuple[float, float]]
    success: bool


@dataclass
class InspectionResult:
    """Single inspection result for one component + one inspection type."""

    inspection_type: InspectionType
    component_id: str
    severity: Severity
    score: float  # 0.0~1.0 normalized
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoardJudgment:
    """Final judgment for an entire board."""

    board_id: str
    overall: Severity
    component_results: dict[str, list[InspectionResult]]
    timestamp: str
    recipe_id: str
    alignment_quality: float
