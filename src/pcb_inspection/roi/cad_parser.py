"""CAD file parsers for extracting component placement data.

Supports:
- CPL (Component Placement List) — CSV format, most common
- ODB++ / IPC-2581 — planned
- Gerber + BOM — planned
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from pcb_inspection.roi.models import CADComponent

logger = logging.getLogger(__name__)


def parse_cpl(
    file_path: str | Path,
    delimiter: str = ",",
    has_header: bool = True,
    column_map: dict[str, int] | None = None,
) -> list[CADComponent]:
    """Parse a Component Placement List (CPL) CSV file.

    Default column order: Designator, Package, X(mm), Y(mm), Rotation, Layer, Value

    Args:
        file_path: Path to CPL file.
        delimiter: CSV delimiter.
        has_header: Whether file has a header row.
        column_map: Custom column index mapping. Keys:
            "designator", "package", "x", "y", "rotation", "layer", "value"

    Returns:
        List of parsed components.
    """
    default_map = {
        "designator": 0,
        "package": 1,
        "x": 2,
        "y": 3,
        "rotation": 4,
        "layer": 5,
        "value": 6,
    }
    cmap = column_map or default_map

    components = []
    path = Path(file_path)

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)

        if has_header:
            header = next(reader, None)
            if header:
                cmap = _detect_columns(header, cmap)

        for row_num, row in enumerate(reader, start=2 if has_header else 1):
            if not row or row[0].startswith("#"):
                continue

            try:
                comp = CADComponent(
                    designator=_get_field(row, cmap, "designator", ""),
                    package=_get_field(row, cmap, "package", ""),
                    x_mm=float(_get_field(row, cmap, "x", "0")),
                    y_mm=float(_get_field(row, cmap, "y", "0")),
                    rotation=float(_get_field(row, cmap, "rotation", "0")),
                    layer=_get_field(row, cmap, "layer", "top"),
                    value=_get_field(row, cmap, "value", ""),
                )
                components.append(comp)
            except (ValueError, IndexError) as e:
                logger.warning("CPL parse error at row %d: %s", row_num, e)

    logger.info("Parsed %d components from %s", len(components), path.name)
    return components


def _get_field(
    row: list[str], cmap: dict[str, int], key: str, default: str
) -> str:
    """Safely get a field from a row using column map."""
    idx = cmap.get(key)
    if idx is None or idx >= len(row):
        return default
    return row[idx].strip()


def _detect_columns(
    header: list[str], fallback: dict[str, int]
) -> dict[str, int]:
    """Auto-detect column indices from header names."""
    aliases = {
        "designator": ["designator", "ref", "refdes", "reference"],
        "package": ["package", "footprint", "pattern"],
        "x": ["x", "x(mm)", "posx", "mid x"],
        "y": ["y", "y(mm)", "posy", "mid y"],
        "rotation": ["rotation", "rot", "angle"],
        "layer": ["layer", "side", "tb"],
        "value": ["value", "val", "comment"],
    }

    result = dict(fallback)
    header_lower = [h.strip().lower() for h in header]

    for field_name, candidates in aliases.items():
        for i, col in enumerate(header_lower):
            if col in candidates:
                result[field_name] = i
                break

    return result
