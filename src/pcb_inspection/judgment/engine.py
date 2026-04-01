"""Judgment engine: aggregate individual inspection results into board-level verdict."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pcb_inspection.common.types import (
    BoardJudgment,
    InspectionResult,
    Severity,
)

logger = logging.getLogger(__name__)


def judge_board(
    board_id: str,
    results: list[InspectionResult],
    recipe_id: str,
    alignment_quality: float,
    thresholds: dict[str, float] | None = None,
) -> BoardJudgment:
    """Aggregate component inspection results into a board-level judgment.

    Args:
        board_id: Unique board identifier.
        results: All inspection results for this board.
        recipe_id: Recipe used for inspection.
        alignment_quality: Alignment quality score.
        thresholds: Optional per-inspection-type threshold overrides.

    Returns:
        BoardJudgment with overall verdict and per-component details.
    """
    # Group results by component
    component_results: dict[str, list[InspectionResult]] = {}
    for r in results:
        component_results.setdefault(r.component_id, []).append(r)

    # Determine overall severity
    overall = Severity.OK
    for component_id, comp_results in component_results.items():
        for r in comp_results:
            if r.severity == Severity.NG:
                overall = Severity.NG
                break
            elif r.severity == Severity.WARNING and overall != Severity.NG:
                overall = Severity.WARNING
        if overall == Severity.NG:
            break

    judgment = BoardJudgment(
        board_id=board_id,
        overall=overall,
        component_results=component_results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        recipe_id=recipe_id,
        alignment_quality=alignment_quality,
    )

    ng_count = sum(
        1 for comp_results in component_results.values()
        for r in comp_results if r.severity == Severity.NG
    )
    logger.info(
        "Board %s: %s (%d NG items, %d components)",
        board_id, overall.value, ng_count, len(component_results),
    )

    return judgment
