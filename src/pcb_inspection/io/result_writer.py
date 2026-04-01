"""Result output: save inspection results to JSON files."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pcb_inspection.common.types import BoardJudgment

logger = logging.getLogger(__name__)


def save_judgment_json(judgment: BoardJudgment, output_dir: str | Path) -> Path:
    """Save a board judgment to a JSON file.

    Args:
        judgment: Board judgment to save.
        output_dir: Directory to write the file.

    Returns:
        Path to the saved JSON file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = f"{judgment.board_id}_{judgment.timestamp.replace(':', '-')}.json"
    filepath = out / filename

    data = {
        "board_id": judgment.board_id,
        "overall": judgment.overall.value,
        "recipe_id": judgment.recipe_id,
        "alignment_quality": judgment.alignment_quality,
        "timestamp": judgment.timestamp,
        "components": {},
    }

    for comp_id, results in judgment.component_results.items():
        data["components"][comp_id] = [
            {
                "type": r.inspection_type.value,
                "severity": r.severity.value,
                "score": round(r.score, 4),
                "detail": r.detail,
            }
            for r in results
        ]

    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved judgment to %s", filepath)
    return filepath
