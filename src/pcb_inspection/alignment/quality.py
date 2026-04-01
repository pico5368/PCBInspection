"""Alignment quality assessment."""

from __future__ import annotations

from pcb_inspection.common.types import AlignmentResult

# Minimum quality score to consider alignment successful
DEFAULT_QUALITY_THRESHOLD = 0.7


def check_alignment_quality(
    result: AlignmentResult,
    threshold: float = DEFAULT_QUALITY_THRESHOLD,
) -> bool:
    """Check if alignment quality meets the threshold.

    Args:
        result: Alignment result to evaluate.
        threshold: Minimum acceptable quality score.

    Returns:
        True if alignment is acceptable.
    """
    return result.success and result.quality_score >= threshold
