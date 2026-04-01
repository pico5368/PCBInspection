"""Pipeline runner: orchestrates the full inspection flow for one board."""

from __future__ import annotations

import logging
import uuid
from typing import Any

import numpy as np

from pcb_inspection.alignment.fiducial import FiducialConfig
from pcb_inspection.alignment.quality import check_alignment_quality
from pcb_inspection.alignment.registration import align_board
from pcb_inspection.common.image_utils import crop_roi
from pcb_inspection.common.types import (
    AlignmentResult,
    BoardJudgment,
    ComponentROI,
    InspectionResult,
    InspectionType,
    Severity,
)
from pcb_inspection.inspection.anomaly import AnomalyInspector
from pcb_inspection.inspection.blob import BlobInspector
from pcb_inspection.inspection.reference import ReferenceInspector
from pcb_inspection.inspection.rule_based import RuleBasedInspector
from pcb_inspection.judgment.engine import judge_board
from pcb_inspection.recipe.manager import Recipe

logger = logging.getLogger(__name__)


class InspectionPipeline:
    """Full PCB inspection pipeline."""

    def __init__(self, recipe: Recipe) -> None:
        self.recipe = recipe

        # Initialize inspectors
        self._inspectors: dict[InspectionType, Any] = {
            InspectionType.REFERENCE: ReferenceInspector(),
            InspectionType.RULE_BASED: RuleBasedInspector(),
            InspectionType.ANOMALY: AnomalyInspector(),
            InspectionType.BLOB: BlobInspector(),
        }

        # Golden images cache: component_id -> image
        self._golden_cache: dict[str, np.ndarray] = {}

    def load_models(self) -> None:
        """Load all models specified in the recipe."""
        for itype_str, model_path in self.recipe.model_paths.items():
            try:
                itype = InspectionType(itype_str)
                inspector = self._inspectors.get(itype)
                if inspector:
                    inspector.load(model_path)
            except (ValueError, KeyError):
                logger.warning("Unknown inspection type in model_paths: %s", itype_str)

    def run(
        self,
        image: np.ndarray,
        rois: list[ComponentROI],
        fiducial_template: np.ndarray | None = None,
        board_id: str | None = None,
    ) -> BoardJudgment:
        """Run the full inspection pipeline on one board image.

        Args:
            image: Raw PCB image (BGR).
            rois: Pre-generated ROI list (from CAD data).
            fiducial_template: Template for fiducial detection.
            board_id: Optional board identifier.

        Returns:
            BoardJudgment with overall and per-component results.
        """
        if board_id is None:
            board_id = uuid.uuid4().hex[:12]

        # Step 1: Alignment
        fid_config = FiducialConfig(**self.recipe.fiducial_config)
        ref_fiducials = self.recipe.fiducial_config.get("reference_positions", [])

        if ref_fiducials:
            alignment = align_board(
                image, ref_fiducials, fid_config, fiducial_template
            )
        else:
            # No fiducial config — skip alignment
            alignment = AlignmentResult(
                transform_matrix=np.eye(3),
                aligned_image=image,
                quality_score=1.0,
                fiducial_positions=[],
                success=True,
            )

        # Check alignment quality
        if not check_alignment_quality(alignment):
            logger.warning("Alignment failed for board %s", board_id)
            return BoardJudgment(
                board_id=board_id,
                overall=Severity.NG,
                component_results={},
                timestamp="",
                recipe_id=self.recipe.recipe_id,
                alignment_quality=alignment.quality_score,
            )

        # Step 2: Run inspections per ROI
        all_results: list[InspectionResult] = []

        for roi in rois:
            roi_image = crop_roi(alignment.aligned_image, roi.bbox, padding=5)
            if roi_image.size == 0:
                continue

            golden = self._get_golden(roi.component_id)

            for itype in roi.inspection_types:
                inspector = self._inspectors.get(itype)
                if inspector is None:
                    continue

                config = self._build_inspection_config(roi, itype)
                result = inspector.inspect(roi_image, golden, config)
                all_results.append(result)

        # Step 3: Judgment
        return judge_board(
            board_id=board_id,
            results=all_results,
            recipe_id=self.recipe.recipe_id,
            alignment_quality=alignment.quality_score,
        )

    def _get_golden(self, component_id: str) -> np.ndarray | None:
        """Get golden reference image for a component."""
        return self._golden_cache.get(component_id)

    def set_golden(self, component_id: str, image: np.ndarray) -> None:
        """Register a golden reference image for a component."""
        self._golden_cache[component_id] = image

    def _build_inspection_config(
        self, roi: ComponentROI, itype: InspectionType
    ) -> dict[str, Any]:
        """Build config dict for a specific inspection."""
        config: dict[str, Any] = {"component_id": roi.component_id}

        # Global thresholds
        if itype.value in self.recipe.thresholds:
            config["anomaly_threshold"] = self.recipe.thresholds[itype.value]

        # Component-level overrides
        overrides = self.recipe.component_overrides.get(roi.component_id, {})
        config.update(overrides)

        return config
