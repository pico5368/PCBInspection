"""Step 1 PoC: End-to-end alignment + ROI pipeline on synthetic data.

Usage:
    python scripts/run_alignment_poc.py

Generates synthetic PCB → aligns → generates ROIs → runs basic inspections → visualizes.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pcb_inspection.alignment.fiducial import FiducialConfig
from pcb_inspection.alignment.registration import align_board
from pcb_inspection.alignment.quality import check_alignment_quality
from pcb_inspection.common.image_utils import crop_roi
from pcb_inspection.common.visualization import (
    draw_fiducials,
    draw_rois,
    draw_alignment_comparison,
    draw_inspection_results,
)
from pcb_inspection.common.types import InspectionType, Severity
from pcb_inspection.inspection.reference import ReferenceInspector
from pcb_inspection.inspection.rule_based import RuleBasedInspector
from pcb_inspection.inspection.blob import BlobInspector
from pcb_inspection.judgment.engine import judge_board
from pcb_inspection.roi.cad_parser import parse_cpl
from pcb_inspection.roi.roi_generator import generate_rois

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/poc_results")
SYNTHETIC_DIR = Path("data/synthetic")
PIXELS_PER_MM = 50.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 0: Generate synthetic data ──
    logger.info("=== Step 0: Generate Synthetic PCB ===")
    from generate_synthetic_pcb import generate_pcb_image, generate_test_image, save_cpl

    ref_img, components, fiducials = generate_pcb_image()
    test_img = generate_test_image(ref_img, offset_x=15, offset_y=-10, rotation_deg=1.5)

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SYNTHETIC_DIR / "reference.png"), ref_img)
    cv2.imwrite(str(SYNTHETIC_DIR / "test_offset.png"), test_img)
    save_cpl(components, SYNTHETIC_DIR / "placement.csv")

    logger.info("Reference: %s, %d components", ref_img.shape, len(components))
    logger.info("Test image: offset=(15, -10), rotation=1.5deg")

    # ── Step 1: Fiducial Detection + Alignment ──
    logger.info("=== Step 1: Alignment ===")

    # Extract fiducial template from reference image
    fid_x, fid_y = int(fiducials[0][0]), int(fiducials[0][1])
    fid_template = ref_img[fid_y - 30:fid_y + 30, fid_x - 30:fid_x + 30].copy()

    fid_config = FiducialConfig(
        method="template",
        match_threshold=0.6,
        expected_count=2,
        pixels_per_mm=PIXELS_PER_MM,
    )

    alignment = align_board(
        image=test_img,
        reference_fiducials=fiducials,
        fiducial_config=fid_config,
        fiducial_template=fid_template,
        output_size=(ref_img.shape[1], ref_img.shape[0]),
    )

    logger.info(
        "Alignment: success=%s, quality=%.3f, fiducials_found=%d",
        alignment.success, alignment.quality_score, len(alignment.fiducial_positions),
    )

    # Visualize fiducials
    vis_fid = draw_fiducials(test_img, alignment.fiducial_positions, color=(0, 255, 0))
    vis_fid = draw_fiducials(vis_fid, fiducials, color=(0, 0, 255), radius=25)
    cv2.imwrite(str(OUTPUT_DIR / "01_fiducials_detected.png"), vis_fid)

    # Visualize alignment
    vis_compare = draw_alignment_comparison(test_img, alignment.aligned_image, scale=0.5)
    cv2.imwrite(str(OUTPUT_DIR / "02_alignment_comparison.png"), vis_compare)

    if not check_alignment_quality(alignment):
        logger.error("Alignment quality too low! Aborting.")
        return

    # ── Step 2: ROI Generation ──
    logger.info("=== Step 2: ROI Generation ===")

    cad_components = parse_cpl(SYNTHETIC_DIR / "placement.csv")
    rois = generate_rois(
        cad_components,
        pixels_per_mm=PIXELS_PER_MM,
        image_size=(alignment.aligned_image.shape[1], alignment.aligned_image.shape[0]),
    )

    logger.info("Generated %d ROIs", len(rois))

    # Visualize ROIs
    vis_rois = draw_rois(alignment.aligned_image, rois)
    cv2.imwrite(str(OUTPUT_DIR / "03_rois.png"), vis_rois)

    # Save individual ROI crops
    roi_dir = OUTPUT_DIR / "roi_crops"
    roi_dir.mkdir(exist_ok=True)
    for roi in rois[:10]:  # first 10
        cropped = crop_roi(alignment.aligned_image, roi.bbox, padding=5)
        if cropped.size > 0:
            cv2.imwrite(str(roi_dir / f"{roi.component_id}_{roi.component_type}.png"), cropped)

    # ── Step 3: Basic Inspection ──
    logger.info("=== Step 3: Basic Inspection ===")

    ref_inspector = ReferenceInspector()
    rule_inspector = RuleBasedInspector()
    blob_inspector = BlobInspector()

    all_results = []
    for roi in rois:
        roi_image = crop_roi(alignment.aligned_image, roi.bbox, padding=5)
        ref_roi = crop_roi(ref_img, roi.bbox, padding=5)

        if roi_image.size == 0:
            continue

        for itype in roi.inspection_types:
            config = {"component_id": roi.component_id}

            if itype == InspectionType.REFERENCE:
                result = ref_inspector.inspect(roi_image, ref_roi, config)
            elif itype == InspectionType.RULE_BASED:
                result = rule_inspector.inspect(roi_image, ref_roi, config)
            elif itype == InspectionType.BLOB:
                result = blob_inspector.inspect(roi_image, ref_roi, config)
            else:
                continue

            all_results.append(result)

    # ── Step 4: Judgment ──
    logger.info("=== Step 4: Judgment ===")

    judgment = judge_board(
        board_id="synthetic_001",
        results=all_results,
        recipe_id="poc_test",
        alignment_quality=alignment.quality_score,
    )

    # Stats
    ng_components = [
        cid for cid, results in judgment.component_results.items()
        if any(r.severity == Severity.NG for r in results)
    ]
    warn_components = [
        cid for cid, results in judgment.component_results.items()
        if any(r.severity == Severity.WARNING for r in results)
        and cid not in ng_components
    ]
    ok_count = len(judgment.component_results) - len(ng_components) - len(warn_components)

    logger.info("Overall: %s", judgment.overall.value.upper())
    logger.info("  OK: %d, WARNING: %d, NG: %d", ok_count, len(warn_components), len(ng_components))

    # Visualize results
    vis_results = draw_inspection_results(alignment.aligned_image, rois, judgment)
    cv2.imwrite(str(OUTPUT_DIR / "04_inspection_results.png"), vis_results)

    # ── Summary ──
    logger.info("=== Results saved to %s ===", OUTPUT_DIR.resolve())
    logger.info("  01_fiducials_detected.png  — Fiducial detection visualization")
    logger.info("  02_alignment_comparison.png — Original vs aligned")
    logger.info("  03_rois.png                — ROI bounding boxes")
    logger.info("  04_inspection_results.png  — Inspection results overlay")
    logger.info("  roi_crops/                 — Individual ROI images")


if __name__ == "__main__":
    main()
