"""Step 1 PoC (tuned): Lower noise + adjusted thresholds to validate pipeline works correctly."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pcb_inspection.alignment.fiducial import FiducialConfig
from pcb_inspection.alignment.registration import align_board
from pcb_inspection.alignment.quality import check_alignment_quality
from pcb_inspection.common.image_utils import crop_roi
from pcb_inspection.common.visualization import draw_inspection_results, draw_rois
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
PIXELS_PER_MM = 50.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from generate_synthetic_pcb import generate_pcb_image, generate_test_image, save_cpl

    # Case 1: Normal board (low noise, small offset — should be mostly OK)
    logger.info("=== Case 1: Normal board (small deviation) ===")
    ref_img, components, fiducials = generate_pcb_image()
    test_normal = generate_test_image(ref_img, offset_x=5, offset_y=-3, rotation_deg=0.3, noise_std=2.0)

    result_normal = run_inspection(ref_img, test_normal, components, fiducials, "normal_001",
                                    sim_threshold=0.6, offset_threshold=15.0)
    vis = draw_inspection_results(result_normal["aligned"], result_normal["rois"], result_normal["judgment"])
    cv2.imwrite(str(OUTPUT_DIR / "05_case1_normal.png"), vis)

    # Case 2: Defective board (component missing — blank region where component should be)
    logger.info("=== Case 2: Board with simulated defect ===")
    defect_img = ref_img.copy()
    # "Remove" a component by painting over it with PCB color
    comp = components[0]
    cx, cy = int(comp["x_mm"] * PIXELS_PER_MM), int(comp["y_mm"] * PIXELS_PER_MM)
    cv2.rectangle(defect_img, (cx - 30, cy - 20), (cx + 30, cy + 20), (30, 80, 40), -1)
    test_defect = generate_test_image(defect_img, offset_x=5, offset_y=-3, rotation_deg=0.3, noise_std=2.0)

    result_defect = run_inspection(ref_img, test_defect, components, fiducials, "defect_001",
                                    sim_threshold=0.6, offset_threshold=15.0)
    vis = draw_inspection_results(result_defect["aligned"], result_defect["rois"], result_defect["judgment"])
    cv2.imwrite(str(OUTPUT_DIR / "06_case2_defect.png"), vis)

    # Summary
    j_normal = result_normal["judgment"]
    j_defect = result_defect["judgment"]

    ng_normal = sum(1 for cid, rs in j_normal.component_results.items() if any(r.severity == Severity.NG for r in rs))
    ng_defect = sum(1 for cid, rs in j_defect.component_results.items() if any(r.severity == Severity.NG for r in rs))

    logger.info("=== Summary ===")
    logger.info("Case 1 (Normal):  %s — NG components: %d / %d",
                j_normal.overall.value, ng_normal, len(j_normal.component_results))
    logger.info("Case 2 (Defect):  %s — NG components: %d / %d",
                j_defect.overall.value, ng_defect, len(j_defect.component_results))


def run_inspection(ref_img, test_img, components, fiducials, board_id, sim_threshold, offset_threshold):
    """Run full inspection pipeline and return results."""
    from generate_synthetic_pcb import save_cpl
    from pathlib import Path
    import tempfile

    # Alignment
    fid_x, fid_y = int(fiducials[0][0]), int(fiducials[0][1])
    fid_template = ref_img[fid_y - 30:fid_y + 30, fid_x - 30:fid_x + 30].copy()

    alignment = align_board(
        image=test_img,
        reference_fiducials=fiducials,
        fiducial_config=FiducialConfig(method="template", match_threshold=0.6, expected_count=2, pixels_per_mm=PIXELS_PER_MM),
        fiducial_template=fid_template,
        output_size=(ref_img.shape[1], ref_img.shape[0]),
    )

    # ROI generation
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["Designator", "Package", "X(mm)", "Y(mm)", "Rotation", "Layer", "Value"])
        for c in components:
            writer.writerow([c["designator"], c["package"], c["x_mm"], c["y_mm"], c["rotation"], c["layer"], c["value"]])
        cpl_path = f.name

    cad_components = parse_cpl(cpl_path)
    rois = generate_rois(cad_components, PIXELS_PER_MM, (alignment.aligned_image.shape[1], alignment.aligned_image.shape[0]))

    # Inspection
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
            config = {
                "component_id": roi.component_id,
                "similarity_threshold": sim_threshold,
                "missing_threshold": 0.3,
                "max_offset_px": offset_threshold,
            }
            if itype == InspectionType.REFERENCE:
                all_results.append(ref_inspector.inspect(roi_image, ref_roi, config))
            elif itype == InspectionType.RULE_BASED:
                all_results.append(rule_inspector.inspect(roi_image, ref_roi, config))
            elif itype == InspectionType.BLOB:
                all_results.append(blob_inspector.inspect(roi_image, ref_roi, config))

    judgment = judge_board(board_id, all_results, "poc_tuned", alignment.quality_score)

    return {"aligned": alignment.aligned_image, "rois": rois, "judgment": judgment, "alignment": alignment}


if __name__ == "__main__":
    main()
