"""Step B: Camera capture → Inspection pipeline integration test.

Captures a live image from the CREVIS camera, then runs it through
the full InspectionPipeline (alignment → inspection → judgment).

Usage:
    python scripts/test_camera_pipeline.py --recipe configs/recipes/test_board.yaml
    python scripts/test_camera_pipeline.py --recipe configs/recipes/test_board.yaml --save
    python scripts/test_camera_pipeline.py --no-recipe   # run with dummy ROIs for quick test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcb_inspection.camera import CameraConfig, create_camera
from pcb_inspection.camera.crevis import CrevisCamera  # noqa: F401  type-only ref
from pcb_inspection.common.types import ComponentROI, InspectionType, Severity
from pcb_inspection.pipeline.runner import InspectionPipeline
from pcb_inspection.recipe.manager import Recipe, load_recipe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "captures"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Camera → Pipeline integration test")
    p.add_argument("--recipe", type=str, help="Recipe YAML path")
    p.add_argument("--no-recipe", action="store_true", help="Run with dummy ROIs (quick test)")
    p.add_argument("--save", action="store_true", help="Save captured image and results")
    p.add_argument("--exposure", type=float, default=3000.0, help="Exposure time (us)")
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument("--show", action="store_true", default=True, help="Display results visually")
    p.add_argument(
        "--backend",
        choices=("auto", "crevis", "mock"),
        default="auto",
        help="Camera backend (auto: crevis if SDK present else mock)",
    )
    p.add_argument("--mock-image", type=str, help="Path to image used by mock backend")
    return p.parse_args()


def make_dummy_recipe() -> Recipe:
    """Create a minimal recipe for quick camera test."""
    return Recipe(
        recipe_id="camera_test",
        product_name="Camera Test Board",
        pixels_per_mm=50.0,
        image_size=(5120, 5120),
        thresholds={"reference": 0.8, "anomaly": 0.5},
    )


def make_dummy_rois(image: np.ndarray) -> list[ComponentROI]:
    """Create grid-based dummy ROIs from image for testing."""
    h, w = image.shape[:2]
    rois = []

    # Divide image into a 4x4 grid of ROIs
    grid_rows, grid_cols = 4, 4
    roi_h, roi_w = h // grid_rows, w // grid_cols

    for r in range(grid_rows):
        for c in range(grid_cols):
            x = c * roi_w
            y = r * roi_h
            rois.append(
                ComponentROI(
                    component_id=f"ROI_R{r}C{c}",
                    component_type="test_region",
                    bbox=(x, y, roi_w, roi_h),
                    inspection_types=[InspectionType.REFERENCE],
                )
            )

    return rois


def visualize_results(
    image: np.ndarray,
    judgment: object,
    rois: list[ComponentROI],
) -> np.ndarray:
    """Draw inspection results on the image."""
    vis = image.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    color_map = {
        Severity.OK: (0, 200, 0),       # green
        Severity.WARNING: (0, 200, 255), # yellow
        Severity.NG: (0, 0, 255),        # red
    }

    for roi in rois:
        x, y, w, h = roi.bbox
        comp_results = judgment.component_results.get(roi.component_id, [])

        # Worst severity for this component
        severity = Severity.OK
        for r in comp_results:
            if r.severity == Severity.NG:
                severity = Severity.NG
                break
            elif r.severity == Severity.WARNING:
                severity = Severity.WARNING

        color = color_map.get(severity, (128, 128, 128))
        thickness = 3 if severity == Severity.NG else 1

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        label = f"{roi.component_id}: {severity.value}"
        cv2.putText(vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Overall verdict banner
    banner_color = color_map.get(judgment.overall, (128, 128, 128))
    cv2.rectangle(vis, (0, 0), (400, 50), banner_color, -1)
    cv2.putText(
        vis,
        f"VERDICT: {judgment.overall.value.upper()}",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    return vis


def save_results(image: np.ndarray, vis: np.ndarray, judgment: object) -> None:
    """Save captured image, visualization, and JSON results."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save raw capture
    raw_path = SAVE_DIR / f"pipeline_{ts}_raw.png"
    cv2.imwrite(str(raw_path), image)

    # Save visualization
    vis_path = RESULTS_DIR / f"pipeline_{ts}_result.png"
    cv2.imwrite(str(vis_path), vis)

    # Save JSON result
    result_data = {
        "board_id": judgment.board_id,
        "overall": judgment.overall.value,
        "recipe_id": judgment.recipe_id,
        "alignment_quality": judgment.alignment_quality,
        "timestamp": judgment.timestamp,
        "components": {},
    }
    for comp_id, results in judgment.component_results.items():
        result_data["components"][comp_id] = [
            {
                "type": r.inspection_type.value,
                "severity": r.severity.value,
                "score": r.score,
                "detail": r.detail,
            }
            for r in results
        ]

    json_path = RESULTS_DIR / f"pipeline_{ts}_result.json"
    json_path.write_text(json.dumps(result_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n  Raw image  : {raw_path}")
    print(f"  Visualized : {vis_path}")
    print(f"  JSON result: {json_path}")


def main() -> None:
    args = parse_args()

    # --- Camera setup ---
    config = CameraConfig(exposure_us=args.exposure)
    mock_kwargs: dict = {}
    if args.mock_image:
        mock_kwargs["image_path"] = args.mock_image
    cam = create_camera(
        backend=args.backend,
        config=config,
        device_index=args.device,
        **mock_kwargs,
    )

    try:
        cam.open()

        # --- Capture ---
        print("\n=== Capturing image from camera ===")
        image = cam.grab()
        print(f"  Captured: {image.shape}, dtype={image.dtype}")

        cam.close()

        # --- Recipe + ROIs ---
        if args.no_recipe or not args.recipe:
            recipe = make_dummy_recipe()
            rois = make_dummy_rois(image)
            print(f"\n  Using dummy recipe with {len(rois)} grid ROIs")
        else:
            recipe = load_recipe(args.recipe)
            # TODO: Load ROIs from CAD data via roi_generator
            rois = make_dummy_rois(image)
            print(f"\n  Loaded recipe: {recipe.recipe_id}")

        # --- Pipeline ---
        print("\n=== Running inspection pipeline ===")
        pipeline = InspectionPipeline(recipe)
        judgment = pipeline.run(image=image, rois=rois)

        # --- Results ---
        print(f"\n=== Results ===")
        print(f"  Board ID  : {judgment.board_id}")
        print(f"  Overall   : {judgment.overall.value.upper()}")
        print(f"  Alignment : {judgment.alignment_quality:.3f}")
        print(f"  Components: {len(judgment.component_results)}")

        ng_count = sum(
            1
            for comp_results in judgment.component_results.values()
            for r in comp_results
            if r.severity == Severity.NG
        )
        warn_count = sum(
            1
            for comp_results in judgment.component_results.values()
            for r in comp_results
            if r.severity == Severity.WARNING
        )
        print(f"  NG items  : {ng_count}")
        print(f"  Warnings  : {warn_count}")

        # --- Visualization ---
        vis = visualize_results(image, judgment, rois)

        if args.save:
            save_results(image, vis, judgment)

        if args.show:
            from pcb_inspection.camera.crevis import CrevisCamera  # noqa: already imported

            display = vis.copy()
            h, w = display.shape[:2]
            if max(h, w) > 1200:
                scale = 1200 / max(h, w)
                display = cv2.resize(display, (int(w * scale), int(h * scale)))

            cv2.imshow("Pipeline Result - Press any key to close", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    finally:
        cam.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
