"""Prepare ROI crops as anomalib-compatible dataset.

anomalib expects a folder structure:
    dataset_root/
    ├── good/          # normal images for training
    │   ├── 000.png
    │   └── ...
    └── defect/        # (optional) defect images for evaluation
        ├── 000.png
        └── ...

This script:
1. Takes a set of golden/normal PCB images
2. Runs alignment + ROI extraction
3. Saves ROI crops grouped by component type
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pcb_inspection.alignment.fiducial import FiducialConfig
from pcb_inspection.alignment.registration import align_board
from pcb_inspection.common.image_utils import crop_roi
from pcb_inspection.roi.cad_parser import parse_cpl
from pcb_inspection.roi.roi_generator import generate_rois

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset_from_images(
    image_paths: list[Path],
    cpl_path: Path,
    fiducials: list[tuple[float, float]],
    fiducial_template: np.ndarray,
    output_dir: Path,
    pixels_per_mm: float = 50.0,
    target_size: tuple[int, int] = (256, 256),
    group_by: str = "component_type",  # "component_type" | "component_id"
):
    """Extract ROI crops from multiple images and organize for anomalib training.

    Args:
        image_paths: List of normal (good) PCB image paths.
        cpl_path: Path to CPL file.
        fiducials: Reference fiducial positions.
        fiducial_template: Fiducial template image.
        output_dir: Root output directory.
        pixels_per_mm: Image resolution.
        target_size: Resize ROIs to this size (w, h).
        group_by: Group ROIs by "component_type" or "component_id".
    """
    fid_config = FiducialConfig(
        method="template",
        match_threshold=0.6,
        expected_count=len(fiducials),
        pixels_per_mm=pixels_per_mm,
    )

    cad_components = parse_cpl(cpl_path)

    total_crops = 0

    for img_idx, img_path in enumerate(image_paths):
        logger.info("Processing image %d/%d: %s", img_idx + 1, len(image_paths), img_path.name)

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Failed to read: %s", img_path)
            continue

        # Align
        alignment = align_board(image, fiducials, fid_config, fiducial_template)
        if not alignment.success or alignment.quality_score < 0.7:
            logger.warning("Alignment failed for %s (quality=%.3f)", img_path.name, alignment.quality_score)
            continue

        # Generate ROIs
        rois = generate_rois(
            cad_components, pixels_per_mm,
            (alignment.aligned_image.shape[1], alignment.aligned_image.shape[0]),
        )

        # Extract and save crops
        for roi in rois:
            cropped = crop_roi(alignment.aligned_image, roi.bbox, padding=5)
            if cropped.size == 0:
                continue

            # Resize to uniform size
            resized = cv2.resize(cropped, target_size)

            # Determine group key
            if group_by == "component_type":
                group_key = roi.component_type.upper()
            else:
                group_key = roi.component_id

            # Save
            group_dir = output_dir / group_key / "good"
            group_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{roi.component_id}_img{img_idx:04d}.png"
            cv2.imwrite(str(group_dir / filename), resized)
            total_crops += 1

    logger.info("Total crops saved: %d in %s", total_crops, output_dir)


def prepare_synthetic_dataset(output_dir: Path, num_images: int = 20):
    """Generate synthetic dataset for testing anomalib integration."""
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_synthetic_pcb import generate_pcb_image, generate_test_image, save_cpl

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate reference
    ref_img, components, fiducials = generate_pcb_image(seed=42)
    cpl_path = output_dir / "placement.csv"
    save_cpl(components, cpl_path)

    # Fiducial template
    fid_x, fid_y = int(fiducials[0][0]), int(fiducials[0][1])
    fid_template = ref_img[fid_y - 30:fid_y + 30, fid_x - 30:fid_x + 30].copy()

    # Generate multiple "normal" images with slight variations
    image_paths = []
    images_dir = output_dir / "raw_images"
    images_dir.mkdir(exist_ok=True)

    for i in range(num_images):
        import random
        rng = random.Random(i + 100)
        test_img = generate_test_image(
            ref_img,
            offset_x=rng.uniform(-3, 3),
            offset_y=rng.uniform(-3, 3),
            rotation_deg=rng.uniform(-0.5, 0.5),
            noise_std=rng.uniform(1.0, 3.0),
        )
        path = images_dir / f"normal_{i:04d}.png"
        cv2.imwrite(str(path), test_img)
        image_paths.append(path)

    logger.info("Generated %d synthetic normal images", num_images)

    # Extract ROI crops
    dataset_dir = output_dir / "anomaly_dataset"
    prepare_dataset_from_images(
        image_paths=image_paths,
        cpl_path=cpl_path,
        fiducials=fiducials,
        fiducial_template=fid_template,
        output_dir=dataset_dir,
        target_size=(256, 256),
        group_by="component_type",
    )

    # Also generate a few "defect" images for evaluation
    defect_images_dir = output_dir / "raw_defect_images"
    defect_images_dir.mkdir(exist_ok=True)

    for i in range(3):
        defect_img = ref_img.copy()
        # Add random bright spots (solder balls)
        for _ in range(5):
            dx = random.Random(i * 100 + _).randint(200, 1800)
            dy = random.Random(i * 100 + _ + 50).randint(200, 1300)
            cv2.circle(defect_img, (dx, dy), random.Random(i + _).randint(3, 8), (200, 210, 200), -1)

        test_defect = generate_test_image(
            defect_img,
            offset_x=random.Random(i + 200).uniform(-3, 3),
            offset_y=random.Random(i + 201).uniform(-3, 3),
            rotation_deg=random.Random(i + 202).uniform(-0.5, 0.5),
            noise_std=2.0,
        )
        path = defect_images_dir / f"defect_{i:04d}.png"
        cv2.imwrite(str(path), test_defect)

    logger.info("Generated %d defect images for evaluation", 3)

    return dataset_dir


if __name__ == "__main__":
    output = Path("data/synthetic")
    dataset_dir = prepare_synthetic_dataset(output, num_images=20)
    print(f"\nDataset ready at: {dataset_dir}")
