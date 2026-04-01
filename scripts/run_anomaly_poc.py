"""Step 2 PoC: End-to-end anomaly detection pipeline.

1. Generate synthetic dataset (normal ROIs)
2. Train EfficientAD / PatchCore on normal data
3. Run inference on normal + defect images
4. Visualize anomaly heatmaps

Usage:
    python scripts/run_anomaly_poc.py
    python scripts/run_anomaly_poc.py --model patchcore
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/anomaly_poc")
PIXELS_PER_MM = 50.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="patchcore", choices=["efficient_ad", "patchcore", "padim"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing model")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Prepare dataset ──
    logger.info("=== Step 1: Prepare Dataset ===")
    dataset_dir = prepare_dataset()

    # Pick one component type to train on
    component_types = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    logger.info("Available component types: %s", component_types)

    # Use the type with most images
    target_type = max(component_types, key=lambda t: len(list((dataset_dir / t / "good").glob("*.png"))))
    target_dir = dataset_dir / target_type
    img_count = len(list((target_dir / "good").glob("*.png")))
    logger.info("Training on: %s (%d images)", target_type, img_count)

    # ── Step 2: Train ──
    model_dir = OUTPUT_DIR / "models" / target_type / args.model
    ckpt_path = find_checkpoint(model_dir)

    if args.skip_train and ckpt_path:
        logger.info("Skipping training, using existing model: %s", ckpt_path)
    else:
        logger.info("=== Step 2: Train %s ===", args.model)
        train_model(target_dir, args.model, model_dir, args.epochs)
        ckpt_path = find_checkpoint(model_dir)

    if ckpt_path is None:
        logger.error("No checkpoint found after training!")
        sys.exit(1)

    logger.info("Using checkpoint: %s", ckpt_path)

    # ── Step 3: Inference ──
    logger.info("=== Step 3: Inference ===")
    run_inference(target_dir, ckpt_path, target_type)

    logger.info("=== Done! Results in %s ===", OUTPUT_DIR)


def prepare_dataset() -> Path:
    """Generate synthetic dataset and return path."""
    from prepare_anomaly_dataset import prepare_synthetic_dataset
    dataset_dir = prepare_synthetic_dataset(Path("data/synthetic"), num_images=20)
    return dataset_dir


def train_model(dataset_dir: Path, model_name: str, output_dir: Path, max_epochs: int):
    """Train anomalib model."""
    from train_anomaly import train_anomaly_model
    train_anomaly_model(
        dataset_dir=dataset_dir,
        model_name=model_name,
        output_dir=output_dir,
        image_size=(256, 256),
        max_epochs=max_epochs,
    )


def find_checkpoint(model_dir: Path) -> Path | None:
    """Find the latest checkpoint in model directory."""
    if not model_dir.exists():
        return None

    # Look for .ckpt files recursively
    ckpts = sorted(model_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        return ckpts[0]

    return None


def run_inference(dataset_dir: Path, ckpt_path: Path, component_type: str):
    """Run inference on normal and defect images, visualize results."""
    from pcb_inspection.inspection.anomaly import AnomalyInspector

    inspector = AnomalyInspector()
    inspector.load(str(ckpt_path), image_size=(256, 256))

    if not inspector.is_loaded:
        logger.error("Failed to load model")
        return

    results_dir = OUTPUT_DIR / "inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Test on normal images
    normal_dir = dataset_dir / "good"
    normal_images = sorted(normal_dir.glob("*.png"))[:5]
    logger.info("Testing on %d normal images...", len(normal_images))

    normal_scores = []
    for img_path in normal_images:
        img = cv2.imread(str(img_path))
        result = inspector.inspect(img, None, {
            "component_id": img_path.stem,
            "anomaly_threshold": 0.5,
            "warning_threshold": 0.3,
        })
        normal_scores.append(result.metadata.get("anomaly_score", 0))
        logger.info("  %s: %s (anomaly=%.3f)", img_path.name, result.severity.value,
                     result.metadata.get("anomaly_score", 0))

    # Create synthetic defect ROIs
    logger.info("Testing on synthetic defect images...")
    defect_scores = []
    for i, img_path in enumerate(normal_images[:3]):
        img = cv2.imread(str(img_path))
        defect_img = create_synthetic_defect(img, defect_type=i % 3)
        cv2.imwrite(str(results_dir / f"defect_{i}.png"), defect_img)

        result = inspector.inspect(defect_img, None, {
            "component_id": f"defect_{i}",
            "anomaly_threshold": 0.5,
            "warning_threshold": 0.3,
        })
        defect_scores.append(result.metadata.get("anomaly_score", 0))
        logger.info("  defect_%d: %s (anomaly=%.3f)", i, result.severity.value,
                     result.metadata.get("anomaly_score", 0))

    # Summary
    logger.info("=== Score Summary ===")
    if normal_scores:
        logger.info("Normal:  mean=%.3f, max=%.3f", np.mean(normal_scores), np.max(normal_scores))
    if defect_scores:
        logger.info("Defect:  mean=%.3f, min=%.3f", np.mean(defect_scores), np.min(defect_scores))

    if normal_scores and defect_scores:
        separation = np.min(defect_scores) - np.max(normal_scores)
        logger.info("Score gap (defect_min - normal_max): %.3f %s",
                     separation, "(good separation)" if separation > 0 else "(overlap — needs tuning)")


def create_synthetic_defect(image: np.ndarray, defect_type: int = 0) -> np.ndarray:
    """Create a synthetic defect on an ROI image."""
    defect = image.copy()
    h, w = defect.shape[:2]

    if defect_type == 0:
        # Solder bridge: bright horizontal line
        y = h // 2
        cv2.line(defect, (w // 4, y), (3 * w // 4, y), (200, 210, 200), 3)
    elif defect_type == 1:
        # Missing solder: dark patch
        cx, cy = w // 3, h // 2
        cv2.circle(defect, (cx, cy), w // 6, (20, 30, 20), -1)
    elif defect_type == 2:
        # Solder ball: bright spots
        for _ in range(5):
            x = np.random.randint(w // 4, 3 * w // 4)
            y = np.random.randint(h // 4, 3 * h // 4)
            cv2.circle(defect, (x, y), np.random.randint(2, 5), (210, 220, 210), -1)

    return defect


if __name__ == "__main__":
    main()
