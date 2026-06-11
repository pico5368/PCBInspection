"""Train anomalib models on ROI crop datasets.

Usage:
    python scripts/train_anomaly.py --dataset data/synthetic/anomaly_dataset/0603
    python scripts/train_anomaly.py --dataset data/synthetic/anomaly_dataset/0603 --model patchcore
    python scripts/train_anomaly.py --dataset data/synthetic/anomaly_dataset/0603 --model efficient_ad

Supports: EfficientAD (default), PatchCore, PaDiM
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_anomaly_model(
    dataset_dir: Path,
    model_name: str = "efficient_ad",
    output_dir: Path | None = None,
    image_size: tuple[int, int] = (256, 256),
    max_epochs: int = 50,
):
    """Train an anomalib model on a Folder dataset.

    Args:
        dataset_dir: Directory with good/ subfolder (and optionally defect/).
        model_name: "efficient_ad", "patchcore", or "padim".
        output_dir: Where to save the trained model.
        image_size: Input image size.
        max_epochs: Maximum training epochs (ignored for memory-bank models).
    """
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.deploy import ExportType
    except ImportError:
        logger.error("anomalib not installed. Run: pip install anomalib")
        sys.exit(1)

    if output_dir is None:
        output_dir = Path("data/models") / dataset_dir.name / model_name

    # ── Dataset ──
    normal_dir = dataset_dir / "good"
    abnormal_dir = dataset_dir / "defect" if (dataset_dir / "defect").exists() else None

    if not normal_dir.exists():
        logger.error("No 'good' directory found in %s", dataset_dir)
        sys.exit(1)

    normal_count = len(list(normal_dir.glob("*.png")))
    logger.info("Dataset: %s (%d normal images)", dataset_dir.name, normal_count)

    if normal_count < 5:
        logger.error("Need at least 5 normal images, got %d", normal_count)
        sys.exit(1)

    # Split: 80% train, 20% val
    datamodule = Folder(
        name=dataset_dir.name,
        root=dataset_dir,
        normal_dir="good",
        abnormal_dir="defect" if abnormal_dir else None,
        train_batch_size=min(8, normal_count),
        eval_batch_size=min(8, normal_count),
        normal_split_ratio=0.8,
    )

    # ── Model ──
    model = _create_model(model_name)
    logger.info("Model: %s", model_name)

    # ── Train ──
    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir=str(output_dir),
    )

    logger.info("Starting training...")
    engine.fit(model=model, datamodule=datamodule)

    # ── Test (if abnormal data exists) ──
    if abnormal_dir and abnormal_dir.exists():
        logger.info("Running evaluation on defect data...")
        engine.test(model=model, datamodule=datamodule)

    # ── Export ──
    logger.info("Exporting model...")
    try:
        export_path = output_dir / "weights" / "exported"
        engine.export(
            model=model,
            export_type=ExportType.ONNX,
            export_root=str(export_path),
        )
        logger.info("ONNX model exported to: %s", export_path)
    except Exception as exc:
        logger.warning("ONNX export failed (non-critical), checkpoint saved instead: %s", exc)

    logger.info("Training complete. Output: %s", output_dir)
    return output_dir


def _create_model(model_name: str):
    """Create an anomalib model by name."""
    model_name = model_name.lower().replace("-", "_")

    if model_name == "efficient_ad":
        from anomalib.models import EfficientAd
        return EfficientAd(
            teacher_out_channels=384,
            model_size="small",  # "small" or "medium"
        )
    elif model_name == "patchcore":
        from anomalib.models import Patchcore
        return Patchcore(
            backbone="wide_resnet50_2",
            layers_to_extract=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
    elif model_name == "padim":
        from anomalib.models import Padim
        return Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Use: efficient_ad, patchcore, padim")


def main():
    parser = argparse.ArgumentParser(description="Train anomalib model on ROI dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset directory (with good/ subfolder)")
    parser.add_argument("--model", type=str, default="efficient_ad", choices=["efficient_ad", "patchcore", "padim"])
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output) if args.output else None

    train_anomaly_model(
        dataset_dir=dataset_dir,
        model_name=args.model,
        output_dir=output_dir,
        image_size=tuple(args.image_size),
        max_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
