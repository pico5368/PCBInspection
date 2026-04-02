"""Train anomalib on the transistor dataset (MVTec format).

Usage:
    python scripts/train_transistor.py --model patchcore
    python scripts/train_transistor.py --model efficient_ad
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATASET_ROOT = Path("transistor")
OUTPUT_ROOT = Path("data/models/transistor")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="patchcore", choices=["efficient_ad", "patchcore", "padim"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1, help="Only for EfficientAD (PatchCore ignores)")
    args = parser.parse_args()

    from anomalib.data import MVTecAD
    from anomalib.engine import Engine

    output_dir = OUTPUT_ROOT / args.model
    img_size = (args.image_size, args.image_size)

    # ── Dataset ──
    # MVTecAD expects: root/category/{train,test,ground_truth}
    # Our structure: transistor/{train,test,ground_truth}
    # So root = parent dir, category = "transistor"
    datamodule = MVTecAD(
        root=str(DATASET_ROOT.parent),
        category="transistor",
        train_batch_size=1 if args.model == "efficient_ad" else 8,
        eval_batch_size=1 if args.model == "efficient_ad" else 8,
        num_workers=0,  # Windows compatibility
    )

    logger.info("Dataset: transistor — 213 train (good), 4 defect types x 10 test")

    # ── Model ──
    model = _create_model(args.model)
    logger.info("Model: %s, image_size=%s", args.model, img_size)

    # ── Train ──
    engine = Engine(
        default_root_dir=str(output_dir),
        max_epochs=args.epochs,
    )

    logger.info("Starting training...")
    engine.fit(model=model, datamodule=datamodule)

    # ── Test ──
    logger.info("Running test evaluation...")
    results = engine.test(model=model, datamodule=datamodule)
    if results:
        for r in results:
            for k, v in r.items():
                logger.info("  %s: %.4f", k, v)

    # ── Find checkpoint ──
    ckpts = sorted(output_dir.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ckpts:
        logger.info("Checkpoint: %s", ckpts[0])

    logger.info("Done! Output: %s", output_dir)


def _create_model(model_name: str):
    model_name = model_name.lower().replace("-", "_")

    if model_name == "efficient_ad":
        from anomalib.models import EfficientAd
        return EfficientAd(
            teacher_out_channels=384,
            model_size="small",
        )
    elif model_name == "patchcore":
        from anomalib.models import Patchcore
        return Patchcore(
            backbone="wide_resnet50_2",
            layers=["layer2", "layer3"],
            coreset_sampling_ratio=0.1,
        )
    elif model_name == "padim":
        from anomalib.models import Padim
        return Padim(
            backbone="resnet18",
            layers=["layer1", "layer2", "layer3"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    main()
