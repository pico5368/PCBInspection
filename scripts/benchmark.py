"""Benchmark: PatchCore vs EfficientAD on transistor dataset.

Compares:
- Image-level AUROC
- Inference speed (per image)
- Memory usage

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pcb_inspection.inspection.anomaly import AnomalyInspector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_DIR = Path("transistor/test")
MODELS = {
    "patchcore": "data/models/transistor/patchcore/Patchcore/MVTecAD/transistor/v0/weights/lightning/model.ckpt",
    "efficient_ad": None,  # Will be found dynamically
}


def find_checkpoint(model_name: str) -> Path | None:
    """Find latest checkpoint for a model."""
    if model_name == "patchcore":
        p = Path(MODELS["patchcore"])
        return p if p.exists() else None

    # Search for EfficientAD checkpoint
    base = Path("data/models/transistor/efficient_ad")
    if not base.exists():
        return None
    ckpts = sorted(base.rglob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None


def load_test_images() -> tuple[list[np.ndarray], list[str], list[bool]]:
    """Load all test images with labels.

    Returns:
        (images, names, is_anomaly)
    """
    images, names, labels = [], [], []

    # Good images
    good_dir = TEST_DIR / "good"
    for img_path in sorted(good_dir.glob("*.png")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
            names.append(f"good/{img_path.name}")
            labels.append(False)

    # Defect images
    for defect_dir in sorted(TEST_DIR.iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        for img_path in sorted(defect_dir.glob("*.png")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                names.append(f"{defect_dir.name}/{img_path.name}")
                labels.append(True)

    return images, names, labels


def benchmark_model(
    model_name: str,
    ckpt_path: Path,
    images: list[np.ndarray],
    names: list[str],
    labels: list[bool],
) -> dict:
    """Run benchmark for a single model."""
    logger.info("=== Benchmarking: %s ===", model_name)
    logger.info("Checkpoint: %s", ckpt_path)

    inspector = AnomalyInspector()
    inspector.load(str(ckpt_path), image_size=(256, 256))

    if not inspector.is_loaded:
        logger.error("Failed to load model: %s", model_name)
        return {}

    scores = []
    inference_times = []

    # Warmup (first inference is slower)
    _ = inspector.inspect(images[0], None, {"component_id": "warmup", "anomaly_threshold": 0.5})

    for i, (img, name, is_anomaly) in enumerate(zip(images, names, labels)):
        t0 = time.perf_counter()
        result = inspector.inspect(img, None, {
            "component_id": name,
            "anomaly_threshold": 0.5,
        })
        t1 = time.perf_counter()

        score = result.metadata.get("anomaly_score", 0)
        scores.append(score)
        inference_times.append(t1 - t0)

    # Compute metrics
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # AUROC
    auroc = _compute_auroc(labels_arr, scores_arr)

    # Score statistics
    normal_scores = scores_arr[~labels_arr]
    anomaly_scores = scores_arr[labels_arr]

    # Optimal threshold (Youden's J)
    best_thresh, best_f1 = _find_optimal_threshold(labels_arr, scores_arr)

    # Speed
    times = np.array(inference_times[1:])  # exclude warmup

    results = {
        "model": model_name,
        "image_auroc": auroc,
        "best_threshold": best_thresh,
        "best_f1": best_f1,
        "normal_score_mean": float(np.mean(normal_scores)),
        "normal_score_std": float(np.std(normal_scores)),
        "normal_score_max": float(np.max(normal_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_std": float(np.std(anomaly_scores)),
        "anomaly_score_min": float(np.min(anomaly_scores)),
        "score_gap": float(np.min(anomaly_scores) - np.max(normal_scores)),
        "inference_mean_ms": float(np.mean(times) * 1000),
        "inference_std_ms": float(np.std(times) * 1000),
        "inference_p95_ms": float(np.percentile(times, 95) * 1000),
        "fps": float(1.0 / np.mean(times)),
        "total_images": len(images),
    }

    return results


def _compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUROC without sklearn."""
    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    tp = 0
    fp = 0
    total_pos = np.sum(labels)
    total_neg = len(labels) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr_prev = 0
    fpr_prev = 0
    auroc = 0.0

    for i in range(len(sorted_labels)):
        if sorted_labels[i]:
            tp += 1
        else:
            fp += 1

        tpr = tp / total_pos
        fpr = fp / total_neg

        # Trapezoidal rule
        auroc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr

    return auroc


def _find_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes F1 score."""
    thresholds = np.linspace(float(np.min(scores)), float(np.max(scores)), 200)
    best_f1 = 0
    best_thresh = 0.5

    for t in thresholds:
        preds = scores >= t
        tp = np.sum(preds & labels)
        fp = np.sum(preds & ~labels)
        fn = np.sum(~preds & labels)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return float(best_thresh), float(best_f1)


def print_comparison(results: list[dict]):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print("  BENCHMARK RESULTS: PatchCore vs EfficientAD")
    print("  Dataset: MVTec Transistor (213 train, 100 test)")
    print("=" * 75)

    # Header
    print(f"\n{'Metric':<30} ", end="")
    for r in results:
        print(f"{'  ' + r['model']:>20}", end="")
    print()
    print("-" * 75)

    metrics = [
        ("Image AUROC", "image_auroc", ".4f"),
        ("Best F1", "best_f1", ".4f"),
        ("Best Threshold", "best_threshold", ".3f"),
        ("", None, None),
        ("Normal Score (mean)", "normal_score_mean", ".3f"),
        ("Normal Score (max)", "normal_score_max", ".3f"),
        ("Anomaly Score (mean)", "anomaly_score_mean", ".3f"),
        ("Anomaly Score (min)", "anomaly_score_min", ".3f"),
        ("Score Gap (min_anom-max_norm)", "score_gap", ".3f"),
        ("", None, None),
        ("Inference (mean ms)", "inference_mean_ms", ".1f"),
        ("Inference (p95 ms)", "inference_p95_ms", ".1f"),
        ("Throughput (FPS)", "fps", ".1f"),
    ]

    for label, key, fmt in metrics:
        if key is None:
            print()
            continue
        print(f"{label:<30} ", end="")
        for r in results:
            val = r.get(key, 0)
            print(f"  {val:>18{fmt}}", end="")
        print()

    # Winner summary
    print("\n" + "=" * 75)
    if len(results) >= 2:
        auroc_winner = max(results, key=lambda r: r.get("image_auroc", 0))
        speed_winner = min(results, key=lambda r: r.get("inference_mean_ms", float("inf")))
        print(f"  Accuracy winner:  {auroc_winner['model']} (AUROC={auroc_winner['image_auroc']:.4f})")
        print(f"  Speed winner:     {speed_winner['model']} ({speed_winner['inference_mean_ms']:.1f}ms)")
    print("=" * 75)


def main():
    # Load test data
    images, names, labels = load_test_images()
    logger.info("Loaded %d test images (%d normal, %d anomaly)",
                len(images), sum(1 for l in labels if not l), sum(labels))

    all_results = []

    # Benchmark each model
    for model_name in ["patchcore", "efficient_ad"]:
        ckpt = find_checkpoint(model_name)
        if ckpt is None:
            logger.warning("Checkpoint not found for %s, skipping", model_name)
            continue

        result = benchmark_model(model_name, ckpt, images, names, labels)
        if result:
            all_results.append(result)

    if all_results:
        print_comparison(all_results)


if __name__ == "__main__":
    main()
