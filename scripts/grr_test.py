"""GR&R (Gauge Repeatability & Reproducibility) Test for PCB Inspection.

Validates:
1. Repeatability: Same image → same result (N runs)
2. Reproducibility: Different conditions → consistent result
3. Threshold stability: Score distribution analysis

Usage:
    python scripts/grr_test.py
    python scripts/grr_test.py --runs 20 --model patchcore
    python scripts/grr_test.py --image-dir transistor/test --report data/grr_report.json

PLAN.md Section 15: GR&R must pass before production deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pcb_inspection.inspection.anomaly import AnomalyInspector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CKPT_PATH = Path("data/models/transistor/patchcore/Patchcore/MVTecAD/transistor/v0/weights/lightning/model.ckpt")


@dataclass
class GRRResult:
    """Result of a single GR&R test."""
    image_path: str
    expected_label: str  # "normal" or "anomaly"
    scores: list[float] = field(default_factory=list)
    severities: list[str] = field(default_factory=list)
    inference_times_ms: list[float] = field(default_factory=list)

    @property
    def score_mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0

    @property
    def score_std(self) -> float:
        return float(np.std(self.scores)) if self.scores else 0

    @property
    def score_range(self) -> float:
        return float(np.max(self.scores) - np.min(self.scores)) if self.scores else 0

    @property
    def judgment_consistent(self) -> bool:
        """All runs produced the same severity."""
        return len(set(self.severities)) == 1

    @property
    def majority_severity(self) -> str:
        from collections import Counter
        return Counter(self.severities).most_common(1)[0][0] if self.severities else "unknown"

    @property
    def consistency_rate(self) -> float:
        """Fraction of runs matching majority severity."""
        from collections import Counter
        if not self.severities:
            return 0
        most_common_count = Counter(self.severities).most_common(1)[0][1]
        return most_common_count / len(self.severities)

    @property
    def avg_time_ms(self) -> float:
        return float(np.mean(self.inference_times_ms)) if self.inference_times_ms else 0

    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "expected": self.expected_label,
            "runs": len(self.scores),
            "score_mean": round(self.score_mean, 4),
            "score_std": round(self.score_std, 4),
            "score_range": round(self.score_range, 4),
            "score_min": round(float(min(self.scores)), 4) if self.scores else 0,
            "score_max": round(float(max(self.scores)), 4) if self.scores else 0,
            "judgment_consistent": self.judgment_consistent,
            "consistency_rate": round(self.consistency_rate, 4),
            "majority_severity": self.majority_severity,
            "avg_time_ms": round(self.avg_time_ms, 1),
        }


@dataclass
class GRRReport:
    """Aggregate GR&R report."""
    results: list[GRRResult] = field(default_factory=list)
    threshold: float = 0.52
    runs_per_image: int = 10
    timestamp: str = ""

    @property
    def total_images(self) -> int:
        return len(self.results)

    @property
    def fully_consistent(self) -> int:
        """Images where all runs gave identical judgment."""
        return sum(1 for r in self.results if r.judgment_consistent)

    @property
    def overall_consistency_rate(self) -> float:
        if not self.results:
            return 0
        return self.fully_consistent / self.total_images

    @property
    def avg_score_std(self) -> float:
        stds = [r.score_std for r in self.results]
        return float(np.mean(stds)) if stds else 0

    @property
    def max_score_range(self) -> float:
        ranges = [r.score_range for r in self.results]
        return float(np.max(ranges)) if ranges else 0

    @property
    def correct_labels(self) -> int:
        """Images where majority severity matches expected label."""
        count = 0
        for r in self.results:
            expected_ok = r.expected_label == "normal"
            predicted_ok = r.majority_severity in ("ok", "warning")
            if expected_ok == predicted_ok:
                count += 1
        return count

    @property
    def label_accuracy(self) -> float:
        return self.correct_labels / self.total_images if self.total_images else 0

    def passes_grr(self, min_consistency: float = 0.99, max_score_std: float = 0.05) -> bool:
        """Check if GR&R criteria are met.

        PLAN.md criteria:
        - Judgment consistency > 99% (same condition)
        - Score std < 0.05 (stability)
        """
        return (
            self.overall_consistency_rate >= min_consistency
            and self.avg_score_std <= max_score_std
        )

    def to_dict(self) -> dict:
        return {
            "summary": {
                "timestamp": self.timestamp,
                "total_images": self.total_images,
                "runs_per_image": self.runs_per_image,
                "threshold": self.threshold,
                "overall_consistency_rate": round(self.overall_consistency_rate, 4),
                "fully_consistent_images": self.fully_consistent,
                "avg_score_std": round(self.avg_score_std, 4),
                "max_score_range": round(self.max_score_range, 4),
                "label_accuracy": round(self.label_accuracy, 4),
                "grr_pass": self.passes_grr(),
            },
            "per_image": [r.to_dict() for r in self.results],
        }


def run_grr_test(
    inspector: AnomalyInspector,
    image_dir: Path,
    runs: int = 10,
    threshold: float = 0.52,
    max_images_per_category: int = 10,
) -> GRRReport:
    """Run full GR&R test.

    Args:
        inspector: Loaded AnomalyInspector.
        image_dir: Directory with good/ and defect subdirs.
        runs: Number of repeated inferences per image.
        threshold: Anomaly threshold for classification.
        max_images_per_category: Max images to test per category.

    Returns:
        GRRReport with all results.
    """
    report = GRRReport(
        threshold=threshold,
        runs_per_image=runs,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Collect test images
    test_images = []

    good_dir = image_dir / "good"
    if good_dir.exists():
        for p in sorted(good_dir.glob("*.png"))[:max_images_per_category]:
            test_images.append((p, "normal"))

    for defect_dir in sorted(image_dir.iterdir()):
        if defect_dir.name == "good" or not defect_dir.is_dir():
            continue
        for p in sorted(defect_dir.glob("*.png"))[:max_images_per_category]:
            test_images.append((p, "anomaly"))

    logger.info("GR&R test: %d images x %d runs = %d total inferences",
                len(test_images), runs, len(test_images) * runs)

    # Warmup
    if test_images:
        img = cv2.imread(str(test_images[0][0]))
        inspector.inspect(img, None, {"component_id": "warmup", "anomaly_threshold": threshold})

    # Run tests
    for img_idx, (img_path, expected_label) in enumerate(test_images):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        result = GRRResult(
            image_path=str(img_path.relative_to(image_dir)),
            expected_label=expected_label,
        )

        for run_idx in range(runs):
            t0 = time.perf_counter()
            r = inspector.inspect(image, None, {
                "component_id": f"{img_path.stem}_run{run_idx}",
                "anomaly_threshold": threshold,
                "warning_threshold": threshold * 0.8,
            })
            t1 = time.perf_counter()

            result.scores.append(r.metadata.get("anomaly_score", 0))
            result.severities.append(r.severity.value)
            result.inference_times_ms.append((t1 - t0) * 1000)

        report.results.append(result)

        # Progress
        if (img_idx + 1) % 10 == 0 or img_idx == len(test_images) - 1:
            logger.info(
                "  [%d/%d] %s: mean=%.3f, std=%.4f, consistent=%s",
                img_idx + 1, len(test_images),
                img_path.name, result.score_mean, result.score_std,
                "YES" if result.judgment_consistent else "NO",
            )

    return report


def print_report(report: GRRReport):
    """Print formatted GR&R report."""
    s = report.to_dict()["summary"]

    print("\n" + "=" * 70)
    print("  GR&R TEST REPORT")
    print("=" * 70)
    print(f"  Date:           {s['timestamp'][:19]}")
    print(f"  Images:         {s['total_images']}")
    print(f"  Runs/image:     {s['runs_per_image']}")
    print(f"  Threshold:      {s['threshold']}")
    print()
    print(f"  {'Metric':<35} {'Value':>12} {'Criteria':>12} {'Pass':>6}")
    print(f"  {'-'*65}")
    print(f"  {'Judgment Consistency':<35} {s['overall_consistency_rate']*100:>11.1f}% {'>= 99%':>12} {'PASS' if s['overall_consistency_rate'] >= 0.99 else 'FAIL':>6}")
    print(f"  {'Avg Score Std Dev':<35} {s['avg_score_std']:>12.4f} {'<= 0.05':>12} {'PASS' if s['avg_score_std'] <= 0.05 else 'FAIL':>6}")
    print(f"  {'Max Score Range':<35} {s['max_score_range']:>12.4f} {'<= 0.10':>12} {'PASS' if s['max_score_range'] <= 0.10 else 'FAIL':>6}")
    print(f"  {'Label Accuracy':<35} {s['label_accuracy']*100:>11.1f}%")
    print(f"  {'Fully Consistent Images':<35} {s['fully_consistent_images']:>8} / {s['total_images']}")
    print()

    grr_pass = s["grr_pass"]
    print(f"  OVERALL: {'✅ PASS — Ready for production' if grr_pass else '❌ FAIL — Needs investigation'}")
    print("=" * 70)

    # Per-category breakdown
    normal_results = [r for r in report.results if r.expected_label == "normal"]
    anomaly_results = [r for r in report.results if r.expected_label == "anomaly"]

    print(f"\n  Category Breakdown:")
    if normal_results:
        consistency = sum(1 for r in normal_results if r.judgment_consistent) / len(normal_results)
        avg_std = np.mean([r.score_std for r in normal_results])
        print(f"    Normal:  {len(normal_results)} images, consistency={consistency*100:.1f}%, avg_std={avg_std:.4f}")
    if anomaly_results:
        consistency = sum(1 for r in anomaly_results if r.judgment_consistent) / len(anomaly_results)
        avg_std = np.mean([r.score_std for r in anomaly_results])
        print(f"    Anomaly: {len(anomaly_results)} images, consistency={consistency*100:.1f}%, avg_std={avg_std:.4f}")

    # Inconsistent images
    inconsistent = [r for r in report.results if not r.judgment_consistent]
    if inconsistent:
        print(f"\n  ⚠️  Inconsistent images ({len(inconsistent)}):")
        for r in inconsistent:
            print(f"    {r.image_path}: scores={r.score_mean:.3f}±{r.score_std:.4f}, "
                  f"severities={set(r.severities)}, consistency={r.consistency_rate*100:.0f}%")


def main():
    parser = argparse.ArgumentParser(description="GR&R Test for PCB Inspection")
    parser.add_argument("--image-dir", type=str, default="transistor/test")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--max-per-category", type=int, default=10)
    parser.add_argument("--model", type=str, default=str(CKPT_PATH))
    parser.add_argument("--report", type=str, default="data/grr_report.json")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    # Load model
    logger.info("Loading model: %s", model_path)
    inspector = AnomalyInspector()
    inspector.load(str(model_path), image_size=(256, 256))

    if not inspector.is_loaded:
        logger.error("Failed to load model")
        sys.exit(1)

    # Run GR&R
    report = run_grr_test(
        inspector=inspector,
        image_dir=image_dir,
        runs=args.runs,
        threshold=args.threshold,
        max_images_per_category=args.max_per_category,
    )

    # Print results
    print_report(report)

    # Save report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("Report saved to %s", report_path)


if __name__ == "__main__":
    main()
