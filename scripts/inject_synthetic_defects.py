"""Inject synthetic defects into anomalib `good/` crops to populate `defect/` folders.

Whole-board defect images don't reliably land a defect inside every component
ROI, so per-component crops sampled from them are mostly normal. Instead we
corrupt a sample of the existing good crops with localized, clearly-anomalous
artifacts. This gives each component dataset a labelled `defect/` set so
anomalib can calibrate its adaptive threshold and report real AUROC/F1.

Defect types: solder blob, scratch, missing region, smudge.

Usage:
    python scripts/inject_synthetic_defects.py                       # all components
    python scripts/inject_synthetic_defects.py --root data/synthetic/anomaly_dataset
    python scripts/inject_synthetic_defects.py --per-component 20 --seed 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFECT_TYPES = ("blob", "scratch", "missing", "smudge")


def _inject_defect(img: np.ndarray, defect_type: str, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of img with one localized synthetic defect applied."""
    out = img.copy()
    h, w = out.shape[:2]
    cx, cy = int(rng.integers(w // 4, 3 * w // 4)), int(rng.integers(h // 4, 3 * h // 4))

    if defect_type == "blob":
        # Bright solder ball / excess solder.
        r = int(rng.integers(max(4, w // 16), max(6, w // 8)))
        cv2.circle(out, (cx, cy), r, (235, 240, 235), -1)
    elif defect_type == "scratch":
        # Thin bright/dark line across the part.
        ang = rng.uniform(0, np.pi)
        length = int(rng.integers(w // 3, w))
        dx, dy = int(np.cos(ang) * length / 2), int(np.sin(ang) * length / 2)
        color = (240, 240, 240) if rng.random() < 0.5 else (15, 15, 15)
        cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), color, int(rng.integers(2, 5)))
    elif defect_type == "missing":
        # Region wiped to PCB-background green (missing / tombstoned part).
        bw, bh = int(rng.integers(w // 5, w // 3)), int(rng.integers(h // 5, h // 3))
        x0, y0 = max(0, cx - bw), max(0, cy - bh)
        out[y0:cy + bh, x0:cx + bw] = (30, 80, 40)
    elif defect_type == "smudge":
        # Localized blur (contamination / lifted part).
        bw, bh = w // 3, h // 3
        x0, y0 = max(0, cx - bw), max(0, cy - bh)
        x1, y1 = min(w, cx + bw), min(h, cy + bh)
        patch = out[y0:y1, x0:x1]
        if patch.size:
            out[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), sigmaX=6)
    else:
        raise ValueError(f"Unknown defect type: {defect_type}")

    return out


def inject_for_component(comp_dir: Path, per_component: int, rng: np.random.Generator) -> int:
    """Create comp_dir/defect/ from a sample of comp_dir/good/. Returns count written."""
    good_dir = comp_dir / "good"
    defect_dir = comp_dir / "defect"
    good_imgs = sorted(good_dir.glob("*.png"))
    if not good_imgs:
        logger.warning("%s: no good crops, skipping", comp_dir.name)
        return 0

    defect_dir.mkdir(exist_ok=True)
    # Wipe any previous synthetic defects so reruns are deterministic.
    for old in defect_dir.glob("*.png"):
        old.unlink()

    n = min(per_component, len(good_imgs))
    picks = rng.choice(len(good_imgs), size=n, replace=False)

    written = 0
    for k, idx in enumerate(picks):
        src = good_imgs[int(idx)]
        img = cv2.imread(str(src))
        if img is None:
            continue
        dtype = DEFECT_TYPES[k % len(DEFECT_TYPES)]
        out = _inject_defect(img, dtype, rng)
        cv2.imwrite(str(defect_dir / f"defect_{k:04d}_{dtype}.png"), out)
        written += 1

    logger.info("%s: wrote %d defect crops (from %d good)", comp_dir.name, written, len(good_imgs))
    return written


def main():
    parser = argparse.ArgumentParser(description="Inject synthetic defects into anomalib good/ crops")
    parser.add_argument("--root", type=str, default="data/synthetic/anomaly_dataset",
                        help="Root containing <COMPONENT>/good/ subfolders")
    parser.add_argument("--per-component", type=int, default=20, help="Max defect crops per component")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        parser.error(f"root not found: {root}")

    comp_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "good").exists())
    if not comp_dirs:
        parser.error(f"no <COMPONENT>/good/ datasets under {root}")

    rng = np.random.default_rng(args.seed)
    total = 0
    for comp_dir in comp_dirs:
        total += inject_for_component(comp_dir, args.per_component, rng)

    logger.info("Done: %d defect crops across %d components", total, len(comp_dirs))


if __name__ == "__main__":
    main()
