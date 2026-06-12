"""Compare combined vs per-position PaDiM models by raw anomaly-score distribution.

Loads each trained PaDiM checkpoint and runs the *raw* model forward (pre
post-processing normalization) over a fixed image set, so scores are directly
comparable across models (same backbone -> same Mahalanobis-distance units).

Answers: does a dedicated per-position model (B1/B2) give a tighter / lower
normal-score distribution on its own panel than the combined poc_panel model?
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2 as T

logging.disable(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TF = T.Compose([
    T.Resize((256, 256), antialias=True),
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CKPT = {
    "poc_panel (합침)": "data/models/poc_panel/padim/Padim/poc_panel/v1/weights/lightning/model.ckpt",
    "B1 (분리)":        "data/models/poc_panel_B1/padim/Padim/poc_panel_B1/v1/weights/lightning/model.ckpt",
    "B2 (분리)":        "data/models/poc_panel_B2/padim/Padim/poc_panel_B2/v1/weights/lightning/model.ckpt",
}
EVAL_DIRS = {
    "B1 images (260)": "data/datasets/poc_panel_B1/good",
    "B2 images (260)": "data/datasets/poc_panel_B2/good",
}


def load_model(ckpt_path: str):
    from anomalib.models import Padim
    model = Padim.load_from_checkpoint(ckpt_path)
    model.eval().to(DEVICE)
    return model


@torch.no_grad()
def raw_scores(model, image_dir: Path, batch=16) -> np.ndarray:
    paths = sorted(p for p in image_dir.glob("*.png"))
    scores: list[float] = []
    for i in range(0, len(paths), batch):
        imgs = [TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch]]
        x = torch.stack(imgs).to(DEVICE)
        out = model.model(x)  # raw PadimModel forward -> InferenceBatch
        ps = out.pred_score if hasattr(out, "pred_score") else out[1]
        scores.extend(ps.detach().cpu().flatten().tolist())
    return np.array(scores)


def stored_threshold(ckpt_path: str) -> tuple[float, float, float]:
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    thr = float(sd["post_processor._image_threshold"])
    lo = float(sd["post_processor.image_min"])
    hi = float(sd["post_processor.image_max"])
    return lo, hi, thr


def main() -> int:
    results = {}
    thresholds = {}
    for name, ckpt in CKPT.items():
        if not Path(ckpt).exists():
            print(f"MISSING ckpt: {ckpt}", file=sys.stderr)
            return 1
        thresholds[name] = stored_threshold(ckpt)
        model = load_model(ckpt)
        for set_name, d in EVAL_DIRS.items():
            results[(name, set_name)] = raw_scores(model, Path(d))
        del model
        torch.cuda.empty_cache()

    def stat(a):
        return (a.mean(), a.std(), np.percentile(a, 50), np.percentile(a, 95), a.max())

    # ── Stored adaptive threshold per model (from its own val split) ──
    print("\n=== 모델별 저장된 적응형 임계값 (자기 val split 기준) ===")
    print(f"{'Model':<18} {'정상 min':>10} {'정상 max':>10} {'임계값':>10}")
    for name, (lo, hi, thr) in thresholds.items():
        print(f"{name:<18} {lo:>10.3f} {hi:>10.3f} {thr:>10.3f}")

    # ── Raw score distribution: combined vs dedicated, on the SAME images ──
    print("\n=== Raw Mahalanobis 점수 분포 (동일 이미지셋, 모델 직접 비교) ===")
    print(f"{'Eval set':<16} {'Model':<18} {'mean':>8} {'std':>8} {'p50':>8} {'p95':>8} {'max':>8}")
    for set_name in EVAL_DIRS:
        dedicated = "B1 (분리)" if "B1" in set_name else "B2 (분리)"
        for name in ("poc_panel (합침)", dedicated):
            m, s, p50, p95, mx = stat(results[(name, set_name)])
            print(f"{set_name:<16} {name:<18} {m:>8.3f} {s:>8.3f} {p50:>8.3f} {p95:>8.3f} {mx:>8.3f}")
        # tightness delta
        c = results[("poc_panel (합침)", set_name)]
        d = results[(dedicated, set_name)]
        print(f"{'  Δ (분리-합침)':<16} {'':<18} {d.mean()-c.mean():>8.3f} {d.std()-c.std():>8.3f} "
              f"{'':>8} {'':>8} {d.max()-c.max():>8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
