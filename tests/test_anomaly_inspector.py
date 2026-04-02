"""Tests for AnomalyInspector integration with anomalib."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pcb_inspection.common.types import InspectionType, Severity
from pcb_inspection.inspection.anomaly import AnomalyInspector

CKPT_PATH = Path("data/models/transistor/patchcore/Patchcore/MVTecAD/transistor/v0/weights/lightning/model.ckpt")
TRANSISTOR_DIR = Path("transistor/test")


def has_model():
    return CKPT_PATH.exists()


def has_dataset():
    return TRANSISTOR_DIR.exists()


@pytest.fixture(scope="module")
def inspector():
    """Load anomaly inspector once for all tests in this module."""
    if not has_model():
        pytest.skip("PatchCore checkpoint not found")
    insp = AnomalyInspector()
    insp.load(str(CKPT_PATH), image_size=(256, 256))
    assert insp.is_loaded
    return insp


class TestAnomalyInspectorLoad:
    def test_load_nonexistent_path(self):
        insp = AnomalyInspector()
        insp.load("/nonexistent/model.ckpt")
        assert not insp.is_loaded

    def test_load_none(self):
        insp = AnomalyInspector()
        insp.load(None)
        assert not insp.is_loaded

    def test_not_loaded_returns_warning(self):
        insp = AnomalyInspector()
        result = insp.inspect(
            np.zeros((100, 100, 3), dtype=np.uint8),
            None,
            {"component_id": "test"},
        )
        assert result.severity == Severity.WARNING
        assert "not loaded" in result.detail


@pytest.mark.skipif(not has_model() or not has_dataset(), reason="Model or dataset not available")
class TestAnomalyInspectorInference:
    def test_normal_image_low_score(self, inspector):
        """Normal transistor image should have low anomaly score."""
        img = cv2.imread(str(TRANSISTOR_DIR / "good/000.png"))
        result = inspector.inspect(img, None, {
            "component_id": "good_000",
            "anomaly_threshold": 0.52,
        })
        score = result.metadata["anomaly_score"]
        assert score < 0.6, f"Normal image score too high: {score}"

    def test_defect_image_high_score(self, inspector):
        """Defect image should have high anomaly score."""
        img = cv2.imread(str(TRANSISTOR_DIR / "bent_lead/000.png"))
        result = inspector.inspect(img, None, {
            "component_id": "bent_000",
            "anomaly_threshold": 0.52,
        })
        score = result.metadata["anomaly_score"]
        assert score > 0.5, f"Defect image score too low: {score}"

    def test_defect_detected_as_ng(self, inspector):
        """Defect image should be classified as NG."""
        img = cv2.imread(str(TRANSISTOR_DIR / "bent_lead/004.png"))
        result = inspector.inspect(img, None, {
            "component_id": "bent_004",
            "anomaly_threshold": 0.52,
        })
        assert result.severity == Severity.NG

    def test_all_defect_types(self, inspector):
        """All defect types should have higher avg score than normal."""
        normal_scores = []
        for p in sorted((TRANSISTOR_DIR / "good").glob("*.png"))[:10]:
            img = cv2.imread(str(p))
            r = inspector.inspect(img, None, {"component_id": p.stem, "anomaly_threshold": 0.52})
            normal_scores.append(r.metadata["anomaly_score"])

        for defect_type in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
            defect_scores = []
            for p in sorted((TRANSISTOR_DIR / defect_type).glob("*.png")):
                img = cv2.imread(str(p))
                r = inspector.inspect(img, None, {"component_id": p.stem, "anomaly_threshold": 0.52})
                defect_scores.append(r.metadata["anomaly_score"])

            avg_normal = np.mean(normal_scores)
            avg_defect = np.mean(defect_scores)
            assert avg_defect > avg_normal, (
                f"{defect_type}: avg_defect={avg_defect:.3f} <= avg_normal={avg_normal:.3f}"
            )

    def test_result_has_metadata(self, inspector):
        """Result should contain expected metadata fields."""
        img = cv2.imread(str(TRANSISTOR_DIR / "good/000.png"))
        result = inspector.inspect(img, None, {"component_id": "test"})

        assert result.inspection_type == InspectionType.ANOMALY
        assert "anomaly_score" in result.metadata
        assert "pred_label" in result.metadata
        assert "has_anomaly_map" in result.metadata
        assert 0.0 <= result.score <= 1.0
