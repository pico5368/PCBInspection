"""Tests for inspection modules."""

from __future__ import annotations

import numpy as np

from pcb_inspection.common.types import InspectionType, Severity
from pcb_inspection.inspection.reference import ReferenceInspector
from pcb_inspection.inspection.rule_based import RuleBasedInspector
from pcb_inspection.inspection.blob import BlobInspector


class TestReferenceInspector:
    def test_identical_images_pass(self):
        inspector = ReferenceInspector()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ref = img.copy()

        result = inspector.inspect(img, ref, {"component_id": "R1"})
        assert result.severity == Severity.OK

    def test_blank_vs_component_fails(self):
        inspector = ReferenceInspector()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)  # blank
        ref = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

        result = inspector.inspect(img, ref, {
            "component_id": "R1",
            "similarity_threshold": 0.8,
        })
        # Different images should have low similarity
        assert result.score < 1.0

    def test_no_reference_warns(self):
        inspector = ReferenceInspector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = inspector.inspect(img, None, {"component_id": "R1"})
        assert result.severity == Severity.WARNING


class TestRuleBasedInspector:
    def test_identical_images_pass(self):
        inspector = RuleBasedInspector()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        ref = img.copy()

        result = inspector.inspect(img, ref, {
            "component_id": "R1",
            "max_offset_px": 10.0,
        })
        assert result.severity == Severity.OK

    def test_no_reference_warns(self):
        inspector = RuleBasedInspector()
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = inspector.inspect(img, None, {"component_id": "R1"})
        assert result.severity == Severity.WARNING


class TestBlobInspector:
    def test_clean_image_passes(self):
        inspector = BlobInspector()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        ref = img.copy()

        result = inspector.inspect(img, ref, {
            "component_id": "C1",
            "max_blob_count": 0,
        })
        assert result.severity == Severity.OK

    def test_detects_added_blob(self):
        inspector = BlobInspector()
        ref = np.full((100, 100, 3), 128, dtype=np.uint8)
        img = ref.copy()
        # Add a bright blob
        img[40:55, 40:55] = [255, 255, 255]

        result = inspector.inspect(img, ref, {
            "component_id": "C1",
            "max_blob_count": 0,
            "min_blob_area": 5,
            "diff_threshold": 30,
        })
        assert result.metadata["blob_count"] > 0
