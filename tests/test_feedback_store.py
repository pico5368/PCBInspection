"""Tests for FeedbackStore data pipeline."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from pcb_inspection.data.feedback_store import FeedbackItem, FeedbackStore, _bbox_equals


@pytest.fixture
def store(tmp_path):
    """Create a FeedbackStore with temp directory."""
    return FeedbackStore(data_root=tmp_path)


@pytest.fixture
def sample_image():
    return np.full((256, 256, 3), 128, dtype=np.uint8)


class TestBboxEquals:
    def test_exact_match(self):
        assert _bbox_equals([10, 20, 30, 40], [10, 20, 30, 40])

    def test_within_tolerance(self):
        assert _bbox_equals([10, 20, 30, 40], [11, 21, 29, 39], tolerance=2)

    def test_outside_tolerance(self):
        assert not _bbox_equals([10, 20, 30, 40], [15, 20, 30, 40], tolerance=2)

    def test_different_length(self):
        assert not _bbox_equals([10, 20], [10, 20, 30])


class TestFeedbackStore:
    def test_save_raw(self, store, sample_image):
        path = store.save_raw(sample_image, "board_001", {"overall": "ok"})
        assert path.exists()
        assert path.with_suffix(".json").exists()

    def test_save_feedback_false_positive(self, store):
        """False positive feedback should be logged."""
        feedbacks = [FeedbackItem(feedback_type="false_positive", target_bbox=[10, 20, 50, 60])]
        result = store.save_feedback("board_001", "R1", feedbacks)
        assert result["final_labels"] == 0

    def test_save_feedback_false_negative(self, store):
        """False negative should add to needs_labeling count."""
        feedbacks = [FeedbackItem(
            feedback_type="false_negative",
            correct_label="solder_bridge",
            target_bbox=[10, 20, 50, 60],
        )]
        result = store.save_feedback("board_001", "R1", feedbacks)
        assert result["false_negatives"] == 1

    def test_save_feedback_with_image_creates_refined(self, store, sample_image):
        """Feedback with image should create refined dataset entry."""
        original_dets = [
            {"defect_type": "missing", "bbox": [10, 20, 50, 60], "class_id": 0, "confidence": 0.9}
        ]
        feedbacks = [FeedbackItem(
            feedback_type="tp_wrong_class",
            correct_label="solder_bridge",
            target_bbox=[10, 20, 50, 60],
        )]
        result = store.save_feedback(
            "board_001", "R1", feedbacks,
            image=sample_image, original_detections=original_dets,
        )
        assert result["refined_path"] is not None
        # Check refined files exist
        assert len(list((store.refined_dir / "images").glob("*.png"))) == 1
        assert len(list((store.refined_dir / "labels").glob("*.txt"))) == 1

    def test_refined_label_yolo_format(self, store, sample_image):
        """Refined labels should be in YOLO format (normalized coords)."""
        original_dets = [
            {"defect_type": "missing", "bbox": [64, 64, 192, 192], "class_id": 0, "confidence": 0.9}
        ]
        store.save_feedback(
            "board_002", "U1", [],
            image=sample_image, original_detections=original_dets,
        )
        label_files = list((store.refined_dir / "labels").glob("*.txt"))
        assert len(label_files) == 1

        content = label_files[0].read_text().strip()
        parts = content.split()
        assert len(parts) == 5  # class_id x_center y_center w h
        assert parts[0] == "0"
        # Normalized coords should be 0-1
        for val in parts[1:]:
            assert 0.0 <= float(val) <= 1.0

    def test_feedback_log_jsonl(self, store):
        """Feedback should be appended to JSONL file."""
        feedbacks = [FeedbackItem(feedback_type="false_positive", target_bbox=[0, 0, 10, 10])]
        store.save_feedback("board_003", "C1", feedbacks)
        store.save_feedback("board_003", "C2", feedbacks)

        log_file = store.feedback_dir / "board_003.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_get_stats(self, store):
        """Stats should reflect saved feedback."""
        store.save_feedback("b1", "R1", [FeedbackItem(feedback_type="false_positive")])
        store.save_feedback("b1", "R2", [FeedbackItem(feedback_type="false_negative", correct_label="x")])
        store.save_feedback("b1", "R3", [FeedbackItem(feedback_type="tp_wrong_class", correct_label="y")])

        stats = store.get_stats()
        assert stats["total_feedback"] == 3
        assert stats["false_rejects"] == 1
        assert stats["escapes"] == 1

    def test_implicit_true_positive(self, store, sample_image):
        """Detections without feedback should be kept as true positives."""
        original_dets = [
            {"defect_type": "missing", "bbox": [10, 20, 50, 60], "class_id": 0, "confidence": 0.9},
            {"defect_type": "bridge", "bbox": [100, 100, 150, 150], "class_id": 1, "confidence": 0.85},
        ]
        # Only give feedback for first detection
        feedbacks = [FeedbackItem(
            feedback_type="false_positive",
            target_bbox=[10, 20, 50, 60],
        )]
        result = store.save_feedback(
            "board_004", "U2", feedbacks,
            image=sample_image, original_detections=original_dets,
        )
        # First det removed (FP), second kept (implicit TP)
        assert result["final_labels"] == 1
