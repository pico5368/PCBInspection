"""Tests for FastAPI backend endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.main import app, _inspection_log
    _inspection_log.clear()
    return TestClient(app)


@pytest.fixture
def sample_png():
    """Create a sample PNG image as bytes."""
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] in ("healthy", "warning", "critical")

    def test_health_has_timestamp(self, client):
        r = client.get("/health")
        assert "timestamp" in r.json()


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        r = client.get("/stats")
        assert r.status_code == 200
        data = r.json()
        assert "total_inspections" in data
        assert "defect_rate" in data
        assert "retrain_ready" in data

    def test_feedback_stats_returns_200(self, client):
        r = client.get("/stats/feedback")
        assert r.status_code == 200


class TestInspectEndpoint:
    def test_inspect_image_returns_result(self, client, sample_png):
        r = client.post("/inspect/image", files={"file": ("test.png", sample_png, "image/png")})
        assert r.status_code == 200
        data = r.json()
        assert "board_id" in data
        assert data["overall"] in ("ok", "warning", "ng")
        assert "timestamp" in data

    def test_inspect_invalid_image_returns_400(self, client):
        r = client.post("/inspect/image", files={"file": ("bad.png", b"not an image", "image/png")})
        assert r.status_code == 400

    def test_inspect_details_after_inspection(self, client, sample_png):
        r = client.post("/inspect/image", files={"file": ("test.png", sample_png, "image/png")})
        board_id = r.json()["board_id"]

        r2 = client.get(f"/inspect/{board_id}/details")
        assert r2.status_code == 200
        assert isinstance(r2.json(), list)

    def test_inspect_details_not_found(self, client):
        r = client.get("/inspect/nonexistent_board/details")
        assert r.status_code == 404


class TestFeedbackEndpoints:
    def test_quick_feedback_ok(self, client, sample_png):
        # First create an inspection
        r = client.post("/inspect/image", files={"file": ("test.png", sample_png, "image/png")})
        board_id = r.json()["board_id"]

        r2 = client.post(f"/feedback/quick?board_id={board_id}&component_id=full_image&is_ok=true")
        assert r2.status_code == 200
        assert r2.json()["status"] == "saved"

    def test_quick_feedback_ng(self, client, sample_png):
        r = client.post("/inspect/image", files={"file": ("test.png", sample_png, "image/png")})
        board_id = r.json()["board_id"]

        r2 = client.post(f"/feedback/quick?board_id={board_id}&component_id=full_image&is_ok=false")
        assert r2.status_code == 200
        assert r2.json()["false_negatives"] == 1

    def test_bulk_feedback(self, client, sample_png):
        r = client.post("/inspect/image", files={"file": ("test.png", sample_png, "image/png")})
        board_id = r.json()["board_id"]

        payload = {
            "board_id": board_id,
            "component_id": "full_image",
            "feedbacks": [
                {"feedback_type": "false_positive", "target_bbox": [0, 0, 50, 50]},
                {"feedback_type": "false_negative", "correct_label": "bridge", "target_bbox": [100, 100, 150, 150]},
            ],
        }
        r2 = client.post("/feedback/bulk", json=payload)
        assert r2.status_code == 200
        assert r2.json()["false_negatives"] == 1
