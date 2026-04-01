"""Tests for ROI module."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

from pcb_inspection.common.types import InspectionType
from pcb_inspection.roi.cad_parser import parse_cpl
from pcb_inspection.roi.roi_generator import generate_rois


def _write_sample_cpl(path: Path) -> None:
    """Write a minimal CPL file for testing."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Designator", "Package", "X(mm)", "Y(mm)", "Rotation", "Layer", "Value"])
        writer.writerow(["R1", "0603", "10.0", "15.0", "0", "top", "10K"])
        writer.writerow(["R2", "0603", "20.0", "15.0", "90", "top", "4.7K"])
        writer.writerow(["U1", "QFP48", "40.0", "30.0", "0", "top", "MCU"])
        writer.writerow(["C1", "0805", "10.0", "30.0", "0", "top", "100nF"])


class TestCPLParser:
    def test_parse_basic_cpl(self, tmp_path):
        cpl_file = tmp_path / "test.csv"
        _write_sample_cpl(cpl_file)

        components = parse_cpl(cpl_file)

        assert len(components) == 4
        assert components[0].designator == "R1"
        assert components[0].package == "0603"
        assert components[0].x_mm == 10.0
        assert components[0].y_mm == 15.0

    def test_parse_with_rotation(self, tmp_path):
        cpl_file = tmp_path / "test.csv"
        _write_sample_cpl(cpl_file)

        components = parse_cpl(cpl_file)
        r2 = next(c for c in components if c.designator == "R2")
        assert r2.rotation == 90.0

    def test_empty_file(self, tmp_path):
        cpl_file = tmp_path / "empty.csv"
        cpl_file.write_text("Designator,Package,X,Y,Rotation,Layer,Value\n")

        components = parse_cpl(cpl_file)
        assert len(components) == 0


class TestROIGenerator:
    def test_generate_rois(self, tmp_path):
        cpl_file = tmp_path / "test.csv"
        _write_sample_cpl(cpl_file)

        components = parse_cpl(cpl_file)
        rois = generate_rois(
            components,
            pixels_per_mm=50.0,
            image_size=(4000, 3000),
        )

        assert len(rois) == 4
        assert rois[0].component_id == "R1"
        assert rois[0].component_type == "0603"
        assert InspectionType.REFERENCE in rois[0].inspection_types

    def test_roi_clipping(self, tmp_path):
        cpl_file = tmp_path / "test.csv"
        _write_sample_cpl(cpl_file)

        components = parse_cpl(cpl_file)
        # Very small image — some ROIs should be clipped
        rois = generate_rois(
            components,
            pixels_per_mm=50.0,
            image_size=(500, 500),
        )

        for roi in rois:
            x, y, w, h = roi.bbox
            assert x >= 0
            assert y >= 0
            assert x + w <= 500
            assert y + h <= 500

    def test_qfp_gets_blob_inspection(self, tmp_path):
        cpl_file = tmp_path / "test.csv"
        _write_sample_cpl(cpl_file)

        components = parse_cpl(cpl_file)
        rois = generate_rois(components, pixels_per_mm=50.0, image_size=(4000, 3000))

        u1 = next(r for r in rois if r.component_id == "U1")
        assert InspectionType.BLOB in u1.inspection_types
