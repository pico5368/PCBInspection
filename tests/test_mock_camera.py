"""Tests for MockCamera and create_camera factory."""

from __future__ import annotations

import numpy as np
import pytest

from pcb_inspection.camera import CameraConfig, MockCamera, create_camera
from pcb_inspection.camera.mock import _MockDevice


def test_open_close_lifecycle() -> None:
    cam = MockCamera()
    assert cam.is_open is False

    info = cam.open()
    assert cam.is_open is True
    assert info["model"] == "MockCamera"
    assert info["serial"].startswith("MOCK-")

    cam.close()
    assert cam.is_open is False


def test_grab_before_open_raises() -> None:
    cam = MockCamera()
    with pytest.raises(RuntimeError, match="not open"):
        cam.grab()


def test_grab_default_synthetic_mono8() -> None:
    cfg = CameraConfig(width=640, height=480, pixel_format="Mono8")
    with MockCamera(config=cfg) as cam:
        img = cam.grab()
    assert img.shape == (480, 640)
    assert img.dtype == np.uint8
    assert img.min() < img.max()  # not constant


def test_grab_color_format() -> None:
    cfg = CameraConfig(width=320, height=240, pixel_format="BGR8")
    with MockCamera(config=cfg) as cam:
        img = cam.grab()
    assert img.shape == (240, 320, 3)
    assert img.dtype == np.uint8


def test_grab_mono16_dtype() -> None:
    cfg = CameraConfig(width=128, height=96, pixel_format="Mono16")
    with MockCamera(config=cfg) as cam:
        img = cam.grab()
    assert img.dtype == np.uint16


def test_explicit_image_returned() -> None:
    src = np.full((100, 100), 77, dtype=np.uint8)
    cfg = CameraConfig(width=100, height=100, pixel_format="Mono8")
    with MockCamera(config=cfg, image=src) as cam:
        out = cam.grab()
    assert out.shape == src.shape
    assert int(out.mean()) == 77


def test_explicit_image_resized_to_config() -> None:
    src = np.full((50, 50), 200, dtype=np.uint8)
    cfg = CameraConfig(width=200, height=200, pixel_format="Mono8")
    with MockCamera(config=cfg, image=src) as cam:
        out = cam.grab()
    assert out.shape == (200, 200)


def test_image_path_loaded(tmp_path) -> None:
    import cv2

    src = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), src)

    cfg = CameraConfig(width=64, height=64, pixel_format="Mono8")
    with MockCamera(config=cfg, image_path=path) as cam:
        out = cam.grab()
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_image_dir_cycles_frames(tmp_path) -> None:
    import cv2

    for i, val in enumerate([10, 80, 200]):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((32, 32), val, dtype=np.uint8))

    cfg = CameraConfig(width=32, height=32, pixel_format="Mono8")
    with MockCamera(config=cfg, image_dir=tmp_path) as cam:
        means = [int(cam.grab().mean()) for _ in range(6)]
    assert means == [10, 80, 200, 10, 80, 200]


def test_image_dir_empty_raises(tmp_path) -> None:
    cfg = CameraConfig(width=32, height=32)
    with pytest.raises(FileNotFoundError):
        MockCamera(config=cfg, image_dir=tmp_path).open()


def test_noise_perturbs_frames() -> None:
    src = np.full((64, 64), 128, dtype=np.uint8)
    cfg = CameraConfig(width=64, height=64, pixel_format="Mono8")
    with MockCamera(config=cfg, image=src, noise_std=5.0, seed=0) as cam:
        a = cam.grab()
        b = cam.grab()
    assert not np.array_equal(a, b)
    assert abs(int(a.mean()) - 128) < 3  # mean preserved


def test_continuous_acquisition() -> None:
    cfg = CameraConfig(width=64, height=64, pixel_format="Mono8")
    with MockCamera(config=cfg) as cam:
        cam.start()
        frames = [cam.grab_continuous() for _ in range(3)]
        cam.stop()
    assert all(f.shape == (64, 64) for f in frames)


def test_feature_get_set() -> None:
    cfg = CameraConfig(width=64, height=64, exposure_us=2000.0)
    with MockCamera(config=cfg) as cam:
        assert cam.get_feature("Width", "int") == 64
        assert cam.get_feature("ExposureTime", "float") == pytest.approx(2000.0)
        cam.set_feature("ExposureTime", 5000.0, "float")
        assert cam.get_feature("ExposureTime", "float") == pytest.approx(5000.0)


def test_device_introspection_methods() -> None:
    """Scripts call ``cam._device.GetIntRegRange`` etc. — keep the surface."""
    cfg = CameraConfig(width=128, height=128)
    with MockCamera(config=cfg) as cam:
        assert isinstance(cam._device, _MockDevice)
        mn, mx, inc = cam._device.GetIntRegRange("Width")
        assert mn < mx and inc > 0
        n = cam._device.GetEnumEntrySize("PixelFormat")
        formats = [cam._device.GetEnumEntryValue("PixelFormat", i) for i in range(n)]
        assert "Mono8" in formats


def test_create_camera_mock_explicit() -> None:
    cam = create_camera(backend="mock", config=CameraConfig(width=32, height=32))
    assert isinstance(cam, MockCamera)


def test_create_camera_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown camera backend"):
        create_camera(backend="bogus")


def test_create_camera_auto_falls_back_to_mock_without_sdk() -> None:
    """In CI/dev where cvsCam isn't installed, auto must yield MockCamera."""
    from pcb_inspection.camera import _cvscam_available

    if _cvscam_available():
        pytest.skip("cvsCam SDK is installed; auto resolves to crevis here.")
    cam = create_camera(backend="auto", config=CameraConfig(width=32, height=32))
    assert isinstance(cam, MockCamera)


def test_pipeline_end_to_end_with_mock() -> None:
    """Mock camera output should flow through InspectionPipeline without error."""
    from pcb_inspection.common.types import ComponentROI, InspectionType
    from pcb_inspection.pipeline.runner import InspectionPipeline
    from pcb_inspection.recipe.manager import Recipe

    cfg = CameraConfig(width=512, height=512, pixel_format="Mono8")
    with MockCamera(config=cfg) as cam:
        image = cam.grab()

    recipe = Recipe(
        recipe_id="mock_test",
        product_name="Mock",
        pixels_per_mm=50.0,
        image_size=(512, 512),
        thresholds={"reference": 0.8, "anomaly": 0.5},
    )
    rois = [
        ComponentROI(
            component_id=f"R{i}",
            component_type="test",
            bbox=(i * 100, i * 100, 80, 80),
            inspection_types=[InspectionType.REFERENCE],
        )
        for i in range(3)
    ]

    pipeline = InspectionPipeline(recipe)
    judgment = pipeline.run(image=image, rois=rois)
    assert judgment is not None
    assert len(judgment.component_results) >= 1
