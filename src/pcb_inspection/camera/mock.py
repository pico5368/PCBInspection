"""Mock camera for development/CI without the cvsCam SDK or hardware.

Drop-in replacement for :class:`CrevisCamera` that returns synthetic or
pre-loaded images. Public surface (open/close/grab/start/stop/grab_continuous/
get_feature/set_feature/_device) matches CrevisCamera so existing scripts and
the inspection pipeline run unchanged.

Image source priority: ``image`` (ndarray) > ``image_path`` > ``image_dir``
(round-robin) > synthetic checkerboard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from pcb_inspection.camera.crevis import CameraConfig

logger = logging.getLogger(__name__)


# Default synthetic frame size when CameraConfig.width/height not set.
# Smaller than HW spec (5120x5120) so dev/CI runs are fast; override via config.
_DEFAULT_W = 2048
_DEFAULT_H = 2048


@dataclass
class _MockDevice:
    """Stub of cvsCam CvsDevice exposing only the methods the project reads.

    Backs ``CrevisCamera._device`` access in scripts (e.g. ``test_camera.test_info``).
    """

    width: int
    height: int
    pixel_format: str
    exposure_us: float

    _int_ranges: dict[str, tuple[int, int, int]] = field(
        default_factory=lambda: {
            "Width": (16, 8192, 4),
            "Height": (16, 8192, 2),
        }
    )
    _float_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "ExposureTime": (1.0, 1_000_000.0),
            "Gain": (0.0, 24.0),
        }
    )
    _enum_entries: dict[str, list[str]] = field(
        default_factory=lambda: {
            "PixelFormat": ["Mono8", "Mono16", "BayerRG8", "BGR8"],
            "TriggerMode": ["Off", "On"],
            "TriggerSource": ["Software", "Line0", "Line1"],
        }
    )

    def GetIntReg(self, name: str) -> int:
        return {"Width": self.width, "Height": self.height}.get(name, 0)

    def SetIntReg(self, name: str, value: int) -> None:
        if name == "Width":
            self.width = int(value)
        elif name == "Height":
            self.height = int(value)

    def GetFloatReg(self, name: str) -> float:
        return {"ExposureTime": self.exposure_us}.get(name, 0.0)

    def SetFloatReg(self, name: str, value: float) -> None:
        if name == "ExposureTime":
            self.exposure_us = float(value)

    def GetBoolReg(self, name: str) -> bool:
        return False

    def SetBoolReg(self, name: str, value: bool) -> None:
        pass

    def GetEnumReg(self, name: str) -> str:
        if name == "PixelFormat":
            return self.pixel_format
        return ""

    def SetEnumReg(self, name: str, value: str) -> None:
        if name == "PixelFormat":
            self.pixel_format = value

    def GetStrReg(self, name: str) -> str:
        return {"DeviceVendorName": "MockVendor", "DeviceVersion": "mock-0.0.0"}.get(name, "")

    def SetStrReg(self, name: str, value: str) -> None:
        pass

    def SetCmdReg(self, name: str) -> None:
        pass

    def GetIntRegRange(self, name: str) -> tuple[int, int, int]:
        return self._int_ranges.get(name, (0, 0, 1))

    def GetFloatRegRange(self, name: str) -> tuple[float, float]:
        return self._float_ranges.get(name, (0.0, 0.0))

    def GetEnumEntrySize(self, name: str) -> int:
        return len(self._enum_entries.get(name, []))

    def GetEnumEntryValue(self, name: str, idx: int) -> str:
        return self._enum_entries.get(name, [])[idx]

    def SetBufferCount(self, count: int) -> None:
        pass

    def SetGrabTimeout(self, ms: int) -> None:
        pass

    def AcqStart(self) -> None:
        pass

    def AcqStop(self) -> None:
        pass


class MockCamera:
    """Synthetic / file-backed camera matching :class:`CrevisCamera` interface.

    Args:
        config: Camera config (exposure/gain/pixel_format/width/height/trigger).
        device_index: Reported only; no hardware access.
        image: Pre-loaded ndarray returned on every grab.
        image_path: Single image file (loaded once on open()).
        image_dir: Directory of images cycled per grab (live-view simulation).
        latency_ms: Simulated per-grab delay.
        noise_std: Per-grab Gaussian noise std (in pixel units).
        seed: RNG seed for synthetic image + noise reproducibility.
    """

    def __init__(
        self,
        config: CameraConfig | None = None,
        device_index: int = 0,
        *,
        image: np.ndarray | None = None,
        image_path: str | Path | None = None,
        image_dir: str | Path | None = None,
        latency_ms: float = 0.0,
        noise_std: float = 0.0,
        seed: int | None = 0,
    ) -> None:
        self.config = config or CameraConfig()
        self._device_index = device_index
        self._explicit_image = image
        self._image_path = Path(image_path) if image_path else None
        self._image_dir = Path(image_dir) if image_dir else None
        self._latency_s = latency_ms / 1000.0
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

        self._is_open = False
        self._is_acquiring = False
        self._device: _MockDevice | None = None
        self._frame_iter: Iterator[np.ndarray] | None = None
        self._base_frame: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def open(self) -> dict[str, str]:
        cfg = self.config
        w = cfg.width or _DEFAULT_W
        h = cfg.height or _DEFAULT_H
        self._device = _MockDevice(
            width=w,
            height=h,
            pixel_format=cfg.pixel_format,
            exposure_us=cfg.exposure_us,
        )

        self._base_frame, self._frame_iter = self._build_image_source(h, w)

        self._is_open = True
        info = {
            "model": "MockCamera",
            "serial": f"MOCK-{self._device_index:04d}",
            "ip": "0.0.0.0",
            "mac": "00:00:00:00:00:00",
            "user_id": "mock",
            "vendor": "MockVendor",
            "firmware": "mock-0.0.0",
        }
        logger.info(
            "MockCamera opened: %s, %dx%d, exposure=%.0fus, trigger=%s",
            cfg.pixel_format,
            w,
            h,
            cfg.exposure_us,
            "on" if cfg.trigger_mode else "off",
        )
        return info

    def close(self) -> None:
        if self._is_acquiring:
            self.stop()
        self._is_open = False
        self._device = None
        logger.info("MockCamera closed.")

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        self._ensure_open()
        if self._latency_s:
            time.sleep(self._latency_s)
        return self._next_frame()

    def start(self) -> None:
        self._ensure_open()
        self._is_acquiring = True

    def stop(self) -> None:
        self._is_acquiring = False

    def grab_continuous(self) -> np.ndarray:
        self._ensure_open()
        if not self._is_acquiring:
            self.start()
        if self._latency_s:
            time.sleep(self._latency_s)
        return self._next_frame()

    # ------------------------------------------------------------------
    # Feature access
    # ------------------------------------------------------------------

    def get_feature(self, name: str, dtype: str = "int") -> Any:
        self._ensure_open()
        assert self._device is not None
        getters = {
            "int": self._device.GetIntReg,
            "float": self._device.GetFloatReg,
            "bool": self._device.GetBoolReg,
            "enum": self._device.GetEnumReg,
            "str": self._device.GetStrReg,
        }
        return getters[dtype](name)

    def set_feature(self, name: str, value: Any, dtype: str = "int") -> None:
        self._ensure_open()
        assert self._device is not None
        if dtype == "cmd":
            self._device.SetCmdReg(name)
            return
        setters = {
            "int": self._device.SetIntReg,
            "float": self._device.SetFloatReg,
            "bool": self._device.SetBoolReg,
            "enum": self._device.SetEnumReg,
            "str": self._device.SetStrReg,
        }
        setters[dtype](name, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_image_source(
        self, h: int, w: int
    ) -> tuple[np.ndarray, Iterator[np.ndarray]]:
        if self._explicit_image is not None:
            base = self._coerce(self._explicit_image, h, w)
            return base, cycle([base])

        if self._image_path is not None:
            base = self._load_file(self._image_path, h, w)
            return base, cycle([base])

        if self._image_dir is not None:
            frames = self._load_dir(self._image_dir, h, w)
            if not frames:
                raise FileNotFoundError(
                    f"No images found in {self._image_dir} (looking for png/jpg/bmp/tif)."
                )
            return frames[0], cycle(frames)

        synthetic = self._synthesize(h, w)
        return synthetic, cycle([synthetic])

    def _next_frame(self) -> np.ndarray:
        assert self._frame_iter is not None
        frame = next(self._frame_iter)
        if self._noise_std > 0:
            noise = self._rng.normal(0.0, self._noise_std, size=frame.shape)
            frame = np.clip(frame.astype(np.float32) + noise, 0, _dtype_max(frame.dtype))
            frame = frame.astype(self._dtype())
        else:
            frame = frame.copy()
        return frame

    def _synthesize(self, h: int, w: int) -> np.ndarray:
        """Checkerboard + gradient + noise — visually nontrivial, fast to build."""
        tile = max(64, min(h, w) // 16)
        yy, xx = np.indices((h, w))
        board = (((yy // tile) + (xx // tile)) % 2).astype(np.float32) * 80.0 + 60.0

        gradient = (xx / max(w - 1, 1)).astype(np.float32) * 40.0
        noise = self._rng.normal(0.0, 4.0, size=(h, w)).astype(np.float32)
        gray = np.clip(board + gradient + noise, 0, 255).astype(np.uint8)

        return self._to_pixel_format(gray)

    def _load_file(self, path: Path, h: int, w: int) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Mock image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return self._coerce(img, h, w)

    def _load_dir(self, path: Path, h: int, w: int) -> list[np.ndarray]:
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
        return [self._load_file(p, h, w) for p in files]

    def _coerce(self, img: np.ndarray, h: int, w: int) -> np.ndarray:
        """Resize + convert to match configured pixel format."""
        cur_h, cur_w = img.shape[:2]
        if (cur_h, cur_w) != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        is_color = img.ndim == 3 and img.shape[2] >= 3
        wants_color = self._wants_color()

        if wants_color and not is_color:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif not wants_color and is_color:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img.astype(self._dtype())

    def _to_pixel_format(self, gray: np.ndarray) -> np.ndarray:
        if self._wants_color():
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(self._dtype())
        return gray.astype(self._dtype())

    def _wants_color(self) -> bool:
        fmt = self.config.pixel_format.lower()
        return fmt.startswith(("bgr", "rgb"))

    def _dtype(self) -> np.dtype:
        return np.dtype(np.uint16) if "16" in self.config.pixel_format else np.dtype(np.uint8)

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("MockCamera is not open. Call open() first.")

    def __enter__(self) -> MockCamera:
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _dtype_max(dtype: np.dtype) -> int:
    return 65535 if dtype == np.uint16 else 255
