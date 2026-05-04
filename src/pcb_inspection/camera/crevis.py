"""CREVIS GigE Vision camera driver using cvsCam SDK."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from pcb_inspection.camera import cvscam_ffi as cvsCam

if not cvsCam.is_available():
    logger.warning("libcvsCamCtrl.so not loadable. Real camera features unavailable.")


@dataclass
class CameraConfig:
    """Camera acquisition parameters."""

    exposure_us: float = 3000.0
    gain: float = 0.0
    pixel_format: str = "Mono8"
    width: int | None = None       # None = use camera default (full sensor)
    height: int | None = None
    trigger_mode: bool = False
    trigger_source: str = "Software"
    grab_timeout_ms: int = 5000
    buffer_count: int = 16


class CrevisCamera:
    """CREVIS GigE Vision camera wrapper.

    Lifecycle:
        cam = CrevisCamera(config)
        cam.open()               # connect + apply settings
        img = cam.grab()         # single capture -> np.ndarray (BGR)
        cam.close()              # release resources
    """

    def __init__(self, config: CameraConfig | None = None, device_index: int = 0) -> None:
        if not cvsCam.is_available():
            raise RuntimeError(
                "libcvsCamCtrl.so not loadable. Install the CREVIS cvsCam SDK first."
            )
        self.config = config or CameraConfig()
        self._device_index = device_index
        self._system: Any = None
        self._device: Any = None
        self._is_open = False
        self._is_acquiring = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def open(self) -> dict[str, str]:
        """Discover camera, open device, apply config.

        Returns:
            Device info dict (model, serial, ip, etc.).
        """
        self._system = cvsCam.CvsSystem()
        num = self._discover_cameras()
        if num == 0:
            self._system.Free()
            self._system = None
            raise RuntimeError("No CREVIS camera found on network.")

        if self._device_index >= num:
            self._system.Free()
            self._system = None
            raise RuntimeError(
                f"Device index {self._device_index} out of range (found {num} cameras)."
            )

        # Read device info before open
        info = self._read_enum_info(self._device_index)
        logger.info(
            "Opening camera %d: %s (S/N %s, IP %s)",
            self._device_index,
            info.get("model", "?"),
            info.get("serial", "?"),
            info.get("ip", "?"),
        )

        self._device = cvsCam.CvsDevice(self._device_index)
        self._device.Open()
        self._is_open = True

        self._apply_config()

        # Read detailed info after open
        info.update(self._read_device_info())
        logger.info("Camera ready: %s", info)
        return info

    def close(self) -> None:
        """Stop acquisition and release resources."""
        if self._is_acquiring:
            self.stop()
        if self._device is not None and self._is_open:
            self._device.Close()
            self._is_open = False
        if self._system is not None:
            self._system.Free()
            self._system = None
        logger.info("Camera closed.")

    @property
    def is_open(self) -> bool:
        return self._is_open

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    def grab(self) -> np.ndarray:
        """Grab a single image.

        For trigger_mode=False, uses SingleGrabImage (one-shot).
        For trigger_mode=True, sends software trigger then GrabImage.

        Returns:
            Image as numpy array (Mono8 -> H×W uint8, Color -> H×W×3 BGR).
        """
        self._ensure_open()

        if self.config.trigger_mode:
            return self._grab_triggered()
        return self._grab_single()

    def start(self) -> None:
        """Start continuous acquisition (for streaming / callback mode)."""
        self._ensure_open()
        if not self._is_acquiring:
            self._device.AcqStart()
            self._is_acquiring = True
            logger.info("Acquisition started.")

    def stop(self) -> None:
        """Stop continuous acquisition."""
        if self._is_acquiring:
            self._device.AcqStop()
            self._is_acquiring = False
            logger.info("Acquisition stopped.")

    def grab_continuous(self) -> np.ndarray:
        """Grab one frame from a running acquisition stream."""
        self._ensure_open()
        if not self._is_acquiring:
            self.start()
        buf = self._device.GrabImage()
        return self._buffer_to_numpy(buf)

    # ------------------------------------------------------------------
    # Feature access (GenICam registers)
    # ------------------------------------------------------------------

    def get_feature(self, name: str, dtype: str = "int") -> Any:
        """Read a GenICam feature register."""
        self._ensure_open()
        getters = {
            "int": self._device.GetIntReg,
            "float": self._device.GetFloatReg,
            "bool": self._device.GetBoolReg,
            "enum": self._device.GetEnumReg,
            "str": self._device.GetStrReg,
        }
        return getters[dtype](name)

    def set_feature(self, name: str, value: Any, dtype: str = "int") -> None:
        """Write a GenICam feature register."""
        self._ensure_open()
        setters = {
            "int": self._device.SetIntReg,
            "float": self._device.SetFloatReg,
            "bool": self._device.SetBoolReg,
            "enum": self._device.SetEnumReg,
            "str": self._device.SetStrReg,
            "cmd": lambda _n, _v=None: self._device.SetCmdReg(name),
        }
        if dtype == "cmd":
            setters["cmd"](name)
        else:
            setters[dtype](name, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_config(self) -> None:
        """Apply CameraConfig to the device."""
        cfg = self.config

        # Pixel format
        self._device.SetEnumReg("PixelFormat", cfg.pixel_format)

        # Resolution (optional override)
        if cfg.width is not None:
            self._device.SetIntReg("Width", cfg.width)
        if cfg.height is not None:
            self._device.SetIntReg("Height", cfg.height)

        # Exposure
        self._device.SetFloatReg("ExposureTime", cfg.exposure_us)

        # Gain (if supported)
        try:
            self._device.SetFloatReg("Gain", cfg.gain)
        except Exception:
            logger.debug("Gain register not available, skipping.")

        # Trigger
        if cfg.trigger_mode:
            self._device.SetEnumReg("TriggerMode", "On")
            self._device.SetEnumReg("TriggerSource", cfg.trigger_source)
        else:
            self._device.SetEnumReg("TriggerMode", "Off")

        # Buffer / timeout
        self._device.SetBufferCount(cfg.buffer_count)
        self._device.SetGrabTimeout(cfg.grab_timeout_ms)

        logger.info(
            "Config applied: %s, %dx%s, exposure=%.0fus, trigger=%s",
            cfg.pixel_format,
            cfg.width or self._device.GetIntReg("Width"),
            cfg.height or self._device.GetIntReg("Height"),
            cfg.exposure_us,
            "on" if cfg.trigger_mode else "off",
        )

    def _grab_single(self) -> np.ndarray:
        """One-shot grab without continuous acquisition."""
        buf = self._device.SingleGrabImage()
        return self._buffer_to_numpy(buf)

    def _grab_triggered(self) -> np.ndarray:
        """Software trigger + grab."""
        if not self._is_acquiring:
            self.start()
        self._device.SetCmdReg("TriggerSoftware")
        buf = self._device.GrabImage()
        return self._buffer_to_numpy(buf)

    @staticmethod
    def _buffer_to_numpy(buf: Any) -> np.ndarray:
        """Convert CVS_BUFFER to numpy array."""
        img = cvsCam.ConvertToNumpy(buf)
        return np.asarray(img)

    def _read_enum_info(self, index: int) -> dict[str, str]:
        """Read device info before Open()."""
        info: dict[str, str] = {}
        mapping = {
            "model": cvsCam.MCAM_DEVICEINFO_MODEL_NAME,
            "serial": cvsCam.MCAM_DEVICEINFO_SERIAL_NUMBER,
            "ip": cvsCam.MCAM_DEVICEINFO_IP_ADDRESS,
            "mac": cvsCam.MCAM_DEVICEINFO_MAC_ADDRESS,
            "user_id": cvsCam.MCAM_DEVICEINFO_USER_ID,
        }
        for key, code in mapping.items():
            try:
                info[key] = self._system.GetEnumDeviceInfo(index, code)
            except Exception:
                pass
        return info

    def _read_device_info(self) -> dict[str, str]:
        """Read additional info after Open()."""
        info: dict[str, str] = {}
        for key, reg in [
            ("vendor", "DeviceVendorName"),
            ("firmware", "DeviceVersion"),
        ]:
            try:
                info[key] = self._device.GetStrReg(reg)
            except Exception:
                pass
        return info

    def _discover_cameras(self) -> int:
        """Run UpdateDevice + count, with one extended retry on cold start.

        GigE discovery is broadcast-based and the first probe after a fresh
        process launch occasionally returns 0 even with the camera live on the
        link. Retry once with a longer timeout before giving up.
        """
        for attempt, timeout_ms in enumerate((1500, 5000)):
            self._system.UpdateDevice(timeout_ms=timeout_ms)
            num = self._system.GetAvailableCameraNum()
            if num > 0:
                return num
            logger.info(
                "Discovery returned 0 cameras (attempt %d, timeout=%dms); retrying.",
                attempt + 1,
                timeout_ms,
            )
        return 0

    def _ensure_open(self) -> None:
        if not self._is_open:
            raise RuntimeError("Camera is not open. Call open() first.")

    def __enter__(self) -> CrevisCamera:
        self.open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        if self._is_open:
            try:
                self.close()
            except Exception:
                pass
