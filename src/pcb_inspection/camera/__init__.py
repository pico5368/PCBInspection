"""Camera acquisition module.

Real hardware: :class:`CrevisCamera` (CREVIS GigE Vision via cvsCam SDK).
Dev/CI fallback: :class:`MockCamera` (synthetic or file-backed frames).

Use :func:`create_camera` for backend-agnostic instantiation.
"""

from __future__ import annotations

from typing import Any

from pcb_inspection.camera.crevis import CameraConfig, CrevisCamera
from pcb_inspection.camera.mock import MockCamera

__all__ = ["CameraConfig", "CrevisCamera", "MockCamera", "create_camera"]


def _cvscam_available() -> bool:
    from pcb_inspection.camera import cvscam_ffi

    return cvscam_ffi.is_available()


def create_camera(
    backend: str = "auto",
    config: CameraConfig | None = None,
    device_index: int = 0,
    **mock_kwargs: Any,
) -> CrevisCamera | MockCamera:
    """Instantiate a camera by backend name.

    Args:
        backend: "auto" (crevis if SDK present, else mock), "crevis", or "mock".
        config: CameraConfig (exposure, pixel format, etc.).
        device_index: Hardware index for crevis; reported only for mock.
        **mock_kwargs: Forwarded to MockCamera (image, image_path, image_dir,
            latency_ms, noise_std, seed).

    Raises:
        ValueError: Unknown backend.
        RuntimeError: backend="crevis" but cvsCam SDK not installed.
    """
    if backend == "auto":
        backend = "crevis" if _cvscam_available() else "mock"

    if backend == "crevis":
        return CrevisCamera(config=config, device_index=device_index)
    if backend == "mock":
        return MockCamera(config=config, device_index=device_index, **mock_kwargs)
    raise ValueError(f"Unknown camera backend: {backend!r} (expected auto|crevis|mock)")
