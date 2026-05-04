"""Step A: Camera connection + single image capture test.

Usage:
    python scripts/test_camera.py                  # basic single grab
    python scripts/test_camera.py --live            # continuous live view
    python scripts/test_camera.py --save            # grab and save to file
    python scripts/test_camera.py --exposure 5000   # custom exposure (us)
    python scripts/test_camera.py --info            # print camera info only
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pcb_inspection.camera import CameraConfig, create_camera
from pcb_inspection.camera.crevis import CrevisCamera  # type alias used in annotations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "captures"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CREVIS IMX530 camera test")
    p.add_argument("--live", action="store_true", help="Continuous live view")
    p.add_argument("--save", action="store_true", help="Save captured image to file")
    p.add_argument("--info", action="store_true", help="Print camera info and exit")
    p.add_argument("--exposure", type=float, default=3000.0, help="Exposure time (us)")
    p.add_argument("--gain", type=float, default=0.0, help="Gain (dB)")
    p.add_argument("--pixel-format", default="Mono8", help="Pixel format (Mono8, BayerRG8, etc.)")
    p.add_argument("--trigger", action="store_true", help="Use software trigger mode")
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument(
        "--backend",
        choices=("auto", "crevis", "mock"),
        default="auto",
        help="Camera backend (auto: crevis if SDK present else mock)",
    )
    p.add_argument("--mock-image", type=str, help="Path to image used by mock backend")
    p.add_argument("--mock-dir", type=str, help="Directory of images cycled by mock backend")
    p.add_argument(
        "--mock-latency-ms", type=float, default=0.0, help="Simulated grab latency for mock"
    )
    return p.parse_args()


def print_image_stats(img: np.ndarray) -> None:
    """Print basic image statistics."""
    print(f"  Shape     : {img.shape}")
    print(f"  Dtype     : {img.dtype}")
    print(f"  Min/Max   : {img.min()} / {img.max()}")
    print(f"  Mean/Std  : {img.mean():.1f} / {img.std():.1f}")
    size_mb = img.nbytes / (1024 * 1024)
    print(f"  Size      : {size_mb:.1f} MB")


def save_image(img: np.ndarray, tag: str = "") -> Path:
    """Save image with timestamp."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    path = SAVE_DIR / f"capture_{ts}{suffix}.png"
    cv2.imwrite(str(path), img)
    print(f"  Saved: {path}")
    return path


def test_info(cam: CrevisCamera) -> None:
    """Print camera info and feature ranges."""
    info = cam.open()
    print("\n=== Camera Info ===")
    for k, v in info.items():
        print(f"  {k:20s}: {v}")

    print("\n=== Feature Ranges ===")
    for reg, dtype in [("Width", "int"), ("Height", "int"), ("ExposureTime", "float")]:
        try:
            if dtype == "int":
                mn, mx, inc = cam._device.GetIntRegRange(reg)
                print(f"  {reg:20s}: {mn} ~ {mx} (step {inc})")
            else:
                mn, mx = cam._device.GetFloatRegRange(reg)
                print(f"  {reg:20s}: {mn} ~ {mx}")
        except Exception:
            pass

    # Pixel format options
    try:
        n = cam._device.GetEnumEntrySize("PixelFormat")
        formats = [cam._device.GetEnumEntryValue("PixelFormat", i) for i in range(n)]
        print(f"  {'PixelFormat':20s}: {', '.join(formats)}")
    except Exception:
        pass

    print()


def test_single_grab(cam: CrevisCamera, args: argparse.Namespace) -> None:
    """Single image grab test."""
    cam.open()

    print("\n=== Single Grab Test ===")
    print("Capturing...")

    img = cam.grab()

    print("  Capture OK!")
    print_image_stats(img)

    if args.save:
        save_image(img, "single")

    # Display (resized for screen)
    display = _resize_for_display(img)
    cv2.imshow("Single Grab - Press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_live_view(cam: CrevisCamera, args: argparse.Namespace) -> None:
    """Continuous live view test."""
    cam.open()

    print("\n=== Live View ===")
    print("Press 'q' to quit, 's' to save current frame")

    cam.start()
    frame_count = 0

    while True:
        try:
            img = cam.grab_continuous()
            frame_count += 1

            display = _resize_for_display(img)

            # Overlay frame info
            h, w = img.shape[:2]
            text = f"Frame: {frame_count} | {w}x{h} | Mean: {img.mean():.0f}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Live View - 'q' quit, 's' save", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                save_image(img, f"live_{frame_count:06d}")

        except Exception as e:
            if "-1011" in str(e):
                continue  # timeout, retry
            logger.error("Grab error: %s", e)
            break

    cam.stop()
    cv2.destroyAllWindows()
    print(f"Total frames: {frame_count}")


def _resize_for_display(img: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """Resize large images to fit display."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h))


def main() -> None:
    args = parse_args()

    config = CameraConfig(
        exposure_us=args.exposure,
        gain=args.gain,
        pixel_format=args.pixel_format,
        trigger_mode=args.trigger,
    )

    mock_kwargs: dict = {"latency_ms": args.mock_latency_ms}
    if args.mock_image:
        mock_kwargs["image_path"] = args.mock_image
    if args.mock_dir:
        mock_kwargs["image_dir"] = args.mock_dir

    cam = create_camera(
        backend=args.backend,
        config=config,
        device_index=args.device,
        **mock_kwargs,
    )
    if args.backend == "auto" and not isinstance(cam, CrevisCamera):
        logger.warning("cvsCam SDK not found — falling back to MockCamera.")

    try:
        if args.info:
            test_info(cam)
        elif args.live:
            test_live_view(cam, args)
        else:
            test_single_grab(cam, args)
    finally:
        cam.close()


if __name__ == "__main__":
    main()
