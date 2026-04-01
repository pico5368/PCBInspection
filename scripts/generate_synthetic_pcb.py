"""Generate synthetic PCB images for alignment and ROI testing.

Creates:
- A reference (golden) PCB image with fiducial markers and components
- A test image with slight offset/rotation to test alignment
- A CPL file matching the component layout
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np


def generate_pcb_image(
    width: int = 2000,
    height: int = 1500,
    num_components: int = 30,
    seed: int = 42,
) -> tuple[np.ndarray, list[dict], list[tuple[float, float]]]:
    """Generate a synthetic PCB image.

    Returns:
        (image, components, fiducial_positions)
    """
    rng = random.Random(seed)
    img = np.full((height, width, 3), (30, 80, 40), dtype=np.uint8)  # dark green PCB

    # Draw PCB traces (random lines)
    for _ in range(50):
        x1, y1 = rng.randint(50, width - 50), rng.randint(50, height - 50)
        x2, y2 = rng.randint(50, width - 50), rng.randint(50, height - 50)
        cv2.line(img, (x1, y1), (x2, y2), (20, 60, 30), rng.randint(1, 3))

    # Draw copper pads (background)
    for _ in range(80):
        cx, cy = rng.randint(100, width - 100), rng.randint(100, height - 100)
        pw, ph = rng.randint(8, 20), rng.randint(8, 20)
        cv2.rectangle(img, (cx - pw, cy - ph), (cx + pw, cy + ph), (50, 120, 70), -1)

    # Fiducial markers — two corners
    fiducials = [
        (100.0, 100.0),
        (width - 100.0, height - 100.0),
    ]
    for fx, fy in fiducials:
        _draw_fiducial(img, int(fx), int(fy))

    # Components
    components = []
    package_types = [
        ("0402", 20, 10),
        ("0603", 30, 15),
        ("0805", 40, 22),
        ("1206", 60, 30),
        ("SOT23", 50, 30),
        ("SOP8", 80, 60),
        ("QFP48", 140, 140),
    ]

    placed_rects = []
    for i in range(num_components):
        pkg_name, pkg_w, pkg_h = rng.choice(package_types)
        rotation = rng.choice([0, 90, 180, 270])

        # Try to place without overlap
        for _ in range(50):
            cx = rng.randint(200, width - 200)
            cy = rng.randint(200, height - 200)
            rect = (cx - pkg_w, cy - pkg_h, cx + pkg_w, cy + pkg_h)
            if not any(_rects_overlap(rect, r) for r in placed_rects):
                placed_rects.append(rect)
                break
        else:
            continue

        # Draw component
        _draw_component(img, cx, cy, pkg_w, pkg_h, pkg_name, rotation, rng)

        # Designator
        prefix = {"0402": "R", "0603": "R", "0805": "C", "1206": "C",
                   "SOT23": "Q", "SOP8": "U", "QFP48": "U"}
        des = f"{prefix.get(pkg_name, 'X')}{i + 1}"

        components.append({
            "designator": des,
            "package": pkg_name,
            "x_mm": cx / 50.0,  # pixels_per_mm = 50
            "y_mm": cy / 50.0,
            "rotation": rotation,
            "layer": "top",
            "value": rng.choice(["10K", "4.7K", "100nF", "10uF", "MCU", "LDO", ""]),
        })

    return img, components, fiducials


def generate_test_image(
    reference: np.ndarray,
    offset_x: float = 15.0,
    offset_y: float = -10.0,
    rotation_deg: float = 1.5,
    noise_std: float = 5.0,
) -> np.ndarray:
    """Generate a test image with offset, rotation, and noise."""
    h, w = reference.shape[:2]
    center = (w / 2, h / 2)

    # Rotation + translation matrix
    M = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    M[0, 2] += offset_x
    M[1, 2] += offset_y

    transformed = cv2.warpAffine(reference, M, (w, h), borderValue=(30, 80, 40))

    # Add noise
    noise = np.random.normal(0, noise_std, transformed.shape).astype(np.int16)
    noisy = np.clip(transformed.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight brightness variation
    hsv = cv2.cvtColor(noisy, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-10, 10), 0, 255)
    noisy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return noisy


def save_cpl(components: list[dict], output_path: Path) -> None:
    """Save component list as CPL CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Designator", "Package", "X(mm)", "Y(mm)", "Rotation", "Layer", "Value"])
        for c in components:
            writer.writerow([
                c["designator"], c["package"],
                f"{c['x_mm']:.2f}", f"{c['y_mm']:.2f}",
                str(int(c["rotation"])), c["layer"], c["value"],
            ])


def _draw_fiducial(img: np.ndarray, cx: int, cy: int) -> None:
    """Draw a circular fiducial marker."""
    # Copper ring
    cv2.circle(img, (cx, cy), 25, (80, 160, 100), -1)
    # Inner circle (bright)
    cv2.circle(img, (cx, cy), 12, (200, 220, 210), -1)
    # Center dot
    cv2.circle(img, (cx, cy), 4, (240, 245, 240), -1)


def _draw_component(
    img: np.ndarray, cx: int, cy: int,
    w: int, h: int, pkg: str, rotation: int,
    rng: random.Random,
) -> None:
    """Draw a synthetic component."""
    # Body
    body_color = rng.choice([
        (40, 40, 40),    # black IC
        (60, 50, 40),    # brown resistor
        (80, 70, 50),    # tan capacitor
        (100, 100, 100), # gray
    ])

    if rotation in (90, 270):
        w, h = h, w

    cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), body_color, -1)

    # Solder pads
    pad_color = (160, 180, 170)
    if pkg.startswith(("0", "1")):  # chip components — pads on left/right
        pad_w = max(4, w // 4)
        cv2.rectangle(img, (cx - w // 2 - pad_w, cy - h // 2), (cx - w // 2, cy + h // 2), pad_color, -1)
        cv2.rectangle(img, (cx + w // 2, cy - h // 2), (cx + w // 2 + pad_w, cy + h // 2), pad_color, -1)
    elif pkg.startswith(("QFP",)):  # QFP — pads on all sides
        pad_len = 8
        for px in range(cx - w // 2 + 5, cx + w // 2, 12):
            cv2.rectangle(img, (px - 2, cy - h // 2 - pad_len), (px + 2, cy - h // 2), pad_color, -1)
            cv2.rectangle(img, (px - 2, cy + h // 2), (px + 2, cy + h // 2 + pad_len), pad_color, -1)
        for py in range(cy - h // 2 + 5, cy + h // 2, 12):
            cv2.rectangle(img, (cx - w // 2 - pad_len, py - 2), (cx - w // 2, py + 2), pad_color, -1)
            cv2.rectangle(img, (cx + w // 2, py - 2), (cx + w // 2 + pad_len, py + 2), pad_color, -1)
    else:  # SOP/SOT — pads on two sides
        pad_len = 6
        for py in range(cy - h // 2 + 5, cy + h // 2, 10):
            cv2.rectangle(img, (cx - w // 2 - pad_len, py - 2), (cx - w // 2, py + 2), pad_color, -1)
            cv2.rectangle(img, (cx + w // 2, py - 2), (cx + w // 2 + pad_len, py + 2), pad_color, -1)

    # Polarity mark (dot on pin 1)
    if pkg in ("SOP8", "QFP48", "SOT23"):
        cv2.circle(img, (cx - w // 2 + 6, cy - h // 2 + 6), 3, (220, 220, 200), -1)

    # Component text
    if w > 40:
        cv2.putText(img, pkg[:4], (cx - w // 4, cy + 4),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.25, (150, 150, 140), 1)


def _rects_overlap(r1: tuple, r2: tuple, margin: int = 20) -> bool:
    """Check if two rectangles overlap with margin."""
    return not (r1[2] + margin < r2[0] or r2[2] + margin < r1[0] or
                r1[3] + margin < r2[1] or r2[3] + margin < r1[1])


def main():
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic PCB...")
    ref_img, components, fiducials = generate_pcb_image()
    cv2.imwrite(str(output_dir / "reference.png"), ref_img)
    print(f"  Reference: {ref_img.shape}, {len(components)} components, {len(fiducials)} fiducials")

    test_img = generate_test_image(ref_img)
    cv2.imwrite(str(output_dir / "test_offset.png"), test_img)
    print(f"  Test image: saved with offset+rotation+noise")

    save_cpl(components, output_dir / "placement.csv")
    print(f"  CPL: {len(components)} components")

    # Save fiducial positions
    with (output_dir / "fiducials.txt").open("w") as f:
        for fx, fy in fiducials:
            f.write(f"{fx},{fy}\n")

    print(f"\nOutput: {output_dir.resolve()}")
    print("Done!")


if __name__ == "__main__":
    main()
