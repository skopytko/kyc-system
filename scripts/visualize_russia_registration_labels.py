#!/usr/bin/env python3
"""Визуализация масок (labels класса 1) на cropped изображениях Russia registration."""

import argparse
import cv2
import numpy as np
from pathlib import Path


def parse_yolo_seg(label_path: Path, target_class: int = 1) -> list[np.ndarray]:
    """Парсит YOLO seg, возвращает полигоны (нормализованные 0-1)."""
    if not label_path.exists():
        return []
    polygons = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            if cls != target_class:
                continue
            coords = [float(x) for x in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            pts = np.array(coords).reshape(-1, 2)
            polygons.append(pts)
    return polygons


def main():
    parser = argparse.ArgumentParser(description="Визуализация labels на cropped")
    parser.add_argument(
        "--input",
        default="dataset/russia/passport/registration_page/cropped",
        help="Папка с cropped изображениями",
    )
    parser.add_argument(
        "--output",
        default="dataset/russia/passport/registration_page/cropped_viz",
        help="Папка для изображений с масками",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    labels_dir = input_dir / "labels"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR
    alpha_fill = 0.35

    ext = (".jpg", ".jpeg", ".png")
    images = [f for f in input_dir.iterdir() if f.suffix.lower() in ext and f.is_file()]

    for img_path in sorted(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        polygons = parse_yolo_seg(label_path)

        overlay = img.copy()
        for i, pts_norm in enumerate(polygons):
            pts_px = (pts_norm * np.array([w, h])).astype(np.int32)
            color = colors[i % len(colors)]
            cv2.fillPoly(overlay, [pts_px], color)
            cv2.polylines(img, [pts_px], True, color, 2)

        result = cv2.addWeighted(overlay, alpha_fill, img, 1 - alpha_fill, 0)
        # Контур поверх
        for i, pts_norm in enumerate(polygons):
            pts_px = (pts_norm * np.array([w, h])).astype(np.int32)
            cv2.polylines(result, [pts_px], True, colors[i % len(colors)], 2)

        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), result)
        print(f"  ✓ {img_path.name} -> {out_path.name}")

    print(f"\nГотово: {len(images)} изображений в {output_dir}")


if __name__ == "__main__":
    main()
