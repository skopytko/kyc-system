#!/usr/bin/env python3
"""Обрезка областей класса 0 из raw по YOLO segmentation labels.
   Аппроксимирует полигоны до 4 точек, объединяет левую и правую области в одно изображение.
   Генерирует labels класса 1 с преобразованными координатами под cropped."""

import argparse
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from pathlib import Path


def order_points(pts: np.ndarray) -> np.ndarray:
    """Сортировка точек: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(
    image: np.ndarray, pts: np.ndarray, return_matrix: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, int, int]]:
    """Исправление перспективы по 4 точкам, обрезка области."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    if return_matrix:
        return warped, M, maxWidth, maxHeight
    return warped


def approximate_to_four_points(points: np.ndarray, epsilon: float = 0.02) -> np.ndarray:
    """Аппроксимация полигона до 4 точек."""
    if len(points) <= 4:
        return points.astype(np.float32)

    pts_arr = points.astype(np.float32).reshape(-1, 1, 2)
    perimeter = cv2.arcLength(pts_arr, True)
    approx = cv2.approxPolyDP(pts_arr, epsilon * perimeter, True)

    while len(approx) > 4 and epsilon < 0.5:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(pts_arr, epsilon * perimeter, True)

    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)
    if len(approx) < 4:
        return points[:4].astype(np.float32)

    # >4 точек: берём углы minAreaRect
    rect = cv2.minAreaRect(pts_arr.reshape(-1, 2))
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def parse_yolo_seg_label(label_path: Path, target_class: int = 0) -> list[np.ndarray]:
    """Читает YOLO segmentation label, возвращает список полигонов класса target_class.
    Координаты нормализованы (0-1)."""
    if not label_path.exists():
        return []

    polygons = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + min 3 points (6 values)
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


def _transform_class1_to_cropped(
    poly_norm: np.ndarray,
    img_w: int,
    img_h: int,
    quads_sorted: list[np.ndarray],
    transforms: list[tuple[np.ndarray, int, int]],
    scales: list[float],
    offset_x: float,
) -> Optional[np.ndarray]:
    """Преобразует полигон класса 1 из оригинальных координат в cropped. Пиксели."""
    pts_px = poly_norm * np.array([img_w, img_h])
    centroid = pts_px.mean(axis=0)

    # Определяем регион по centroid (внутри какого quad)
    region_idx = 0
    for i, quad in enumerate(quads_sorted):
        if cv2.pointPolygonTest(quad.astype(np.float32), (centroid[0], centroid[1]), False) >= 0:
            region_idx = i
            break

    if region_idx >= len(transforms):
        return None

    M, _, _ = transforms[region_idx]
    scale = scales[region_idx]

    # perspectiveTransform ожидает Nx1x2
    pts_in = pts_px.astype(np.float32).reshape(-1, 1, 2)
    pts_out = cv2.perspectiveTransform(pts_in, M)

    # Масштаб при resize + смещение для правой области
    x_offset = offset_x if region_idx == 1 else 0.0
    transformed = []
    for pt in pts_out.reshape(-1, 2):
        x = pt[0] * scale + x_offset
        y = pt[1] * scale
        transformed.append([x, y])

    return np.array(transformed, dtype=np.float32)


def process_image(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    labels_output_dir: Optional[Path],
    target_class: int = 0,
) -> bool:
    """Обрабатывает одно изображение: вырезает области target_class, объединяет лево+право.
    Сохраняет labels класса 1 с преобразованными координатами."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠ Не удалось загрузить {image_path.name}")
        return False

    h, w = img.shape[:2]
    polygons = parse_yolo_seg_label(label_path, target_class=target_class)

    if not polygons:
        print(f"  ⚠ Нет полигонов класса {target_class} в {label_path.name}")
        return False

    # Конвертируем нормализованные координаты в пиксели
    pixel_polygons = [pts * np.array([w, h]) for pts in polygons]

    # Аппроксимируем до 4 точек
    quads = []
    for pts in pixel_polygons:
        quad = approximate_to_four_points(pts)
        quads.append(quad)

    # Сортируем по среднему X: левая область первая
    centers_x = [q[:, 0].mean() for q in quads]
    sorted_indices = np.argsort(centers_x)
    quads_sorted = [quads[i] for i in sorted_indices]

    cropped_regions = []
    transforms = []  # (M, crop_w, crop_h) для каждой области

    for i in sorted_indices:
        warped, M, cw, ch = four_point_transform(img, quads[i], return_matrix=True)
        cropped_regions.append(warped)
        transforms.append((M, cw, ch))

    # Выравниваем по высоте
    max_h = max(r.shape[0] for r in cropped_regions)
    scales = [max_h / r.shape[0] for r in cropped_regions]
    resized = []
    for i, r in enumerate(cropped_regions):
        if r.shape[0] != max_h:
            r = cv2.resize(r, (int(r.shape[1] * scales[i]), max_h))
        resized.append(r)

    combined = np.hstack(resized)
    total_w = combined.shape[1]
    total_h = combined.shape[0]
    w_left = resized[0].shape[1] if len(resized) > 0 else 0

    # Сохраняем изображение
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)

    # Генерируем labels класса 1
    if labels_output_dir:
        class1_polys = parse_yolo_seg_label(label_path, target_class=1)
        if class1_polys:
            label_lines = []
            for poly_norm in class1_polys:
                tf = _transform_class1_to_cropped(
                    poly_norm, w, h, quads_sorted, transforms, scales, float(w_left)
                )
                if tf is not None and len(tf) >= 3:
                    # Нормализуем для YOLO
                    tf_norm = tf / np.array([total_w, total_h])
                    tf_norm = np.clip(tf_norm, 0.0, 1.0)
                    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in tf_norm.tolist())
                    label_lines.append(f"1 {coords}")

            if label_lines:
                labels_output_dir.mkdir(parents=True, exist_ok=True)
                label_out = labels_output_dir / (output_path.stem + ".txt")
                label_out.write_text("\n".join(label_lines), encoding="utf-8")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Crop class-0 regions from Russia registration raw images"
    )
    parser.add_argument(
        "--input",
        default="dataset/russia/passport/registration_page/raw",
        help="Папка с изображениями (рядом или в подпапке labels — аннотации)",
    )
    parser.add_argument(
        "--output",
        default="dataset/russia/passport/registration_page/cropped",
        help="Папка для cropped изображений",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="ID класса областей для crop (по умолчанию 0)",
    )
    parser.add_argument(
        "--no-export-labels",
        action="store_true",
        help="Не сохранять labels класса 1",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    labels_dir = input_dir / "labels"
    output_dir = Path(args.output)

    ext = (".jpg", ".jpeg", ".png")
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in ext and f.is_file()]

    print(f"Найдено {len(image_files)} изображений")
    print(f"Labels: {labels_dir}")
    print(f"Output: {output_dir}")

    labels_output_dir = None if args.no_export_labels else output_dir / "labels"

    saved = 0
    for img_path in sorted(image_files):
        label_path = labels_dir / (img_path.stem + ".txt")
        out_path = output_dir / (img_path.stem + img_path.suffix)

        if process_image(
            img_path, label_path, out_path, labels_output_dir, target_class=args.class_id
        ):
            saved += 1
            print(f"  ✓ {img_path.name} -> {out_path.name}")

    print(f"\nГотово: {saved}/{len(image_files)} сохранено в {output_dir}")
    if labels_output_dir:
        print(f"Labels класса 1: {labels_output_dir}")


if __name__ == "__main__":
    main()
