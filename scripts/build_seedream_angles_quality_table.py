#!/usr/bin/env python3
"""
Таблица по seedream_output/russia_passport_swapped:
  угол Angle90 + Blur/Glare из полного pipeline (без OCR).

Выход:
  seedream_output/russia_passport_swapped/angles_table.csv
  seedream_output/russia_passport_swapped/angles_table.md
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kyc_document.document_processing import Pipeline
from kyc_document.document_processing.pipeline_modules.doc_detector.image_transformation import (
    _quad_from_hull,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMG_DIR = PROJECT_ROOT / "seedream_output" / "russia_passport_swapped"
OUT_CSV = IMG_DIR / "angles_table.csv"
OUT_MD = IMG_DIR / "angles_table.md"


def doc_approx_area_fraction(result) -> Tuple[float, int]:
    """Доля площади кадра, занятая суммой аппроксимированных областей документа.

    Считается на кадре после Angle90 (как в DocDetector): для каждой маски —
    контур → convex hull → approxPolyDP-четырёхугольник (как в fix_perspective),
    площади суммируются.
    """
    dd = result.meta_results.get("DocDetector") or {}
    segments = dd.get("segm") or []
    img = result.rotated_image
    if img is None or not segments:
        return 0.0, 0

    h, w = img.shape[:2]
    img_area = float(h * w)
    if img_area <= 0:
        return 0.0, 0

    total_area = 0.0
    n_regions = 0
    for seg in segments:
        if seg is None or len(seg) < 3:
            continue
        seg_arr = np.asarray(seg, dtype=np.float32).reshape(-1, 2)
        seg_int = np.clip(seg_arr, [0, 0], [w - 1, h - 1]).astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [seg_int], -1, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        quad = _quad_from_hull(hull)
        if quad is not None:
            area = float(cv2.contourArea(quad.astype(np.float32)))
        else:
            area = float(cv2.contourArea(hull))
        if area > 0:
            total_area += area
            n_regions += 1

    return min(total_area / img_area, 1.0), n_regions


def quality_on_pipeline_image(pipeline: Pipeline, result) -> tuple:
    """Blur/Glare на том же кадре, что и в pipeline (после angle + DocDetector)."""
    img = result.img_with_fixed_perspective
    if img is None:
        img = result.rotated_image
    blur_label, blur_score = pipeline.blur.predict(img)["Blur"]
    glare_label, glare_score = pipeline.glare.predict(img)["Glare"]
    return blur_label, blur_score, glare_label, glare_score


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Макс. число файлов (0 = все)")
    args = parser.parse_args()

    images = sorted(p for p in IMG_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"Нет изображений в {IMG_DIR}")

    print(f"Найдено {len(images)} файлов")
    print("Инициализация Pipeline (ONNX/cpu)...")
    pipeline = Pipeline(model_format="ONNX", device="cpu", verbose=False)

    rows = []
    for i, path in enumerate(images, 1):
        res = pipeline(
            str(path),
            ocr=False,
            find_text_fields=False,
            check_quality=True,
            low_quality=True,
            reset_text_fields=False,
        )
        ang = res.meta_results.get("Angle90", {})
        q = res.quality or {}
        blur_p, blur_s, glare_p, glare_s = quality_on_pipeline_image(pipeline, res)
        doc_area_frac, doc_regions = doc_approx_area_fraction(res)

        rows.append({
            "filename": path.name,
            "angle_total_deg": round(float(ang.get("angle", 0)), 2),
            "skew_deg": round(float(ang.get("skew_deg", 0)), 2),
            "quadrant_deg": int(ang.get("quadrant_deg", 0)),
            "quadrant_source": ang.get("quadrant_source", ""),
            "angle_confidence": round(float(ang.get("confidence", 0)), 3),
            "doctype": res.doctype or "",
            "blur": blur_p,
            "blur_score": round(float(blur_s), 4),
            "glare": glare_p,
            "glare_score": round(float(glare_s), 4),
            "blur_pipeline": q.get("Blur", ""),
            "glare_pipeline": q.get("Glare", ""),
            "doc_area_frac": round(doc_area_frac, 4),
            "doc_regions_count": doc_regions,
        })
        if i % 5 == 0 or i == len(images):
            print(
                f"  [{i:>3}/{len(images)}] {path.name}  blur={blur_p} glare={glare_p} "
                f"doc_area={doc_area_frac:.2%} ({doc_regions} reg)",
                flush=True,
            )

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Углы и качество — russia_passport_swapped\n",
        "Pipeline: Angle90 → DocDetector → DocType → **Blur** / **Glare** (без OCR).\n",
        "**Doc area %** — сумма площадей approxPolyDP-четырёхугольников всех найденных "
        "областей / площадь кадра (после поворота).\n",
        f"Всего файлов: **{len(rows)}**\n",
        "| Файл | Угол (°) | Skew (°) | Квадр. | Doc area % | N reg. | Blur | Blur score | "
        "Glare | Glare score | DocType |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['filename']} | {r['angle_total_deg']:+.2f} | {r['skew_deg']:+.2f} | "
            f"{r['quadrant_deg']} | {r['doc_area_frac'] * 100:.1f}% | {r['doc_regions_count']} | "
            f"**{r['blur']}** | {r['blur_score']:.4f} | "
            f"**{r['glare']}** | {r['glare_score']:.4f} | {r['doctype']} |"
        )
    n_blur_bad = sum(1 for r in rows if r["blur"] == "bad")
    n_glare_bad = sum(1 for r in rows if r["glare"] == "bad")
    lines.append(f"\n**Blur bad:** {n_blur_bad}/{len(rows)}  |  **Glare bad:** {n_glare_bad}/{len(rows)}\n")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nCSV: {OUT_CSV}")
    print(f"MD:  {OUT_MD}")
    print(f"Blur bad: {n_blur_bad}, Glare bad: {n_glare_bad}")


if __name__ == "__main__":
    main()
