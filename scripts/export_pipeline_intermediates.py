#!/usr/bin/env python3
"""
Прогон изображений через KYC Pipeline и сохранение всех промежуточных кадров.

На каждое входное фото — отдельная подпапка с этапами обработки.

Пример:
    python scripts/export_pipeline_intermediates.py \\
        --input seedream_output/russia_passport_swapped \\
        --output seedream_output/russia_passport_swapped/pipeline_steps
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kyc_document.document_processing import Pipeline

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _save_rgb(img: np.ndarray, path: Path) -> None:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def _save_images_list(img: Union[np.ndarray, list], out_dir: Path, stem: str) -> List[str]:
    saved: List[str] = []
    if img is None:
        return saved
    if isinstance(img, list):
        for i, part in enumerate(img):
            if isinstance(part, np.ndarray):
                name = f"{stem}_{i}.jpg" if len(img) > 1 else f"{stem}.jpg"
                _save_rgb(part, out_dir / name)
                saved.append(name)
        return saved
    if isinstance(img, np.ndarray):
        _save_rgb(img, out_dir / f"{stem}.jpg")
        saved.append(f"{stem}.jpg")
    return saved


def save_pipeline_intermediates(
    result,
    out_dir: Path,
    pipeline: Optional[Pipeline] = None,
    save_quality_annotated: bool = True,
) -> Dict[str, Any]:
    """Сохраняет промежуточные изображения и meta.json в out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_files: Dict[str, List[str]] = {}

    if result.meta_results.get("original_img") is not None:
        saved_files["original"] = _save_images_list(
            result.meta_results["original_img"], out_dir, "01_original"
        )

    if getattr(result, "rotated_image", None) is not None:
        saved_files["rotated"] = _save_images_list(result.rotated_image, out_dir, "02_rotated")

    dd = result.meta_results.get("DocDetector") or {}
    if dd.get("border_img") is not None:
        saved_files["doc_detection"] = _save_images_list(
            dd["border_img"], out_dir, "03_doc_detection"
        )

    warped = getattr(result, "img_with_fixed_perspective", None)
    if warped is not None:
        saved_files["fixed_perspective"] = _save_images_list(
            warped, out_dir, "04_fixed_perspective"
        )

    base_for_quality = warped
    if base_for_quality is None:
        base_for_quality = getattr(result, "rotated_image", None)
    if isinstance(base_for_quality, list) and base_for_quality:
        base_for_quality = base_for_quality[0]

    if pipeline is not None and save_quality_annotated and isinstance(base_for_quality, np.ndarray):
        try:
            blur_meta = pipeline.blur.predict_transform(base_for_quality)
            if blur_meta.get("warped_img") is not None:
                saved_files["blur_annotated"] = _save_images_list(
                    blur_meta["warped_img"], out_dir, "05_blur_annotated"
                )
        except Exception:
            pass
        try:
            glare_meta = pipeline.glare.predict_transform(base_for_quality)
            if glare_meta.get("warped_img") is not None:
                saved_files["glare_annotated"] = _save_images_list(
                    glare_meta["warped_img"], out_dir, "06_glare_annotated"
                )
        except Exception:
            pass

    if result.text_fields is not None and warped is not None:
        coords, _ = result.text_fields
        canvas = warped.copy() if isinstance(warped, np.ndarray) else warped[0].copy()
        for box in coords:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if len(box) > 5:
                label = str(box[-1])
                try:
                    conf = float(box[4])
                    label = f"{label} {conf:.2f}"
                except (TypeError, ValueError, IndexError):
                    pass
                cv2.putText(
                    canvas, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
                )
        saved_files["text_fields"] = _save_images_list(canvas, out_dir, "07_text_fields")

    seal = result.meta_results.get("PassportSealDetector") or {}
    for key, stem in (("seals_img", "08_seals"), ("fixed_seal_img", "09_fixed_seal")):
        if seal.get(key) is not None:
            saved_files[key] = _save_images_list(seal[key], out_dir, stem)

    words_dir = out_dir / "words"
    if result.words_patches:
        words_dir.mkdir(exist_ok=True)
        saved_files["words"] = []
        for field_name, block in result.words_patches.items():
            patches = block.get("patches", []) if isinstance(block, dict) else block
            for j, patch_info in enumerate(patches):
                patch = patch_info
                if isinstance(patch_info, (list, tuple)) and patch_info:
                    patch = patch_info[0]
                if not isinstance(patch, np.ndarray):
                    continue
                safe = re.sub(r"[^\w.-]+", "_", field_name)
                fname = f"{safe}_{j}.jpg"
                _save_rgb(patch, words_dir / fname)
                saved_files["words"].append(f"words/{fname}")

    tf_meta = None
    for key in ("TextFieldsDetectorRussia", "TextFieldsDetectorBelarus", "TextFieldsDetectorUSA"):
        if result.meta_results.get(key):
            tf_meta = result.meta_results[key]
            break
    if tf_meta and tf_meta.get("warped_img"):
        patches_dir = out_dir / "field_patches"
        patches_dir.mkdir(exist_ok=True)
        saved_files["field_patches"] = []
        bboxes = tf_meta.get("bbox", [])
        for i, patch in enumerate(tf_meta["warped_img"]):
            if not isinstance(patch, np.ndarray):
                continue
            label = bboxes[i][-1] if i < len(bboxes) and len(bboxes[i]) > 5 else f"field_{i}"
            safe = re.sub(r"[^\w.-]+", "_", str(label))
            fname = f"{safe}.jpg"
            _save_rgb(patch, patches_dir / fname)
            saved_files["field_patches"].append(f"field_patches/{fname}")

    ang = result.meta_results.get("Angle90") or {}
    meta = {
        "saved_files": saved_files,
        "doctype": result.doctype,
        "passport_page_type": result.passport_page_type,
        "quality": dict(result.quality) if result.quality else {},
        "angle90": {
            k: v for k, v in ang.items() if k != "warped_img"
        },
        "ocr": result.ocr,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return meta


def folder_name_for(path: Path) -> str:
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Экспорт промежуточных кадров pipeline")
    parser.add_argument(
        "--input", "-i", type=Path,
        default=PROJECT_ROOT / "seedream_output" / "russia_passport_swapped",
        help="Папка с входными изображениями",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=PROJECT_ROOT / "seedream_output" / "russia_passport_swapped" / "pipeline_steps",
        help="Корневая папка: внутри — подпапка на каждое фото",
    )
    parser.add_argument("--format", default="ONNX", choices=["ONNX", "OpenVINO", "TFlite"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--limit", type=int, default=0, help="Макс. число файлов (0 = все)")
    parser.add_argument("--img-size", type=int, default=1500)
    parser.add_argument("--no-ocr", action="store_true", help="Без OCR и word-патчей (быстрее)")
    parser.add_argument(
        "--no-quality-annotated", action="store_true",
        help="Не сохранять разметку blur/glare",
    )
    args = parser.parse_args()

    images = sorted(
        p for p in args.input.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"Нет изображений в {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Вход:  {args.input} ({len(images)} файлов)")
    print(f"Выход: {args.output}")
    print(f"Pipeline {args.format}/{args.device}...")

    pipeline = Pipeline(model_format=args.format, device=args.device, verbose=False)

    for i, img_path in enumerate(images, 1):
        sub = args.output / folder_name_for(img_path)
        print(f"[{i:>4}/{len(images)}] {img_path.name} -> {sub.name}/", flush=True)
        try:
            result = pipeline(
                str(img_path),
                ocr=not args.no_ocr,
                get_doc_borders=True,
                find_text_fields=not args.no_ocr,
                check_quality=True,
                low_quality=True,
                img_size=args.img_size,
                reset_text_fields=True,
            )
            save_pipeline_intermediates(
                result,
                sub,
                pipeline=pipeline,
                save_quality_annotated=not args.no_quality_annotated,
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            (sub / "error.txt").write_text(str(exc), encoding="utf-8")

    print(f"\nГотово: {args.output}")


if __name__ == "__main__":
    main()
