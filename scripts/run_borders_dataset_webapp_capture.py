#!/usr/bin/env python3
"""
Прогон с той же логикой, что POST /api/ocr в webapp:
Pipeline(..., check_quality=True) → evaluate_capture(full_report, meta_results).

Источник картинок:
  yolo — split model_borders/dataset (train/val/test).
  cropped_random — случайная подвыборка файлов из dataset/**/cropped (без папки labels).

Запуск из корня репозитория:
  PYTHONPATH=. python scripts/run_borders_dataset_webapp_capture.py
  PYTHONPATH=. python scripts/run_borders_dataset_webapp_capture.py \\
      --image-source cropped_random --random-cropped 200 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from kyc_document.document_processing import Pipeline
from webapp.capture_validation import evaluate_capture

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_EXTENSIONS = tuple(ALLOWED_SUFFIXES)


def sanitize_for_json(obj: Any) -> Any:
    """Как в webapp.main — для совместимого вызова evaluate_capture."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dataset_has_border_gt(label_path: Optional[Path]) -> Optional[bool]:
    """True — в разметке есть bbox класса border (0); None — файла лейбла нет."""
    if label_path is None or not label_path.is_file():
        return None
    try:
        text = label_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("0 ") or s == "0":
            return True
    return False


def _pct(num: float, den: float, *, ndigits: int = 1) -> str:
    if not den:
        return "0%"
    return f"{round(100.0 * float(num) / float(den), ndigits)}%"


def _dataset_country_from_path_marker(path_or_rel: str) -> str:
    """Страна по подстроке пути dataset/usa|belarus|russia/..."""
    p = Path(path_or_rel.replace("\\", "/")).as_posix().lower()
    if "/belarus/" in p or p.startswith("belarus/"):
        return "Беларусь"
    if "/russia/" in p or p.startswith("russia/"):
        return "Россия"
    if "/usa/" in p or p.startswith("usa/"):
        return "США"
    return "прочее"


def iter_all_cropped_image_files(repo_dataset: Path) -> List[Path]:
    """Все файлы-картинки непосредственно в dataset/**/cropped/ (не в cropped/labels/)."""
    dr = repo_dataset.resolve()
    paths: List[Path] = []
    for cropped_dir in dr.rglob("cropped"):
        if not cropped_dir.is_dir() or cropped_dir.name != "cropped":
            continue
        for ext in IMAGE_EXTENSIONS:
            paths.extend(sorted(cropped_dir.glob(f"*{ext}")))
            u = ext.upper()
            if u != ext:
                paths.extend(sorted(cropped_dir.glob(f"*{u}")))
    seen: set = set()
    uniq: List[Path] = []
    for p in paths:
        if "labels" in p.parts:
            continue
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    return sorted(uniq)


def sample_random_cropped(repo_dataset: Path, k: int, seed: Optional[int]) -> List[Path]:
    pool = iter_all_cropped_image_files(repo_dataset)
    rng = random.Random(seed)
    if k >= len(pool):
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, k)


def write_percent_only_summary(
    rows: List[Dict[str, Any]],
    summary_path: Path,
    split: str,
    csv_path: Path,
    dataset_root: Path,
    *,
    image_source: str = "yolo",
) -> str:
    """Текст сводки: только проценты, без абсолютных количеств."""
    n = len(rows)
    ok = sum(1 for r in rows if r.get("capture_ok"))
    rej = n - ok
    reason_labels = {
        "blur_bad": "размытие (blur ≠ good)",
        "incomplete_textfields": "не все зоны полей (textfields)",
        "low_doc_confidence": "низкая уверенность типа документа (DocConf < 0.9)",
        "pipeline_exception": "ошибка пайплайна",
    }
    lines: List[str] = [f"split: {split}", f"источник изображений: {image_source}", ""]

    lines.append("=== Всего ===")
    lines.append(f"принято (capture_ok): {_pct(ok, n)}")
    lines.append(f"отклонено: {_pct(rej, n)}")
    lines.append("")
    lines.append("причины отказа (доля среди отклонённых; в одном кадре может быть несколько причин):")
    rc_g = Counter()
    for r in rows:
        if not r.get("capture_ok"):
            for part in (r.get("capture_reason_codes") or "").split("|"):
                part = part.strip()
                if part:
                    rc_g[part] += 1
    for code, cnt in rc_g.most_common():
        label = reason_labels.get(code, code)
        lines.append(f"  • {label}: {_pct(cnt, rej)}")
    lines.append("")

    lines.append("=== По странам (по пути под dataset/) ===")
    by_c: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        rel_img = str(r.get("image", ""))
        cropped = str(r.get("cropped_path", ""))
        by_c[
            _dataset_country_from_path_marker(cropped if cropped.strip() else rel_img)
        ].append(r)

    for cname in sorted(by_c.keys(), key=lambda x: (-len(by_c[x]), x)):
        sub = by_c[cname]
        nc = len(sub)
        ok_c = sum(1 for r in sub if r.get("capture_ok"))
        rej_c = nc - ok_c
        lines.append("")
        lines.append(f"— {cname}: доля в выборке {_pct(nc, n)}")
        lines.append(f"  принято: {_pct(ok_c, nc)}")
        lines.append(f"  отклонено: {_pct(rej_c, nc)}")
        lines.append("  причины среди отклонённых в стране:")
        if rej_c == 0:
            lines.append("    —")
        else:
            rc = Counter()
            for r in sub:
                if not r.get("capture_ok"):
                    for part in (r.get("capture_reason_codes") or "").split("|"):
                        part = part.strip()
                        if part:
                            rc[part] += 1
            for code, cnt in rc.most_common():
                label = reason_labels.get(code, code)
                lines.append(f"    • {label}: {_pct(cnt, rej_c)}")

    repo_root = dataset_root.resolve().parent.parent.parent
    try:
        csv_rel = str(csv_path.resolve().relative_to(repo_root))
    except ValueError:
        csv_rel = str(csv_path)
    lines.extend(["", f"csv: {csv_rel}"])
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    return text


def iter_split_images(images_dir: Path) -> List[Path]:
    paths: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(sorted(images_dir.glob(f"*{ext}")))
        u = ext.upper()
        if u != ext:
            paths.extend(sorted(images_dir.glob(f"*{u}")))
    seen = set()
    out: List[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return sorted(out)


def run_one(pipeline: Pipeline, image_path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Returns (row_dict_extensions, error_message_or_None).
    row keys: capture_ok, doc_type, capture_reason_codes, pipeline_error (optional)
    """
    try:
        result = pipeline(image_path, check_quality=True)
        report = sanitize_for_json(getattr(result, "full_report", {}) or {})
        meta = sanitize_for_json(getattr(result, "meta_results", {}) or {})
        capture = evaluate_capture(report, meta)
        doc_type = (report.get("DocType") or "").strip()

        codes = "|".join(r.get("code", "") for r in capture.get("reasons", []))

        row = {
            "capture_ok": bool(capture.get("ok")),
            "doc_type": doc_type or "NONE",
            "capture_reason_codes": codes,
        }
        return row, None
    except Exception as e:  # noqa: BLE001 — сводная таблица по датасету
        err = f"{type(e).__name__}: {e}"
        logger.exception("Ошибка пайплайна для %s", image_path)
        return {
            "capture_ok": False,
            "doc_type": "",
            "capture_reason_codes": "pipeline_exception",
            "pipeline_error": err,
        }, err


def main() -> None:
    parser = argparse.ArgumentParser(description="Датасет model_borders vs логика webapp capture_validation")
    default_root = Path(__file__).resolve().parent.parent / "models" / "model_borders" / "dataset"
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=default_root,
        help="Корень датасета (родитель train/val/test)",
    )
    parser.add_argument(
        "--split",
        choices=("test", "val", "train"),
        default="test",
        help="Какой split прогонять",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV-отчёт (по умолчанию: dataset_root/runs/webapp_capture_{split}.csv)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Остановиться после N изображений (0 = без лимита)")
    parser.add_argument(
        "--model-format",
        default=None,
        help="Формат моделей (по умолчанию env KYC_MODEL_FORMAT или RDOCR_MODEL_FORMAT или ONNX)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Устройство (по умолчанию env KYC_DEVICE или RDOCR_DEVICE или cpu)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--image-source",
        choices=("yolo", "cropped_random"),
        default="yolo",
        help="yolo — split model_borders/dataset; cropped_random — случайные файлы из dataset/**/cropped/",
    )
    parser.add_argument(
        "--random-cropped",
        type=int,
        default=0,
        help="Сколько случайных cropped взять (--image-source cropped_random; 0 = ошибка)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed для воспроизводимой выборки cropped (по умолчанию недетерминировано)",
    )
    parser.add_argument(
        "--repo-dataset",
        type=Path,
        default=None,
        help="Корень dataset/ репозитория (по умолчанию <корень репо>/dataset)",
    )
    args = parser.parse_args()

    if args.image_source == "cropped_random":
        if args.random_cropped <= 0:
            raise SystemExit(
                "Для --image-source cropped_random укажите положительное --random-cropped N"
            )

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    import os

    model_format = args.model_format or os.getenv("KYC_MODEL_FORMAT", os.getenv("RDOCR_MODEL_FORMAT", "ONNX"))
    device = args.device or os.getenv("KYC_DEVICE", os.getenv("RDOCR_DEVICE", "cpu"))

    ds = args.dataset_root.resolve()
    # ds = .../models/model_borders/dataset → корень репо на три уровня выше
    repo_root = ds.parent.parent.parent
    repo_dataset = (args.repo_dataset or (repo_root / "dataset")).resolve()
    if args.image_source == "yolo":
        images_dir = ds / args.split / "images"
        labels_dir = ds / args.split / "labels"
        if not images_dir.is_dir():
            raise SystemExit(f"Нет каталога изображений: {images_dir}")
        images = iter_split_images(images_dir)
        if args.limit and args.limit > 0:
            images = images[: args.limit]
        split_label = args.split
        image_source_summary = "yolo"
    else:
        labels_dir = Path()  # не используется; GT по желанию из cropped/labels ниже
        images = sample_random_cropped(repo_dataset, args.random_cropped, args.seed)
        if not images:
            raise SystemExit(f"Не найдено изображений в {repo_dataset}/**/cropped/")
        seed_part = str(args.seed) if args.seed is not None else "none"
        split_label = f"cropped_random_n{args.random_cropped}_seed{seed_part}"
        image_source_summary = "cropped_random"

    out_csv = args.output
    if out_csv is None:
        runs = ds.parent / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        if args.image_source == "yolo":
            out_csv = runs / f"webapp_capture_{args.split}.csv"
        else:
            seed_part = str(args.seed) if args.seed is not None else "none"
            out_csv = runs / f"webapp_capture_cropped_random_n{args.random_cropped}_s{seed_part}.csv"

    pipeline = Pipeline(model_format=model_format, device=device)

    rows: List[Dict[str, Any]] = []
    for img_path in images:
        cropped_used = ""
        suffix_ok = img_path.suffix.lower() in ALLOWED_SUFFIXES
        infer_path = img_path.resolve()

        if args.image_source == "yolo":
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_acceptable = dataset_has_border_gt(label_path)
            try:
                image_rel = str(img_path.resolve().relative_to(ds))
            except ValueError:
                image_rel = str(img_path)
        else:
            try:
                image_rel = str(infer_path.relative_to(repo_root))
            except ValueError:
                image_rel = str(infer_path)
            cropped_used = image_rel
            label_in_cropped = infer_path.parent / "labels" / f"{infer_path.stem}.txt"
            gt_acceptable = dataset_has_border_gt(label_in_cropped)

        ext_row, _err = run_one(pipeline, infer_path)
        status = "ACCEPT" if ext_row["capture_ok"] else "REJECT"

        mismatch = ""
        if gt_acceptable is True and not ext_row.get("capture_ok"):
            mismatch = "gt_ok_capture_reject"
        elif gt_acceptable is False and ext_row.get("capture_ok"):
            mismatch = "gt_bad_capture_accept"

        row = {
            "image": image_rel,
            "cropped_path": cropped_used,
            "format_ok_webapp": suffix_ok,
            "dataset_border_gt": "" if gt_acceptable is None else ("1" if gt_acceptable else "0"),
            **ext_row,
            "mismatch": mismatch,
        }
        rows.append(row)
        logger.info("%s → %s", infer_path.name, status)

    fieldnames = [
        "image",
        "cropped_path",
        "format_ok_webapp",
        "dataset_border_gt",
        "capture_ok",
        "doc_type",
        "capture_reason_codes",
        "mismatch",
        "pipeline_error",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    summary_path = out_csv.with_suffix(".summary.txt")
    summary_text = write_percent_only_summary(
        rows,
        summary_path,
        split_label,
        out_csv,
        ds,
        image_source=image_source_summary,
    )
    print(summary_text.rstrip())


if __name__ == "__main__":
    main()
