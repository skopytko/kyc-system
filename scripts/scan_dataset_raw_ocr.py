#!/usr/bin/env python3
"""Сканирует dataset/**/raw/** (без labels).

Пишет:
  - dataset/raw_ocr_scan.tsv — has_ocr, capture_ok (как в webapp evaluate_capture), doctype, ...
  - dataset/raw_files_with_ocr.txt — только пути, где есть OCR и все проверки capture прошли
    (как POST /api/ocr: sanitize_for_json + evaluate_capture).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# корень репозитория
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("KYC_MODEL_FORMAT", "ONNX")
os.environ.setdefault("KYC_DEVICE", "cpu")


def iter_raw_images(dataset: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(dataset.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        parts = p.parts
        if "raw" not in parts or "labels" in parts:
            continue
        yield p


def main():
    import time

    from kyc_document.document_processing.pipeline.pipeline import Pipeline
    from webapp.capture_validation import evaluate_capture
    from webapp.main import sanitize_for_json

    dataset = ROOT / "dataset"
    out = ROOT / "dataset" / "raw_ocr_scan.tsv"
    list_path = ROOT / "dataset" / "raw_files_with_ocr.txt"
    paths = list(iter_raw_images(dataset))
    pipe = Pipeline(model_format=os.environ.get("KYC_MODEL_FORMAT", "ONNX"), device=os.environ.get("KYC_DEVICE", "cpu"), verbose=False)

    t0 = time.perf_counter()
    strict_paths: list[str] = []
    with out.open("w", encoding="utf-8") as f:
        f.write(
            "path\thas_ocr\tcapture_ok\tcapture_reason_codes\tdoctype\tpassport_page_type\tnote\n"
        )
        for i, img in enumerate(paths):
            rel = img.relative_to(ROOT)
            rel_s = rel.as_posix()
            try:
                r = pipe(str(img), low_quality=True, ocr=True)
            except Exception as e:
                f.write(f"{rel_s}\t0\t0\tERROR\t\t\tERROR:{e!r}\n")
                f.flush()
                continue
            ocr = r.meta_results.get("OCR")
            has = 1 if ocr else 0
            report = sanitize_for_json(r.full_report)
            meta = sanitize_for_json(r.meta_results)
            cap = evaluate_capture(report, meta)
            cap_ok = 1 if cap.get("ok") else 0
            codes = ";".join(str(x.get("code", "")) for x in cap.get("reasons", []))
            dt = r.doctype or ""
            ppt = r.meta_results.get("PassportPageType", {}).get("page_type", "")
            note = ""
            if dt == "NONE":
                note = "doctype_none"
            elif dt and dt.startswith("russia_") and ppt == "passport_pages":
                note = "russia_seal_branch"
            elif dt and dt.startswith("russia_") and ppt != "passport_centerfold":
                note = f"russia_no_centerfold:{ppt or 'empty'}"
            elif not has and dt:
                note = "no_text_fields_or_split_or_ocr_empty"
            f.write(f"{rel_s}\t{has}\t{cap_ok}\t{codes}\t{dt}\t{ppt}\t{note}\n")
            if has and cap_ok:
                strict_paths.append(rel_s)
            if (i + 1) % 50 == 0:
                f.flush()
                print(f"{i+1}/{len(paths)} {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    list_path.write_text("\n".join(strict_paths) + ("\n" if strict_paths else ""), encoding="utf-8")
    print(
        f"Wrote {out} ({len(paths)} rows), {list_path} ({len(strict_paths)} paths, OCR+capture_ok) "
        f"in {time.perf_counter()-t0:.1f}s",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
