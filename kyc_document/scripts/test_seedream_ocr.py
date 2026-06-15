"""
Тест всей системы KYC-OCR на Seedream-сканах против ground truth из CSV.

Имя файла → строка CSV по последнему числу в имени:
    passport_0001.png      → строка №1
    passport_0042_seedream.png → строка №42

Пример:
    python kyc_document/scripts/test_seedream_ocr.py \
        --images_dir seedream_output/russia_passport_swapped \
        --csv dataset/generator_docs/russia/personal_data/personal_data.csv \
        --report-dir reports/seedream_swapped

Считает:
    - CER (Character Error Rate) по каждому полю и в среднем
    - WER (Word Error Rate) — там, где имеет смысл (ФИО, place)
    - Exact match (после нормализации)
    - Длину расхождения, топ-проблемных файлов / полей
    - JSON + Markdown отчёт + CSV с попольной разбивкой
    - Capture validation как в веб (/api/ocr): отказ переснять (blur, DocConf, поля)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kyc_document.document_processing import Pipeline  # noqa: E402
from webapp.capture_validation import MIN_DOC_CONFIDENCE, evaluate_capture  # noqa: E402
from webapp.main import sanitize_for_json  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


CSV_FIELDS = [
    "lastname",
    "firstname",
    "middlename",
    "sex",
    "birth_date",
    "birth_place",
    "date_of_issue",
    "department_code",
    "passport_issued_full",
    "series_and_number",
]

OCR_FIELD_MAP: Dict[str, str] = {
    "lastname": "Last_name_ru",
    "firstname": "First_name_ru",
    "middlename": "Middle_name_ru",
    "sex": "Sex_ru",
    "birth_date": "Birth_date",
    "birth_place": "Birth_place_ru",
    "date_of_issue": "Issue_date",
    "department_code": "Issue_organisation_code",
    "passport_issued_full": "Issue_organization_ru",
    "series_and_number": "Licence_number",
}

NUMERIC_FIELDS = {"birth_date", "date_of_issue", "department_code", "series_and_number"}


def enrich_csv_row(row: dict) -> dict:
    """Добавляет атомарные поля; passport_issued_full — одна строка, не 3 колонки при сравнении."""
    r = dict(row)
    if not (r.get("passport_issued_full") or "").strip():
        parts = [
            r.get("passport_issued"),
            r.get("passport_issued2"),
            r.get("passport_issued3"),
        ]
        r["passport_issued_full"] = " ".join(
            p.strip() for p in parts if p and str(p).strip()
        )
    return r


def load_csv(csv_path: Path) -> List[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return [enrich_csv_row(row) for row in csv.DictReader(f)]


def row_index_for(image: Path) -> Optional[int]:
    m = re.search(r"passport_centerfold_(\d+)", image.stem)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"(\d+)(?!.*\d)", image.stem)
    if not m:
        return None
    return int(m.group(1)) - 1


def find_label_json(image: Path, labels_dir: Path) -> Optional[Path]:
    stem = image.stem
    for name in (f"{stem}.json",):
        p = labels_dir / name
        if p.is_file():
            return p
    m = re.search(r"(passport_centerfold_\d+)", stem)
    if m:
        p = labels_dir / f"{m.group(1)}.json"
        if p.is_file():
            return p
    for pattern in (
        r"(passport_generated2_\d+)",
        r"(passport_generated_\d+)",
        r"(passport2?_\d+)",
    ):
        m = re.match(pattern, stem)
        if m:
            base = m.group(1)
            for cand in (f"{stem}.json", f"{base}.json"):
                p = labels_dir / cand
                if p.is_file():
                    return p
    return None


def row_from_model_fields(mf: dict) -> dict:
    """model_fields (labels JSON) → словарь полей для gt_for_field."""
    org = (mf.get("Issue_organization_ru") or "").strip()
    return {
        "lastname": (mf.get("Last_name_ru") or "").strip(),
        "firstname": (mf.get("First_name_ru") or "").strip(),
        "middlename": (mf.get("Middle_name_ru") or "").strip(),
        "sex": (mf.get("Sex_ru") or "").strip(),
        "birth_date": (mf.get("Birth_date") or "").strip(),
        "birth_place": (mf.get("Birth_place_ru") or "").strip(),
        "date_of_issue": (mf.get("Issue_date") or "").strip(),
        "department_code": (mf.get("Issue_organisation_code") or "").strip(),
        "passport_issued_full": org,
        "series_and_number": (mf.get("Licence_number") or "").strip(),
    }


def gt_row_for_image(
    image: Path, rows: List[dict], labels_dir: Optional[Path]
) -> Tuple[Optional[dict], int, Optional[str]]:
    """Возвращает (row, row_index, error)."""
    if labels_dir and labels_dir.is_dir():
        label_path = find_label_json(image, labels_dir)
        if label_path:
            data = json.loads(label_path.read_text(encoding="utf-8"))
            mf = data.get("model_fields") or data
            if isinstance(mf, dict) and mf:
                idx = row_index_for(image)
                return row_from_model_fields(mf), idx if idx is not None else -1, None
    idx = row_index_for(image)
    if idx is None or idx < 0 or idx >= len(rows):
        return None, idx if idx is not None else -1, "ground truth not found"
    return rows[idx], idx, None


def gt_for_field(row: dict, field_name: str) -> str:
    """GT — значение одноимённого ключа в row (passport_issued_full не собирается на лету)."""
    return (row.get(field_name) or "").strip()


def normalize_generic(text: str) -> str:
    if not text:
        return ""
    text = text.replace("ё", "е").replace("Ё", "Е")
    text = re.sub(r"\s+", " ", text.strip())
    return text.upper()


def normalize_numeric(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\D", "", text)


def normalize_sex(text: str) -> str:
    """Сводит пол к M/F: Ж/ЖЕН. и М/МУЖ. — одно и то же, важен пол, не форма записи."""
    t = normalize_generic(text).rstrip(".")
    if not t:
        return ""
    compact = re.sub(r"[^А-ЯA-Z]", "", t)
    if compact in ("M", "М") or compact.startswith("МУЖ"):
        return "M"
    if compact in ("F", "Ж") or compact.startswith("ЖЕН"):
        return "F"
    if "МУЖ" in compact or (compact and compact[0] == "М"):
        return "M"
    if "ЖЕН" in compact or (compact and compact[0] == "Ж"):
        return "F"
    return compact


def normalize_birth_place(text: str) -> str:
    """Г. КАЛУГА и Г КАЛУГА — одно и то же после нормализации."""
    t = normalize_generic(text)
    t = re.sub(r"\b([А-Я])\.\s*", r"\1 ", t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_for_field(text: str, field_name: str) -> str:
    if field_name == "sex":
        return normalize_sex(text)
    if field_name == "birth_place":
        return normalize_birth_place(text)
    if field_name in NUMERIC_FIELDS:
        return normalize_numeric(text)
    return normalize_generic(text)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def char_error_rate(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 0.0
    if not gt:
        return 1.0
    return levenshtein(gt, pred) / len(gt)


def word_error_rate(gt: str, pred: str) -> float:
    gt_tokens = gt.split()
    pred_tokens = pred.split()
    if not gt_tokens and not pred_tokens:
        return 0.0
    if not gt_tokens:
        return 1.0
    return levenshtein(" ".join(gt_tokens), " ".join(pred_tokens)) / max(1, len(" ".join(gt_tokens)))


@dataclass
class FieldComparison:
    field: str
    gt: str
    pred: str
    gt_norm: str
    pred_norm: str
    cer: float
    wer: float
    exact: bool
    edit_distance: int


@dataclass
class ImageResult:
    image: str
    row_index: int
    error: Optional[str]
    elapsed_sec: float
    fields: List[FieldComparison] = field(default_factory=list)
    capture_ok: bool = True
    capture_reason_codes: str = ""
    capture_reasons: List[Dict[str, str]] = field(default_factory=list)
    doc_type: str = ""

    @property
    def cer_mean(self) -> float:
        return mean(f.cer for f in self.fields) if self.fields else 1.0

    @property
    def exact_ratio(self) -> float:
        if not self.fields:
            return 0.0
        return sum(1 for f in self.fields if f.exact) / len(self.fields)


def run_pipeline(
    pipeline: Pipeline,
    image_path: Path,
    *,
    min_doc_confidence: float,
) -> Tuple[Dict[str, str], Dict[str, Any], str]:
    """OCR + capture_validation (как POST /api/ocr в webapp)."""
    result = pipeline(
        str(image_path),
        ocr=True,
        get_doc_borders=True,
        find_text_fields=True,
        check_quality=True,
        low_quality=True,
        reset_text_fields=True,
    )
    report = sanitize_for_json(getattr(result, "full_report", {}) or {})
    meta = sanitize_for_json(getattr(result, "meta_results", {}) or {})
    capture = evaluate_capture(report, meta, min_doc_confidence=min_doc_confidence)
    doc_type = (report.get("DocType") or "").strip()
    return result.ocr or {}, capture, doc_type


def evaluate_image(
    pipeline: Pipeline,
    image_path: Path,
    rows: List[dict],
    fields: List[str],
    labels_dir: Optional[Path] = None,
    *,
    min_doc_confidence: float = MIN_DOC_CONFIDENCE,
) -> ImageResult:
    row, idx, gt_err = gt_row_for_image(image_path, rows, labels_dir)
    if gt_err or row is None:
        return ImageResult(
            image=image_path.name,
            row_index=idx if idx is not None else -1,
            error=gt_err or "ground truth not found",
            elapsed_sec=0.0,
        )
    start = time.time()
    try:
        ocr_results, capture, doc_type = run_pipeline(
            pipeline, image_path, min_doc_confidence=min_doc_confidence
        )
    except Exception as exc:
        traceback.print_exc()
        return ImageResult(image=image_path.name, row_index=idx, error=str(exc), elapsed_sec=time.time() - start)

    reasons = capture.get("reasons") or []
    reason_codes = "|".join(r.get("code", "") for r in reasons if r.get("code"))
    capture_ok = bool(capture.get("ok"))

    comparisons: List[FieldComparison] = []
    for fname in fields:
        ocr_key = OCR_FIELD_MAP[fname]
        gt_raw = gt_for_field(row, fname)
        pred_raw = (ocr_results.get(ocr_key) or "").strip()
        gt_norm = normalize_for_field(gt_raw, fname)
        pred_norm = normalize_for_field(pred_raw, fname)
        cer = char_error_rate(gt_norm, pred_norm)
        wer = word_error_rate(gt_norm, pred_norm)
        comparisons.append(FieldComparison(
            field=fname,
            gt=gt_raw,
            pred=pred_raw,
            gt_norm=gt_norm,
            pred_norm=pred_norm,
            cer=cer,
            wer=wer,
            exact=(gt_norm == pred_norm),
            edit_distance=levenshtein(gt_norm, pred_norm),
        ))
    return ImageResult(
        image=image_path.name,
        row_index=idx,
        error=None,
        elapsed_sec=time.time() - start,
        fields=comparisons,
        capture_ok=capture_ok,
        capture_reason_codes=reason_codes,
        capture_reasons=reasons,
        doc_type=doc_type,
    )


def _per_field_stats(ok: List[ImageResult], fields: List[str]) -> Dict[str, Dict]:
    per_field: Dict[str, Dict] = {}
    for f in fields:
        per: List[FieldComparison] = []
        for r in ok:
            for c in r.fields:
                if c.field == f:
                    per.append(c)
                    break
        if not per:
            per_field[f] = {"count": 0}
            continue
        cer_list = [c.cer for c in per]
        wer_list = [c.wer for c in per]
        per_field[f] = {
            "count": len(per),
            "exact_match": sum(1 for c in per if c.exact),
            "exact_match_rate": sum(1 for c in per if c.exact) / len(per),
            "cer_mean": mean(cer_list),
            "cer_median": median(cer_list),
            "cer_max": max(cer_list),
            "cer_p90": sorted(cer_list)[int(0.9 * (len(cer_list) - 1))] if len(cer_list) > 1 else cer_list[0],
            "wer_mean": mean(wer_list),
            "edit_distance_total": sum(c.edit_distance for c in per),
            "samples_with_error": sum(1 for c in per if not c.exact),
        }
    return per_field


def _ocr_totals(ok: List[ImageResult]) -> Dict[str, float | int]:
    if not ok:
        return {
            "count": 0,
            "fully_correct_images": 0,
            "fully_correct_rate": 0.0,
            "overall_cer_mean": 1.0,
            "overall_field_accuracy": 0.0,
        }
    fully_correct = sum(1 for r in ok if all(c.exact for c in r.fields))
    return {
        "count": len(ok),
        "fully_correct_images": fully_correct,
        "fully_correct_rate": fully_correct / len(ok),
        "overall_cer_mean": mean(r.cer_mean for r in ok),
        "overall_field_accuracy": mean(r.exact_ratio for r in ok),
    }


def aggregate(
    results: List[ImageResult],
    fields: List[str],
    *,
    min_doc_confidence: float = MIN_DOC_CONFIDENCE,
) -> Dict:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    capture_ok_list = [r for r in ok if r.capture_ok]
    capture_reject_list = [r for r in ok if not r.capture_ok]
    reason_counter: Counter = Counter()
    for r in capture_reject_list:
        for code in (r.capture_reason_codes or "").split("|"):
            if code:
                reason_counter[code] += 1

    all_ocr = _ocr_totals(ok)
    accepted_ocr = _ocr_totals(capture_ok_list)

    return {
        "totals": {
            "images": len(results),
            "processed": len(ok),
            "errors": len(failed),
            "fully_correct_images": all_ocr["fully_correct_images"],
            "fully_correct_rate": all_ocr["fully_correct_rate"],
            "overall_cer_mean": all_ocr["overall_cer_mean"],
            "overall_field_accuracy": all_ocr["overall_field_accuracy"],
            "avg_inference_sec": mean(r.elapsed_sec for r in ok) if ok else 0.0,
            "capture_accepted": len(capture_ok_list),
            "capture_rejected": len(capture_reject_list),
            "capture_accept_rate": len(capture_ok_list) / len(ok) if ok else 0.0,
            "capture_reject_rate": len(capture_reject_list) / len(ok) if ok else 0.0,
            "capture_reason_counts": dict(reason_counter),
            "min_doc_confidence": min_doc_confidence,
            "accepted_count": accepted_ocr["count"],
            "fully_correct_images_accepted": accepted_ocr["fully_correct_images"],
            "fully_correct_rate_accepted": accepted_ocr["fully_correct_rate"],
            "overall_cer_mean_accepted": accepted_ocr["overall_cer_mean"],
            "overall_field_accuracy_accepted": accepted_ocr["overall_field_accuracy"],
        },
        "per_field": _per_field_stats(ok, fields),
        "per_field_accepted": _per_field_stats(capture_ok_list, fields),
    }


def render_markdown(report: Dict, results: List[ImageResult], fields: List[str],
                    top_n: int = 20) -> str:
    totals = report["totals"]
    lines: List[str] = []
    lines.append("# KYC OCR test report — Seedream\n")
    lines.append(f"- Изображений: **{totals['images']}**")
    lines.append(f"- Успешно обработано: **{totals['processed']}**")
    lines.append(f"- Ошибки pipeline: **{totals['errors']}**")
    lines.append(f"- Полностью совпавших (все поля): **{totals['fully_correct_images']}** "
                 f"({totals['fully_correct_rate']:.1%})")
    lines.append(f"- Средний **CER** (все обработанные): **{totals['overall_cer_mean']:.3f}**")
    lines.append(f"- Средняя точность поля (все): **{totals['overall_field_accuracy']:.1%}**")
    n_acc = totals.get("accepted_count", 0)
    lines.append(
        f"- Средний **CER** среди **принятых** capture (N={n_acc}): "
        f"**{totals.get('overall_cer_mean_accepted', 0):.3f}**"
    )
    lines.append(
        f"- Точность поля среди принятых: "
        f"**{totals.get('overall_field_accuracy_accepted', 0):.1%}**"
    )
    lines.append(
        f"- Полностью совпавших среди принятых: "
        f"**{totals.get('fully_correct_images_accepted', 0)}** "
        f"({totals.get('fully_correct_rate_accepted', 0):.1%})"
    )
    lines.append(f"- Среднее время на картинку: **{totals['avg_inference_sec']:.2f} сек**\n")

    lines.append("## Capture validation (как в веб)\n")
    lines.append(
        f"- Принято (снимок подходит): **{totals.get('capture_accepted', 0)}** "
        f"({totals.get('capture_accept_rate', 0):.1%})"
    )
    lines.append(
        f"- Отклонено (нужно переснять): **{totals.get('capture_rejected', 0)}** "
        f"({totals.get('capture_reject_rate', 0):.1%})"
    )
    lines.append(
        f"- Порог DocConf: **{totals.get('min_doc_confidence', MIN_DOC_CONFIDENCE)}**"
    )
    reason_counts = totals.get("capture_reason_counts") or {}
    if reason_counts:
        lines.append("\nПричины отказа (число снимков; на одном может быть несколько):\n")
        lines.append("| Код | Снимков |")
        lines.append("| --- | --: |")
        for code, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| `{code}` | {cnt} |")
    lines.append("")

    rejected = [r for r in results if r.error is None and not r.capture_ok]
    if rejected:
        lines.append("### Отклонённые снимки\n")
        lines.append("| image | doc_type | причины | CER mean |")
        lines.append("| --- | --- | --- | --: |")
        for r in sorted(rejected, key=lambda x: x.cer_mean, reverse=True)[:30]:
            titles = "; ".join(
                (x.get("title") or x.get("code") or "") for x in (r.capture_reasons or [])
            )
            lines.append(
                f"| {r.image} | {r.doc_type or '—'} | {titles or r.capture_reason_codes} | "
                f"{r.cer_mean:.3f} |"
            )
        if len(rejected) > 30:
            lines.append(f"\n… и ещё {len(rejected) - 30} отклонённых.\n")
    lines.append("")

    def _append_per_field_table(title: str, per_field_key: str) -> None:
        lines.append(f"## {title}\n")
        header = (
            "| Поле | N | Exact | Acc | CER mean | CER p90 | CER max | WER mean | Edit dist total |"
        )
        lines.append(header)
        lines.append("| --- | --: | --: | --: | --: | --: | --: | --: | --: |")
        for f in fields:
            s = report.get(per_field_key, {}).get(f, {})
            if not s.get("count"):
                lines.append(f"| {f} | 0 | — | — | — | — | — | — | — |")
                continue
            lines.append(
                f"| {f} | {s['count']} | {s['exact_match']} | {s['exact_match_rate']:.1%} | "
                f"{s['cer_mean']:.3f} | {s['cer_p90']:.3f} | {s['cer_max']:.3f} | "
                f"{s['wer_mean']:.3f} | {s['edit_distance_total']} |"
            )
        lines.append("")

    _append_per_field_table("По полям (все обработанные)", "per_field")
    _append_per_field_table(
        f"По полям (только принятые capture, N={n_acc})", "per_field_accepted"
    )

    ok_results = [r for r in results if r.error is None]
    accepted_results = [r for r in ok_results if r.capture_ok]
    worst = sorted(ok_results, key=lambda r: r.cer_mean, reverse=True)[:top_n]
    lines.append(f"\n## Топ-{top_n} худших картинок (по среднему CER)\n")
    lines.append("| # | image | row | capture | CER mean | Exact fields |")
    lines.append("| --- | --- | --: | --- | --: | --: |")
    for i, r in enumerate(worst, 1):
        exact_fields = sum(1 for c in r.fields if c.exact)
        cap = "ok" if r.capture_ok else "reject"
        lines.append(
            f"| {i} | {r.image} | {r.row_index + 1} | {cap} | {r.cer_mean:.3f} | "
            f"{exact_fields}/{len(r.fields)} |"
        )

    def _append_cer_buckets(title: str, subset: List[ImageResult]) -> None:
        lines.append(f"\n## {title}\n")
        buckets = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, float("inf")]
        bucket_labels = ["0", "≤0.05", "≤0.10", "≤0.20", "≤0.30", "≤0.50", "≤1.0", ">1.0"]
        counts = [0] * len(buckets)
        for r in subset:
            c = r.cer_mean
            for i, hi in enumerate(buckets):
                if c <= hi:
                    counts[i] += 1
                    break
        lines.append("| Bucket | Count |")
        lines.append("| --- | --: |")
        for label, c in zip(bucket_labels, counts):
            lines.append(f"| {label} | {c} |")

    _append_cer_buckets("Распределение CER (все обработанные)", ok_results)
    _append_cer_buckets(
        f"Распределение CER (принятые capture, N={len(accepted_results)})",
        accepted_results,
    )

    worst_acc = sorted(accepted_results, key=lambda r: r.cer_mean, reverse=True)[:top_n]
    if worst_acc:
        lines.append(f"\n## Топ-{top_n} худших среди принятых (CER)\n")
        lines.append("| # | image | row | CER mean | Exact fields |")
        lines.append("| --- | --- | --: | --: | --: |")
        for i, r in enumerate(worst_acc, 1):
            exact_fields = sum(1 for c in r.fields if c.exact)
            lines.append(
                f"| {i} | {r.image} | {r.row_index + 1} | {r.cer_mean:.3f} | "
                f"{exact_fields}/{len(r.fields)} |"
            )

    failed = [r for r in results if r.error]
    if failed:
        lines.append("\n## Ошибки pipeline\n")
        for r in failed[:30]:
            lines.append(f"- `{r.image}` (row={r.row_index + 1}): {r.error}")
        if len(failed) > 30:
            lines.append(f"- … и ещё {len(failed) - 30}")

    lines.append("\n## Примеры расхождений (топ-10 по полю)\n")
    for f in fields:
        per = []
        for r in ok_results:
            for c in r.fields:
                if c.field == f and not c.exact:
                    per.append((r.image, c))
                    break
        per.sort(key=lambda x: x[1].cer, reverse=True)
        if not per:
            continue
        lines.append(f"### {f}")
        lines.append("| image | gt | pred | CER |")
        lines.append("| --- | --- | --- | --: |")
        for img_name, c in per[:10]:
            lines.append(f"| {img_name} | `{c.gt}` | `{c.pred}` | {c.cer:.3f} |")
        lines.append("")
    return "\n".join(lines)


def save_per_field_csv(results: List[ImageResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "row", "capture_ok", "capture_reason_codes", "doc_type",
            "field", "gt", "pred", "gt_norm", "pred_norm",
            "edit_distance", "cer", "wer", "exact", "error",
        ])
        for r in results:
            if r.error:
                w.writerow([
                    r.image, r.row_index + 1, "", "", "", "", "", "", "", "", "",
                    "", "", "", r.error,
                ])
                continue
            cap_ok = int(r.capture_ok)
            for c in r.fields:
                w.writerow([
                    r.image, r.row_index + 1, cap_ok, r.capture_reason_codes, r.doc_type,
                    c.field, c.gt, c.pred, c.gt_norm, c.pred_norm, c.edit_distance,
                    f"{c.cer:.6f}", f"{c.wer:.6f}", int(c.exact), "",
                ])


def save_capture_csv(results: List[ImageResult], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image", "row", "capture_ok", "capture_reason_codes", "doc_type",
            "cer_mean", "exact_ratio", "error",
        ])
        for r in results:
            w.writerow([
                r.image,
                r.row_index + 1,
                int(r.capture_ok) if r.error is None else "",
                r.capture_reason_codes,
                r.doc_type,
                f"{r.cer_mean:.6f}" if r.error is None else "",
                f"{r.exact_ratio:.6f}" if r.error is None else "",
                r.error or "",
            ])


def results_from_report_json(data: dict) -> List[ImageResult]:
    """Восстановить ImageResult из сохранённого report.json (без повторного pipeline)."""
    out: List[ImageResult] = []
    for r in data.get("results", []):
        fields = [
            FieldComparison(
                field=c["field"],
                gt=c["gt"],
                pred=c["pred"],
                gt_norm=c["gt_norm"],
                pred_norm=c["pred_norm"],
                cer=float(c["cer"]),
                wer=float(c["wer"]),
                exact=bool(c["exact"]),
                edit_distance=int(c["edit_distance"]),
            )
            for c in r.get("fields", [])
        ]
        out.append(
            ImageResult(
                image=r["image"],
                row_index=int(r.get("row", 1)) - 1,
                error=r.get("error"),
                elapsed_sec=float(r.get("elapsed_sec", 0)),
                fields=fields,
                capture_ok=bool(r.get("capture_ok", True)),
                capture_reason_codes=r.get("capture_reason_codes", "") or "",
                capture_reasons=r.get("capture_reasons") or [],
                doc_type=r.get("doc_type", "") or "",
            )
        )
    return out


def recompute_gt_in_report_json(
    json_path: Path,
    report_dir: Path,
    csv_path: Path,
    images_dir: Path,
    labels_dir: Optional[Path],
    top_n: int,
) -> None:
    """Пересчитать GT/exact/CER из labels/CSV, pred OCR оставить из report.json."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    fields = list(data.get("config", {}).get("fields") or CSV_FIELDS)
    cfg_labels = data.get("config", {}).get("labels_dir")
    if labels_dir is None and cfg_labels:
        labels_dir = Path(cfg_labels)
    if labels_dir is not None and not labels_dir.is_dir():
        labels_dir = None

    rows = load_csv(csv_path)
    updated: List[ImageResult] = []

    for entry in data.get("results", []):
        img_name = entry.get("image", "")
        img_path = images_dir / img_name
        pred_by_field = {c["field"]: c for c in entry.get("fields", [])}
        row, idx, gt_err = gt_row_for_image(img_path, rows, labels_dir)

        if gt_err or row is None:
            updated.append(
                ImageResult(
                    image=img_name,
                    row_index=int(entry.get("row", 1)) - 1,
                    error=gt_err or entry.get("error") or "ground truth not found",
                    elapsed_sec=float(entry.get("elapsed_sec", 0)),
                    capture_ok=bool(entry.get("capture_ok", True)),
                    capture_reason_codes=entry.get("capture_reason_codes", "") or "",
                    capture_reasons=entry.get("capture_reasons") or [],
                    doc_type=entry.get("doc_type", "") or "",
                )
            )
            continue

        comparisons: List[FieldComparison] = []
        for fname in fields:
            old = pred_by_field.get(fname, {})
            pred_raw = (old.get("pred") or "").strip()
            gt_raw = gt_for_field(row, fname)
            gt_norm = normalize_for_field(gt_raw, fname)
            pred_norm = normalize_for_field(pred_raw, fname)
            cer = char_error_rate(gt_norm, pred_norm)
            comparisons.append(
                FieldComparison(
                    field=fname,
                    gt=gt_raw,
                    pred=pred_raw,
                    gt_norm=gt_norm,
                    pred_norm=pred_norm,
                    cer=cer,
                    wer=word_error_rate(gt_norm, pred_norm),
                    exact=(gt_norm == pred_norm),
                    edit_distance=levenshtein(gt_norm, pred_norm),
                )
            )
        updated.append(
            ImageResult(
                image=img_name,
                row_index=idx,
                error=entry.get("error"),
                elapsed_sec=float(entry.get("elapsed_sec", 0)),
                fields=comparisons,
                capture_ok=bool(entry.get("capture_ok", True)),
                capture_reason_codes=entry.get("capture_reason_codes", "") or "",
                capture_reasons=entry.get("capture_reasons") or [],
                doc_type=entry.get("doc_type", "") or "",
            )
        )

    min_doc = float(
        data.get("config", {}).get("min_doc_confidence") or MIN_DOC_CONFIDENCE
    )
    report = aggregate(updated, fields, min_doc_confidence=min_doc)
    data["summary"] = report
    data["results"] = [
        {
            "image": r.image,
            "row": r.row_index + 1,
            "error": r.error,
            "elapsed_sec": r.elapsed_sec,
            "cer_mean": r.cer_mean,
            "exact_ratio": r.exact_ratio,
            "capture_ok": r.capture_ok,
            "capture_reason_codes": r.capture_reason_codes,
            "capture_reasons": r.capture_reasons,
            "doc_type": r.doc_type,
            "fields": [asdict(c) for c in r.fields],
        }
        for r in updated
    ]
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = report_dir / "report.md"
    save_per_field_csv(updated, report_dir / "per_field.csv")
    save_capture_csv(updated, report_dir / "capture.csv")
    md_path.write_text(
        render_markdown(report, updated, fields, top_n=top_n), encoding="utf-8"
    )
    t = report["totals"]
    print(f"GT пересчитан в {json_path}")
    print(
        f"fully_correct: {t['fully_correct_images']}/{t['images']} "
        f"({t['fully_correct_rate']:.1%})"
    )
    print(f"CER mean (all): {t['overall_cer_mean']:.3f}")


def regenerate_reports_from_json(json_path: Path, report_dir: Path, top_n: int) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    fields = list(data.get("config", {}).get("fields") or CSV_FIELDS)
    min_doc = float(
        data.get("config", {}).get("min_doc_confidence") or MIN_DOC_CONFIDENCE
    )
    results = results_from_report_json(data)
    report = aggregate(results, fields, min_doc_confidence=min_doc)
    report_dir.mkdir(parents=True, exist_ok=True)
    data["summary"] = report
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = report_dir / "report.md"
    csv_path = report_dir / "per_field.csv"
    capture_csv_path = report_dir / "capture.csv"
    md_path.write_text(render_markdown(report, results, fields, top_n=top_n), encoding="utf-8")
    save_per_field_csv(results, csv_path)
    save_capture_csv(results, capture_csv_path)
    t = report["totals"]
    print(f"Пересобран отчёт из {json_path}")
    print(f"CER mean (all): {t['overall_cer_mean']:.3f}")
    print(
        f"CER mean (accepted, N={t.get('accepted_count', 0)}): "
        f"{t.get('overall_cer_mean_accepted', 0):.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Тест системы KYC-OCR на Seedream-датасете")
    parser.add_argument("--images_dir", type=Path,
                        default=PROJECT_ROOT / "seedream_output" / "russia_passport_swapped",
                        help="Папка с Seedream-сканами")
    parser.add_argument("--csv", type=Path,
                        default=PROJECT_ROOT / "dataset" / "generator_docs" / "russia"
                                / "personal_data" / "personal_data.csv",
                        help="CSV с ground-truth значениями")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="JSON labels с model_fields (приоритет над CSV, для centerfold)",
    )
    parser.add_argument("--report-dir", type=Path,
                        default=PROJECT_ROOT / "reports" / "seedream_ocr",
                        help="Куда писать отчёты")
    parser.add_argument("--format", default="ONNX", choices=["ONNX", "OpenVINO", "TFlite"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    parser.add_argument("--limit", type=int, default=0, help="Тест на N первых файлах (0 = все)")
    parser.add_argument("--top", type=int, default=20, help="Сколько худших картинок в отчёт")
    parser.add_argument("--fields", nargs="+", choices=CSV_FIELDS, default=CSV_FIELDS,
                        help="Какие поля проверять")
    parser.add_argument(
        "--min-doc-confidence",
        type=float,
        default=MIN_DOC_CONFIDENCE,
        help=f"Порог DocConf для отказа съёмки (как веб, по умолчанию {MIN_DOC_CONFIDENCE})",
    )
    parser.add_argument(
        "--regenerate-from-json",
        type=Path,
        default=None,
        help="Пересобрать report.md из report.json (без повторного OCR)",
    )
    parser.add_argument(
        "--recompute-gt",
        type=Path,
        default=None,
        metavar="REPORT.JSON",
        help="Пересчитать GT/exact/CER по labels+CSV, pred из report.json (без OCR)",
    )
    args = parser.parse_args()

    if args.recompute_gt:
        if not args.recompute_gt.is_file():
            raise SystemExit(f"Не найден JSON: {args.recompute_gt}")
        recompute_gt_in_report_json(
            args.recompute_gt,
            args.report_dir,
            args.csv,
            args.images_dir,
            args.labels_dir,
            args.top,
        )
        return

    if args.regenerate_from_json:
        if not args.regenerate_from_json.is_file():
            raise SystemExit(f"Не найден JSON: {args.regenerate_from_json}")
        regenerate_reports_from_json(
            args.regenerate_from_json, args.report_dir, args.top
        )
        return

    if not args.images_dir.exists():
        raise SystemExit(f"Не найдена папка: {args.images_dir}")
    if not args.csv.exists():
        raise SystemExit(f"Не найден CSV: {args.csv}")

    rows = load_csv(args.csv)
    print(f"CSV: {args.csv} ({len(rows)} строк)")
    if args.labels_dir:
        if not args.labels_dir.exists():
            raise SystemExit(f"Не найдена папка labels: {args.labels_dir}")
        print(f"Labels: {args.labels_dir} ({len(list(args.labels_dir.glob('*.json')))} json)")

    images = sorted(p for p in args.images_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"Нет изображений в {args.images_dir}")
    print(f"Найдено {len(images)} изображений в {args.images_dir}")

    print(f"Инициализация Pipeline ({args.format}/{args.device})...")
    print(f"Capture validation: min_doc_confidence={args.min_doc_confidence}")
    pipeline = Pipeline(model_format=args.format, device=args.device, verbose=False)

    results: List[ImageResult] = []
    fields = list(args.fields)
    for i, image_path in enumerate(images, 1):
        print(f"[{i:>4}/{len(images)}] {image_path.name}", end=" ", flush=True)
        res = evaluate_image(
            pipeline,
            image_path,
            rows,
            fields,
            args.labels_dir,
            min_doc_confidence=args.min_doc_confidence,
        )
        results.append(res)
        if res.error:
            print(f"ERROR {res.error[:80]}")
        else:
            exact = sum(1 for c in res.fields if c.exact)
            cap = "OK" if res.capture_ok else "REJECT"
            print(
                f"{cap}  CER {res.cer_mean:.3f}  exact {exact}/{len(res.fields)}"
                + (f"  [{res.capture_reason_codes}]" if not res.capture_ok else "")
            )

    report = aggregate(results, fields, min_doc_confidence=args.min_doc_confidence)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / "report.json"
    md_path = args.report_dir / "report.md"
    csv_path = args.report_dir / "per_field.csv"
    capture_csv_path = args.report_dir / "capture.csv"

    full_json = {
        "config": {
            "images_dir": str(args.images_dir),
            "csv": str(args.csv),
            "labels_dir": str(args.labels_dir) if args.labels_dir else None,
            "format": args.format,
            "device": args.device,
            "limit": args.limit,
            "fields": fields,
            "min_doc_confidence": args.min_doc_confidence,
        },
        "summary": report,
        "results": [
            {
                "image": r.image,
                "row": r.row_index + 1,
                "error": r.error,
                "elapsed_sec": r.elapsed_sec,
                "cer_mean": r.cer_mean,
                "exact_ratio": r.exact_ratio,
                "capture_ok": r.capture_ok,
                "capture_reason_codes": r.capture_reason_codes,
                "capture_reasons": r.capture_reasons,
                "doc_type": r.doc_type,
                "fields": [asdict(c) for c in r.fields],
            }
            for r in results
        ],
    }
    json_path.write_text(json.dumps(full_json, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report, results, fields, top_n=args.top), encoding="utf-8")
    save_per_field_csv(results, csv_path)
    save_capture_csv(results, capture_csv_path)

    print()
    print("=" * 70)
    print(f"Готово.")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"  CSV: {csv_path}")
    print(f"  Capture CSV: {capture_csv_path}")
    print()
    t = report["totals"]
    print(f"Capture accepted: {t['capture_accepted']}/{t['processed']} ({t['capture_accept_rate']:.1%})")
    print(f"Capture rejected: {t['capture_rejected']}/{t['processed']} ({t['capture_reject_rate']:.1%})")
    if t.get("capture_reason_counts"):
        print(f"Reject reasons: {t['capture_reason_counts']}")
    print(f"CER mean (all): {t['overall_cer_mean']:.3f}")
    print(
        f"CER mean (accepted capture, N={t.get('accepted_count', 0)}): "
        f"{t.get('overall_cer_mean_accepted', 0):.3f}"
    )
    print(f"Field accuracy (all): {t['overall_field_accuracy']:.1%}")
    print(
        f"Field accuracy (accepted): {t.get('overall_field_accuracy_accepted', 0):.1%}"
    )
    print(f"Fully correct images: {t['fully_correct_images']}/{t['processed']} "
          f"({t['fully_correct_rate']:.1%})")


if __name__ == "__main__":
    main()
