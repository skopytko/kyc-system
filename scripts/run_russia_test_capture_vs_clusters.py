#!/usr/bin/env python3
"""Прогон capture_validation по списку из CSV и сравнительные таблички (%)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from kyc_document.document_processing import Pipeline
from webapp.capture_validation import evaluate_capture

CLUSTER_KEYS = [
    "перспектива_наклон",
    "много_фона",
    "блики_водяной_знак",
    "низкое_разрешение",
    "скан",
    "мятый_заломы_углов",
    "грязный_документ",
    "блюр_текста",
]

LABEL_SHORT = {
    "перспектива_наклон": "перспектива / наклон",
    "много_фона": "много фона",
    "блики_водяной_знак": "блики / водяной знак",
    "низкое_разрешение": "низкое разрешение",
    "скан": "скан",
    "мятый_заломы_углов": "мятый / заломы",
    "грязный_документ": "грязный документ",
    "блюр_текста": "блюр текста",
}


def sanitize_for_json(obj: Any) -> Any:
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


def pct(num: float, den: float) -> str:
    if not den:
        return "0%"
    return f"{round(100.0 * num / den, 1)}%"


def run_rows(pipeline: Pipeline, img_root: Path, csv_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for row in csv_rows:
        fn = row["имя_файла"].strip()
        path = img_root / fn
        codes: List[str] = []
        err: Optional[str] = None
        capture_ok = False
        doc_type = ""
        if not path.is_file():
            err = "file_not_found"
        else:
            try:
                result = pipeline(path, check_quality=True)
                report = sanitize_for_json(getattr(result, "full_report", {}) or {})
                meta = sanitize_for_json(getattr(result, "meta_results", {}) or {})
                capture = evaluate_capture(report, meta)
                capture_ok = bool(capture.get("ok"))
                doc_type = (report.get("DocType") or "").strip() or "NONE"
                codes = [str(r.get("code", "")) for r in capture.get("reasons", [])]
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}"
                capture_ok = False

        reason_set = set(c.strip() for c in codes if c.strip())
        merged.append(
            {
                **row,
                "path_ok": path.is_file(),
                "capture_ok": capture_ok,
                "doc_type": doc_type,
                "reason_blur": int("blur_bad" in reason_set),
                "reason_tf": int("incomplete_textfields" in reason_set),
                "reason_docconf": int("low_doc_confidence" in reason_set),
                "error": err or "",
                "reason_codes": "|".join(sorted(reason_set)),
            }
        )
    return merged


def md_table(headers: List[str], data_rows: List[List[str]]) -> str:
    h = "| " + " | ".join(headers) + " |\n"
    sep = "|" + "|".join(["---"] * len(headers)) + "|\n"
    body = "".join("| " + " | ".join(cells) + " |\n" for cells in data_rows)
    return h + sep + body


def stat_line(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    n = len(rows)
    rej = sum(1 for r in rows if not r["capture_ok"])
    acc = n - rej
    rrows = [r for r in rows if not r["capture_ok"]]
    rn = len(rrows)
    b = sum(r["reason_blur"] for r in rrows)
    tf = sum(r["reason_tf"] for r in rrows)
    dc = sum(r["reason_docconf"] for r in rrows)
    return {
        "Принято": pct(acc, n),
        "Отказано": pct(rej, n),
        "Из отказов: blur_bad": pct(b, rn),
        "Из отказов: textfields": pct(tf, rn),
        "Из отказов: low DocConf": pct(dc, rn),
    }


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    csv_path = repo / "models/model_borders/runs/test_images_russia_clustered_from_description.csv"
    img_root = repo / "models/model_borders/dataset/test/images"
    out_md = repo / "models/model_borders/runs/russia_test_capture_vs_annotation.md"
    out_csv = repo / "models/model_borders/runs/russia_test_capture_vs_annotation_detail.csv"

    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    model_format = os.getenv("KYC_MODEL_FORMAT", os.getenv("RDOCR_MODEL_FORMAT", "ONNX"))
    device = os.getenv("KYC_DEVICE", os.getenv("RDOCR_DEVICE", "cpu"))
    pipeline = Pipeline(model_format=model_format, device=device)
    merged = run_rows(pipeline, img_root, csv_rows)

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        extras = [
            "path_ok",
            "capture_ok",
            "doc_type",
            "reason_blur",
            "reason_tf",
            "reason_docconf",
            "reason_codes",
            "error",
        ]
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) + extras)
        w.writeheader()
        for r in merged:
            flat = {k: r[k] for k in w.fieldnames if k in r}
            w.writerow(flat)

    lines: List[str] = []
    n_all = len(merged)
    lines.append("# Система отказов vs аннотации (Россия, test)\n")
    lines.append(f"- Аннотации: `{csv_path.relative_to(repo)}`\n")
    lines.append(
        f"- Прогон `Pipeline` + `evaluate_capture`: `{out_csv.relative_to(repo)}`\n\n"
    )

    overall = stat_line(merged)
    overall_order = [
        "Принято",
        "Отказано",
        "Из отказов: blur_bad",
        "Из отказов: textfields",
        "Из отказов: low DocConf",
    ]
    lines.append(f"## По всей выборке из CSV ({n_all} кадров)\n\n")
    lines.append(md_table(overall_order, [[overall[k] for k in overall_order]]))

    lines.append("\n## По бинарным признакам из описания\n\n")
    lines.append(
        "Подвыборка **признак = 1** или **= 0**. **Отказ** — доля `capture_ok=False` среди подвыборки. "
        "**Из отказов: …** — доля строк с кодом среди **отклонённых** в подвыборке.\n\n"
    )
    header = [
        "Признак (из описания)",
        "Значение",
        "Принято",
        "Отказ",
        "Из отказов: blur",
        "Из отказов: textfields",
        "Из отказов: DocConf",
    ]
    body: List[List[str]] = []
    for key in CLUSTER_KEYS:
        for val, subset in ("1", 1), ("0", 0):
            sel = []
            for r in merged:
                try:
                    v = int(float(r[key]))
                except (TypeError, ValueError):
                    v = 0
                if v == subset:
                    sel.append(r)
            if not sel:
                continue
            st = stat_line(sel)
            body.append(
                [
                    LABEL_SHORT[key],
                    val,
                    st["Принято"],
                    st["Отказано"],
                    st["Из отказов: blur_bad"],
                    st["Из отказов: textfields"],
                    st["Из отказов: low DocConf"],
                ]
            )

    lines.append(md_table(header, body))
    lines.append("\n")

    errs = [_r for _r in merged if _r.get("error")]
    if errs:
        lines.append("## Ошибки / пропуски\n")
        for e in errs:
            lines.append(f"- `{e['имя_файла']}`: {_esc(e['error'])}\n")

    out_md.write_text("".join(lines), encoding="utf-8")
    print(out_md.resolve())


def _esc(s: str) -> str:
    return s.replace("|", "\\|")


if __name__ == "__main__":
    main()
