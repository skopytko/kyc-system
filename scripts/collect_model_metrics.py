#!/usr/bin/env python3
"""
Сбор метрик по обученным моделям в `models/**`.

Идея: метрики уже сохраняются в артефактах обучения (например, YOLO `results.csv`),
а для "ручных" тестов можно опционально прогонять `test.py` и парсить stdout.

Примеры:
  python scripts/collect_model_metrics.py
  python scripts/collect_model_metrics.py --format markdown
  python scripts/collect_model_metrics.py --format json --out metrics.json
  python scripts/collect_model_metrics.py --run-tests
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


@dataclass
class MetricRow:
    model: str
    country: Optional[str]
    run: str
    kind: str  # yolo/train, yolo/val, classifier/test, ...
    metrics: Dict[str, Any]


def _read_last_row_csv(path: Path) -> Optional[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            last = None
            for row in reader:
                last = row
            return last
    except Exception:
        return None


def _parse_float(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except Exception:
        return v


def _detect_country(p: Path) -> Optional[str]:
    parts = {x.lower() for x in p.parts}
    for c in ("russia", "belarus", "usa"):
        if c in parts:
            return c
    return None


def _detect_model_name(p: Path) -> str:
    # ожидаем `.../models/<model_name>/...`
    try:
        idx = p.parts.index("models")
        return p.parts[idx + 1]
    except Exception:
        return p.parent.name


def _collect_yolo_results() -> List[MetricRow]:
    out: List[MetricRow] = []
    for csv_path in MODELS_DIR.rglob("results.csv"):
        # MLflow может дублировать файлы в `models/mlruns/**/artifacts` — их пропускаем,
        # чтобы не захламлять отчёт.
        parts_lower = [p.lower() for p in csv_path.parts]
        if "mlruns" in parts_lower or "mlflow" in parts_lower:
            continue
        # фильтруем только YOLO-подобные результаты (есть колонки metrics/* или segment/bbox)
        last = _read_last_row_csv(csv_path)
        if not last:
            continue
        cols = set(last.keys())
        is_yolo = any(k.startswith("metrics/") for k in cols) or any(
            k in cols for k in ("box_loss", "cls_loss", "dfl_loss", "seg_loss", "metrics/mAP50(B)")
        )
        if not is_yolo:
            continue

        model = _detect_model_name(csv_path)
        if model.lower() == "mlruns":
            continue
        country = _detect_country(csv_path)
        run = str(csv_path.parent.relative_to(MODELS_DIR))

        metrics: Dict[str, Any] = {}
        # Вытащим наиболее полезные ключи, если они есть
        preferred = [
            "epoch",
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "metrics/precision(M)",
            "metrics/recall(M)",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)",
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP50",
            "metrics/mAP50-95",
        ]
        for k in preferred:
            if k in last:
                metrics[k] = _parse_float(last.get(k))

        # добавим всё остальное полезное (с ограничением)
        if not metrics:
            for k, v in last.items():
                if k and (k.startswith("metrics/") or k.endswith("_loss")):
                    metrics[k] = _parse_float(v)

        out.append(
            MetricRow(
                model=model,
                country=country,
                run=run,
                kind="yolo/results.csv",
                metrics=metrics,
            )
        )
    return out


_ACC_RE = re.compile(r"Accuracy:\s*([0-9]*\.?[0-9]+)")
_PREC_MACRO_RE = re.compile(r"Precision \\(macro\\):\\s*([0-9]*\\.?[0-9]+)")
_REC_MACRO_RE = re.compile(r"Recall \\(macro\\):\\s*([0-9]*\\.?[0-9]+)")
_F1_MACRO_RE = re.compile(r"F1-score \\(macro\\):\\s*([0-9]*\\.?[0-9]+)")


def _run_test_py(folder: Path) -> Optional[Tuple[float, str]]:
    test_py = folder / "test.py"
    if not test_py.exists():
        return None
    try:
        p = subprocess.run(
            [sys.executable, str(test_py)],
            cwd=str(folder),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        s = p.stdout or ""
        m = _ACC_RE.search(s)
        if not m:
            return None
        return float(m.group(1)), s
    except Exception:
        return None


def _collect_classifier_tests(run_tests: bool) -> List[MetricRow]:
    out: List[MetricRow] = []
    if not run_tests:
        return out

    # Запускаем `test.py` для всех `models/model_*/.../test.py` (включая корневые модели),
    # а также для country-подпапок, если они есть.
    for test_py in sorted(MODELS_DIR.rglob("test.py")):
        # не трогаем mlruns/mlflow
        parts_lower = [p.lower() for p in test_py.parts]
        if "mlruns" in parts_lower or "mlflow" in parts_lower:
            continue

        folder = test_py.parent
        model = _detect_model_name(test_py)
        country = _detect_country(test_py)

        res = _run_test_py(folder)
        if not res:
            continue
        acc, stdout = res
        run = str(folder.relative_to(MODELS_DIR))

        metrics: Dict[str, Any] = {"accuracy": acc}
        for rgx, key in (
            (_PREC_MACRO_RE, "precision_macro"),
            (_REC_MACRO_RE, "recall_macro"),
            (_F1_MACRO_RE, "f1_macro"),
        ):
            mm = rgx.search(stdout)
            if mm:
                metrics[key] = float(mm.group(1))

        out.append(MetricRow(model=model, country=country, run=run, kind="test.py", metrics=metrics))
    return out


def _to_markdown(rows: List[MetricRow]) -> str:
    # компактная таблица "по ключам"
    all_keys: List[str] = []
    seen = set()
    for r in rows:
        for k in r.metrics.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)

    header = ["model", "country", "kind", "run"] + all_keys
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in sorted(rows, key=lambda x: (x.model, x.country or "", x.kind, x.run)):
        base = [r.model, r.country or "", r.kind, r.run]
        vals = []
        for k in all_keys:
            v = r.metrics.get(k)
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append("" if v is None else str(v))
        lines.append("| " + " | ".join(base + vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["markdown", "json"], default="markdown")
    ap.add_argument("--out", default="", help="Файл для сохранения (если пусто — печать в stdout)")
    ap.add_argument("--run-tests", action="store_true", help="Запускать test.py (может занять время)")
    args = ap.parse_args()

    rows: List[MetricRow] = []
    rows.extend(_collect_yolo_results())
    rows.extend(_collect_classifier_tests(run_tests=bool(args.run_tests)))

    if args.format == "json":
        payload = [asdict(r) for r in rows]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        text = _to_markdown(rows)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(str(out_path))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

