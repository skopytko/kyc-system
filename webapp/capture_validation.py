"""Проверка качества съёмки для ответа API: когда показывать запрос переснять документ."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MIN_DOC_CONFIDENCE = 0.9

# TextFieldsDetectorBelarus — полный набор классов (после 1 bbox на класс).
BELARUS_TEXTFIELD_LABELS_EXPECTED: Set[str] = frozenset(
    [
        "authority",
        "authority2",
        "code_of_issuing",
        "date_of_birth",
        "date_of_expiry",
        "date_of_issue",
        "identification_no",
        "names",
        "nationality",
        "passport_no",
        "photo",
        "place_of_birth",
        "sex",
        "signature",
        "signature2",
        "surname",
        "type",
    ]
)

# TextFieldsDetectorUSA — field_0..field_19.
USA_TEXTFIELD_LABELS_EXPECTED: Set[str] = frozenset(f"field_{i}" for i in range(20))

# Не требуем латинские дубликаты и «место жительства»: модель часто их не бьёт в bbox при
# нормальном OCR русской стороны — иначе ложное «снимок не подходит» (см. RUSSIA_FIELDS в index.html).
RUSSIA_CENTERFOLD_TEXTFIELD_LABELS_REQUIRED: Set[str] = frozenset(
    {
        "Last_name_ru",
        "First_name_ru",
        "Middle_name_ru",
        "Sex_ru",
        "Birth_date",
        "Birth_place_ru",
        "Issue_organization_ru",
        "Issue_date",
        "Issue_organisation_code",
        "Licence_number",
    }
)


def _extract_text_field_labels(meta_results: Optional[dict]) -> Tuple[Optional[str], List[str]]:
    if not meta_results:
        return None, []
    for key in ("TextFieldsDetectorUSA", "TextFieldsDetectorBelarus", "TextFieldsDetectorRussia"):
        block = meta_results.get(key)
        if not block or not block.get("bbox"):
            continue
        labels = [str(b[-1]) for b in block["bbox"] if len(b) >= 7]
        return key, labels
    return None, []


def _needs_textfield_validation(doc_type: str, page_type: Optional[str]) -> bool:
    dt = (doc_type or "").lower()
    if dt.startswith("belarus"):
        return True
    if dt.startswith("usa_"):
        return True
    if dt.startswith("russia") and page_type == "passport_centerfold":
        return True
    return False


def _expected_labels(doc_type: str, page_type: Optional[str]) -> Optional[Set[str]]:
    dt = (doc_type or "").lower()
    if dt.startswith("belarus"):
        return BELARUS_TEXTFIELD_LABELS_EXPECTED
    if dt.startswith("usa_"):
        return USA_TEXTFIELD_LABELS_EXPECTED
    if dt.startswith("russia") and page_type == "passport_centerfold":
        return RUSSIA_CENTERFOLD_TEXTFIELD_LABELS_REQUIRED
    return None


def evaluate_capture(
    report: Dict[str, Any],
    meta_results: Optional[dict],
    *,
    min_doc_confidence: float = MIN_DOC_CONFIDENCE,
) -> Dict[str, Any]:
    """
    Возвращает ok=False, если нужно попросить пользователя переснять документ.

    Условия:
      - DocConf < min_doc_confidence
      - Blur в отчёте есть и не равен 'good'
      - не все ожидаемые поля TextFieldsDetector для типа документа
    """
    reasons: List[Dict[str, str]] = []

    quality = report.get("Quality") or {}
    doc_conf = quality.get("DocConf")
    if isinstance(doc_conf, (int, float)) and float(doc_conf) < float(min_doc_confidence):
        reasons.append(
            {
                "code": "low_doc_confidence",
                "title": "Низкая уверенность типа документа",
                "detail": f"Уверенность классификатора {float(doc_conf):.3f} < {min_doc_confidence:.2f}",
            }
        )

    blur = quality.get("Blur")
    if blur is not None and blur != "good":
        reasons.append(
            {
                "code": "blur_bad",
                "title": "Изображение размыто",
                "detail": f"Модель размытия: {blur} (ожидается good)",
            }
        )

    doc_type = (report.get("DocType") or "").strip()
    page_type = report.get("PassportPageType")
    if isinstance(page_type, dict):
        page_type = page_type.get("page_type")
    page_type_s = page_type if isinstance(page_type, str) else None

    if _needs_textfield_validation(doc_type, page_type_s):
        expected = _expected_labels(doc_type, page_type_s)
        _, found_list = _extract_text_field_labels(meta_results)
        found_set = set(found_list)
        if expected:
            missing = sorted(expected - found_set)
            if missing:
                preview = ", ".join(missing[:8])
                if len(missing) > 8:
                    preview += f" … (+{len(missing) - 8})"
                reasons.append(
                    {
                        "code": "incomplete_textfields",
                        "title": "Не все зоны полей найдены",
                        "detail": f"Нет детекций для: {preview}",
                    }
                )

    ok = len(reasons) == 0
    details_text = "\n".join(f"• {r['title']}: {r['detail']}" for r in reasons)

    if not ok:
        logger.warning(
            "capture_validation: отклонено (%s причин): %s",
            len(reasons),
            [r["code"] for r in reasons],
        )

    return {
        "ok": ok,
        "min_doc_confidence": min_doc_confidence,
        "reasons": reasons,
        "details_text": details_text,
    }
