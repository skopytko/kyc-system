"""Проверка качества съёмки для ответа API: когда показывать запрос переснять документ."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

MIN_DOC_CONFIDENCE = 0.9

# USA: важные поля, без которых считаем распознавание неприемлемым.
# Остальные поля могут отсутствовать (из-за бликов/обрезки/особенностей штата/разметки).
USA_TEXTFIELD_LABELS_REQUIRED: Set[str] = frozenset(
    {
        "firstname",
        "lastname",
        "address",
        "sex",
        "dob",
        "dln",
        "iss",
        "exp",
    }
)

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

# Беларусь: важные поля (для решения «годится ли снимок»).
# Остальные (photo/signature/type/nationality/authority2/...) могут отсутствовать.
BELARUS_TEXTFIELD_LABELS_REQUIRED: Set[str] = frozenset(
    {
        "passport_no",
        "surname",
        "names",
        "nationality",
        "identification_no",
        "date_of_birth",
        "place_of_birth",
        "date_of_issue",
        "date_of_expiry",
        "sex",
        "authority",
        "code_of_issuing",
        "type",
    }
)
# TextFieldsDetectorUSA — names in docs_generator/usa/generator.py:
# IMAGE_FIELDS (0–2) + TEXT_FIELDS without dob_short (through dob).
USA_TEXTFIELD_LABELS_EXPECTED: Set[str] = frozenset(
    {
        "photo",
        "mini_photo",
        "handwritten_signature",
        "class",
        "end",
        "rest",
        "firstname",
        "lastname",
        "address",
        "sex",
        "hgt",
        "wgt",
        "eyes",
        "hair",
        "dd",
        "dln",
        "iss",
        "iss_duplicate",
        "exp",
        "dob",
    }
)

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
        # Для capture_validation требуем только важные поля.
        return BELARUS_TEXTFIELD_LABELS_REQUIRED
    if dt.startswith("usa_"):
        return USA_TEXTFIELD_LABELS_EXPECTED
    if dt.startswith("russia") and page_type == "passport_centerfold":
        return RUSSIA_CENTERFOLD_TEXTFIELD_LABELS_REQUIRED
    return None


def _missing_fields_allowed(doc_type: str, page_type: Optional[str]) -> int:
    dt = (doc_type or "").lower()
    if dt.startswith("usa_"):
        # Для USA используем критерий по обязательным полям (а не по количеству пропусков).
        return 0
    return 0


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
    textfields_debug: Dict[str, Any] = {}

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
        detector_key, found_list = _extract_text_field_labels(meta_results)
        found_set = set(found_list)
        if expected:
            missing = sorted(expected - found_set)
            allowed = _missing_fields_allowed(doc_type, page_type_s)

            # USA: отклоняем только если отсутствуют обязательные поля
            dt = (doc_type or "").lower()
            if dt.startswith("usa_"):
                missing_required = sorted(USA_TEXTFIELD_LABELS_REQUIRED - found_set)
                textfields_debug = {
                    "detector": detector_key,
                    "expected": sorted(expected),
                    "found": sorted(found_set),
                    "missing": missing,
                    "missing_count": len(missing),
                    "missing_allowed": allowed,
                    "required": sorted(USA_TEXTFIELD_LABELS_REQUIRED),
                    "missing_required": missing_required,
                }
                if missing_required:
                    reasons.append(
                        {
                            "code": "incomplete_textfields",
                            "title": "Не все зоны важных полей найдены",
                            "detail": "Не найдены важные зоны полей",
                        }
                    )
            else:
                textfields_debug = {
                    "detector": detector_key,
                    "expected": sorted(expected),
                    "found": sorted(found_set),
                    "missing": missing,
                    "missing_count": len(missing),
                    "missing_allowed": allowed,
                }
                if missing and len(missing) > allowed:
                    preview = ", ".join(missing[:8])
                    if len(missing) > 8:
                        preview += f" … (+{len(missing) - 8})"
                    reasons.append(
                        {
                            "code": "incomplete_textfields",
                            "title": "Не все зоны полей найдены",
                            "detail": (
                                f"Нет детекций для: {preview}"
                                + (f" (пропущено {len(missing)} > допуска {allowed})" if allowed else "")
                            ),
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
        # Debug info for UI: which textfields were missing/found/expected
        "textfields": textfields_debug,
    }
