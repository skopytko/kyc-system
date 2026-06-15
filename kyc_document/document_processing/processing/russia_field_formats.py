"""Строгие форматы и проверка заполненности полей российского паспорта (разворот)."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Pattern, Set, Tuple

# Серия и номер: 00 00 000000
SERIES_NUMBER_RE: Pattern[str] = re.compile(r"^\d{2} \d{2} \d{6}$")
# Код подразделения: 000-000
DEPARTMENT_CODE_RE: Pattern[str] = re.compile(r"^\d{3}-\d{3}$")
# Даты: 00.00.0000
DATE_DDMMYYYY_RE: Pattern[str] = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")

RUSSIA_STRICT_FORMAT_FIELDS: Tuple[Tuple[str, Pattern[str], str], ...] = (
    ("Licence_number", SERIES_NUMBER_RE, "серия и номер"),
    ("Issue_organisation_code", DEPARTMENT_CODE_RE, "код подразделения"),
    ("Birth_date", DATE_DDMMYYYY_RE, "дата рождения"),
    ("Issue_date", DATE_DDMMYYYY_RE, "дата выдачи"),
)

_STRICT_BY_FIELD: Dict[str, Tuple[Pattern[str], str]] = {
    fname: (pat, title) for fname, pat, title in RUSSIA_STRICT_FORMAT_FIELDS
}

# Подписи для сообщений об отказе (все обязательные поля разворота)
RUSSIA_CENTERFOLD_FIELD_TITLES: Dict[str, str] = {
    "Last_name_ru": "фамилия",
    "First_name_ru": "имя",
    "Middle_name_ru": "отчество",
    "Sex_ru": "пол",
    "Birth_date": "дата рождения",
    "Birth_place_ru": "место рождения",
    "Issue_organization_ru": "кем выдан",
    "Issue_date": "дата выдачи",
    "Issue_organisation_code": "код подразделения",
    "Licence_number": "серия и номер",
}


def _digits_only(text: str) -> str:
    t = str(text or "").replace("O", "0").replace("o", "0")
    return "".join(c for c in t if c.isdigit())


def format_series_number(text: str) -> Optional[str]:
    digits = _digits_only(text)
    if len(digits) != 10:
        return None
    return f"{digits[:2]} {digits[2:4]} {digits[4:10]}"


def format_department_code(text: str) -> Optional[str]:
    digits = _digits_only(text)
    if len(digits) != 6:
        return None
    return f"{digits[:3]}-{digits[3:6]}"


def format_date_ddmmyyyy(text: str) -> Optional[str]:
    raw = str(text or "").replace("O", "0").replace("o", "0")
    digits = "".join(c for c in raw if c.isdigit())
    if len(digits) != 8:
        return None
    out = f"{digits[:2]}.{digits[2:4]}.{digits[4:8]}"
    return out if DATE_DDMMYYYY_RE.match(out) else None


def is_valid_value(field_name: str, value: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False
    strict = _STRICT_BY_FIELD.get(field_name)
    if strict:
        return bool(strict[0].match(v))
    return True


def validate_russia_ocr_fields(
    ocr: Dict[str, str],
    required_fields: Iterable[str],
) -> List[dict]:
    """Отказ, если любое обязательное поле пустое или не проходит строгий формат."""
    reasons: List[dict] = []
    for field_name in required_fields:
        title = RUSSIA_CENTERFOLD_FIELD_TITLES.get(field_name, field_name)
        val = (ocr.get(field_name) or "").strip()
        if not val:
            reasons.append(
                {
                    "code": "empty_ocr_field",
                    "title": "Не заполнено поле",
                    "detail": f"Поле «{title}» пустое или не распознано",
                    "field": field_name,
                }
            )
            continue
        strict = _STRICT_BY_FIELD.get(field_name)
        if strict and not strict[0].match(val):
            fmt_title = strict[1]
            reasons.append(
                {
                    "code": "invalid_field_format",
                    "title": "Неверный формат поля",
                    "detail": f"Поле «{fmt_title}»: «{val}» — ожидается строгий формат",
                    "field": field_name,
                }
            )
    return reasons
