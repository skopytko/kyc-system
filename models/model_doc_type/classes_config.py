"""
Конфигурация классов для DocType.
Маппинг: cropped_path должен содержать указанные подстроки -> class_id.
"""

from typing import Optional

# Порядок классов (class_id -> имя)
CLASS_NAMES = [
    "belarus_passport_1996",      # 0
    "belarus_passport_biometric_2021",  # 1
    "russia_passport",            # 2
    "usa_alabama",               # 3
    "usa_alaska",                # 4
    "usa_arizona",               # 5
    "usa_arkansas",              # 6
    "usa_idaho",                 # 7
    "usa_iowa",                  # 8
    "usa_vermont",               # 9
    "usa_virginia",              # 10
    "usa_washington",            # 11
    "usa_wisconsin",             # 12
    "usa_wyoming",               # 13
]

N_CLASSES = len(CLASS_NAMES)


def cropped_path_to_class(cropped_path: str) -> Optional[int]:
    """Определяет class_id по пути к папке cropped. Возвращает None если не подходит."""
    path = cropped_path.replace("\\", "/").lower()
    if "belarus/passport_1996" in path:
        return 0
    if "belarus/passport_biometric_2021" in path:
        return 1
    if "russia/passport" in path:
        return 2
    if "/usa/alabama" in path:
        return 3
    if "/usa/alaska" in path:
        return 4
    if "/usa/arizona" in path:
        return 5
    if "/usa/arkansas" in path:
        return 6
    if "/usa/idaho" in path:
        return 7
    if "/usa/iowa" in path:
        return 8
    if "/usa/vermont" in path:
        return 9
    if "/usa/virginia" in path:
        return 10
    if "/usa/washington" in path:
        return 11
    if "/usa/wisconsin" in path:
        return 12
    if "/usa/wyoming" in path:
        return 13
    return None
