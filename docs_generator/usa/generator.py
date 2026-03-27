import sys
import random
import os
import json
import csv
import textwrap
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple

# Для импорта augment_photoreal из belarus
_generator_dir = Path(__file__).resolve().parent
_project_root = _generator_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from psd_tools import PSDImage


# Путь по умолчанию (если запустить без аргументов)
DEFAULT_INPUT = "dataset/generator_docs/face/100807.png"

TEXT_FIELDS = [
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
    "dob_short",
]

IMAGE_FIELDS = [
    "photo",
    "mini_photo",
    "handwritten_signature",
]


_DERIVED_TOKEN_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)(?::month_short|\[(.*?)\])?\}")

MONTH_ABBR = {
    "01": "JAN", "02": "FEB", "03": "MAR", "04": "APR",
    "05": "MAY", "06": "JUN", "07": "JUL", "08": "AUG",
    "09": "SEP", "10": "OCT", "11": "NOV", "12": "DEC"
}


def _eval_derived_pattern(pattern: str, source: dict) -> str:
    """
    Простейший движок шаблонов.
    Поддерживает конструкции вида:
      {FIELD}              -> подстановка целиком
      {FIELD[0]}           -> символ по индексу
      {FIELD[-2:]}         -> срез, как в Python
      {FIELD[2:5]}         -> срез [start:end]
      {FIELD:month_short}  -> трёхбуквенное сокращение месяца (для DOB формата MM/DD/YYYY)
    """

    def repl(match: re.Match) -> str:
        field = match.group(1)
        slice_expr = match.group(2)  # для [start:end] или [i]
        full_match = match.group(0)  # полное совпадение для проверки :month_short

        raw = (
            source.get(field)
            or source.get(field.upper())
            or source.get(field.capitalize())
            or ""
        )
        value = str(raw)

        # Преобразование месяца в сокращение (проверяем полное совпадение)
        if ":month_short" in full_match:
            # Формат DOB: MM/DD/YYYY, берём первые 2 символа
            month = value[:2] if len(value) >= 2 else ""
            return MONTH_ABBR.get(month, month)

        # Без модификатора и без среза — возвращаем всё поле
        if not slice_expr:
            return value

        # Срез вида [start:end] или [i]
        expr = slice_expr.strip("[]")
        try:
            # Срез вида [start:end]
            if ":" in expr:
                start_s, end_s = expr.split(":", 1)
                start = int(start_s) if start_s else None
                end = int(end_s) if end_s else None
                return value[start:end]
            # Один индекс [i]
            idx = int(expr)
            if -len(value) <= idx < len(value):
                return value[idx]
            return ""
        except Exception:
            # На всякий случай не роняем генерацию
            return ""

    return _DERIVED_TOKEN_RE.sub(repl, pattern)


def apply_derived_fields(data: dict, derived_cfg: dict) -> dict:
    """
    Формирует дополнительные поля на основе существующих
    по правилам в конфиге ("derived_fields").

    Ключи в derived_cfg сравниваются в lower-case:
      "dob_short": {"pattern": "..."}
    """
    if not derived_cfg:
        return data

    # Работать с копией, чтобы не портить исходные данные CSV
    result = dict(data)
    normalized_rules = {str(k).lower(): v for k, v in derived_cfg.items()}

    for field_name, rule in normalized_rules.items():
        pattern = rule.get("pattern")
        if not pattern:
            continue

        value = _eval_derived_pattern(str(pattern), result)

        # Сохраняем под разными вариантами имени,
        # чтобы insert_text мог его найти.
        result[field_name] = value
        result[field_name.upper()] = value
        result[field_name.capitalize()] = value

    return result


def center_crop(img: Image.Image, keep_w: float = 1.0, keep_h: float = 1.0):
    """Кроп по центру: оставляем keep_w/keep_h долю от ширины/высоты."""
    w, h = img.size
    nw = max(1, int(w * float(keep_w)))
    nh = max(1, int(h * float(keep_h)))
    left = (w - nw) // 2
    top = (h - nh) // 2
    return img.crop((left, top, left + nw, top + nh)), (left, top)


def get_corner_points_from_config(w: int, h: int, cfg: dict) -> np.ndarray:
    """
    Генерирует 4 угловые точки по конфигу corner_crop.
    cfg: lt, rt, rb, lb — каждый с w: [min,max], h: [min,max] (доли 0..1 которые ОСТАВЛЯЕМ от края).
    LT: точка (x от левого, y от верхнего), RT: (x от правого, y от верхнего), и т.д.
    """
    def rnd(key: str, sub: str) -> float:
        r = cfg.get(key, {}).get(sub, [0.92, 0.95])
        lo, hi = (r[0], r[1]) if isinstance(r, list) else (r, r)
        return random.uniform(float(lo), float(hi))

    # lt: x = w*(1-keep_w), y = h*(1-keep_h)  — точка от левого верхнего угла
    keep_lt_w = rnd("lt", "w")
    keep_lt_h = rnd("lt", "h")
    lt = (w * (1 - keep_lt_w), h * (1 - keep_lt_h))

    keep_rt_w = rnd("rt", "w")
    keep_rt_h = rnd("rt", "h")
    rt = (w * keep_rt_w, h * (1 - keep_rt_h))  # x от правого = w*keep

    keep_rb_w = rnd("rb", "w")
    keep_rb_h = rnd("rb", "h")
    rb = (w * keep_rb_w, h * keep_rb_h)

    keep_lb_w = rnd("lb", "w")
    keep_lb_h = rnd("lb", "h")
    lb = (w * (1 - keep_lb_w), h * keep_lb_h)

    return np.array([lt, rt, rb, lb], dtype=np.float32)


def perspective_crop_image(img: np.ndarray, src_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Вырезает четырёхугольник и выпрямляет перспективу в прямоугольник."""
    lt, rt, rb, lb = src_pts
    top_w = np.linalg.norm(rt - lt)
    bottom_w = np.linalg.norm(rb - lb)
    left_h = np.linalg.norm(lb - lt)
    right_h = np.linalg.norm(rb - rt)
    dst_w = max(1, int((top_w + bottom_w) / 2))
    dst_h = max(1, int((left_h + right_h) / 2))
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return cropped, M, dst_w, dst_h


def transform_bbox_perspective(
    x1: float, y1: float, x2: float, y2: float,
    M: np.ndarray, dst_w: int, dst_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Трансформирует bbox через матрицу перспективы."""
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, M)
    xs = transformed[:, 0, 0]
    ys = transformed[:, 0, 1]
    nx1 = max(0.0, float(np.floor(xs.min())))
    ny1 = max(0.0, float(np.floor(ys.min())))
    nx2 = min(dst_w, float(np.ceil(xs.max())))
    ny2 = min(dst_h, float(np.ceil(ys.max())))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def corner_crop(
    img: Image.Image,
    corner_cfg: dict,
    boxes: List[Tuple[int, float, float, float, float]],
) -> Tuple[Image.Image, List[Tuple[int, float, float, float, float]]]:
    """
    Обрезка по углам с коррекцией перспективы.
    Возвращает (cropped_image, transformed_boxes).
    """
    w, h = img.size
    src_pts = get_corner_points_from_config(w, h, corner_cfg)
    img_np = np.array(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cropped, M, dst_w, dst_h = perspective_crop_image(img_bgr, src_pts)
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    out_img = Image.fromarray(cropped_rgb)

    new_boxes = []
    for cid, x1, y1, x2, y2 in boxes:
        t = transform_bbox_perspective(float(x1), float(y1), float(x2), float(y2), M, dst_w, dst_h)
        if t:
            new_boxes.append((cid, t[0], t[1], t[2], t[3]))
    return out_img, new_boxes


def preprocess_face_photo(
    input_path,
    output_path=None,
    color_from=None,
    color_to=None,
    aspect_ratio=None,
    grayscale=False,
    opacity=1.0,
    border_blur=0,
    border_opacity=1.0,
    border_size=0.15,
):
    """Предварительная обработка фотографии лица."""
    img = Image.open(input_path).convert("RGBA")
    orig_alpha = img.split()[-1]
    filled_bg = bool(color_from and color_to)

    # Закрашиваем фон, если указаны цвета
    if filled_bg:
        t = random.random()
        r = int((1 - t) * color_from[0] + t * color_to[0])
        g = int((1 - t) * color_from[1] + t * color_to[1])
        b = int((1 - t) * color_from[2] + t * color_to[2])
        background = Image.new("RGBA", img.size, (r, g, b, 255))
        result = Image.alpha_composite(background, img).convert("RGBA")
        orig_alpha = result.split()[-1]
    else:
        result = img.copy()

    # Обрезаем, если указано соотношение
    if aspect_ratio:
        w, h = result.size
        aspect_w, aspect_h = aspect_ratio
        target_w = int(h * aspect_w / aspect_h)
        if w > target_w:
            left = (w - target_w) // 2
            result = result.crop((left, 0, left + target_w, h))
            orig_alpha = orig_alpha.crop((left, 0, left + target_w, h))

    # Черно-белое изображение (если указано)
    if grayscale:
        gray = result.convert("L")
        result = Image.merge("RGBA", (gray, gray, gray, orig_alpha))
    else:
        # если фон не заливали, сохраняем исходную альфу
        if not filled_bg:
            r, g, b, _ = result.split()
            result = Image.merge("RGBA", (r, g, b, orig_alpha))

    # Задаем прозрачность (умножаем альфу)
    alpha_value = max(0, min(1, float(opacity)))
    if alpha_value < 1:
        a = result.split()[-1].point(lambda v: int(v * alpha_value))
        r, g, b, _ = result.split()
        result = Image.merge("RGBA", (r, g, b, a))

    # Применяем размытые и полупрозрачные границы
    if border_blur > 0 or border_opacity < 1.0:
        w, h = result.size
        # Создаем градиентную маску для границ
        border_mask = Image.new("L", (w, h), 255)
        
        # Размер области границы (в процентах от меньшей стороны)
        border_size_pixels = min(w, h) * float(border_size)
        
        # Создаем градиент от центра к краям
        pixels = border_mask.load()
        for y in range(h):
            for x in range(w):
                # Расстояние до ближайшего края
                dist_to_left = x
                dist_to_right = w - x
                dist_to_top = y
                dist_to_bottom = h - y
                min_dist = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
                
                if min_dist < border_size_pixels:
                    # Градиент: от 255 в центре до меньшего значения на краю
                    ratio = min_dist / border_size_pixels
                    # Применяем размытие и прозрачность
                    alpha_val = int(255 * border_opacity * ratio)
                    pixels[x, y] = alpha_val
        
        # Применяем размытие к маске, если указано
        if border_blur > 0:
            border_mask = border_mask.filter(ImageFilter.GaussianBlur(border_blur))
        
        # Применяем маску к альфа-каналу
        current_alpha = result.split()[-1]
        # Умножаем текущую альфу на маску границ
        new_alpha = Image.new("L", (w, h))
        alpha_pixels = new_alpha.load()
        current_alpha_pixels = current_alpha.load()
        mask_pixels = border_mask.load()
        
        for y in range(h):
            for x in range(w):
                alpha_pixels[x, y] = min(255, (current_alpha_pixels[x, y] * mask_pixels[x, y]) // 255)
        
        r, g, b, _ = result.split()
        result = Image.merge("RGBA", (r, g, b, new_alpha))

    # Сохраняем (если попросили)
    if output_path:
        result.save(output_path)

    return result


def insert_text(
    image: Image.Image,
    field_name: str,
    font_path: str,
    config: dict,
    data: dict,
    boxes_out=None,
    class_id: Optional[int] = None,
) -> Image.Image:
    """Вставляет текст из данных на изображение для указанного поля."""
    if field_name not in config:
        return image
    
    field_cfg = config[field_name]
    img_text = image.convert("RGBA")
    draw0 = ImageDraw.Draw(img_text)
    font = ImageFont.truetype(font_path, field_cfg.get("font_size", 24))
    
    color = field_cfg.get("color", [0, 0, 0])
    fill_color = (*color, 255) if len(color) == 3 else tuple(color)

    
    # Пробуем разные варианты названий полей в CSV/вычисленных полей
    text = (
        data.get(field_name)
        or data.get(field_name.upper())
        or data.get(field_name.capitalize())
        or ""
    )
    if not text:
        return img_text
    
    trdg = config.get("trdg", {})
    b0, b1 = trdg.get("blur_radius", [0.0, 0.0])
    a0, a1 = trdg.get("alpha", [255, 255])
    space_w = int(trdg.get("space_width", 1))
    line_sp = max(0, int(trdg.get("line_spacing", 0)))
    bold_s = float(trdg.get("bold_strength", trdg.get("bold_radius", 0)))
    pad_l, pad_t, pad_r, pad_b = (trdg.get("bbox_pad") or [0, 0, 0, 0])
    pad_l, pad_t, pad_r, pad_b = int(pad_l), int(pad_t), int(pad_r), int(pad_b)
    blur = random.uniform(b0, b1)
    # Проверяем, есть ли индивидуальная прозрачность для поля
    if "opacity" in field_cfg:
        opacity_value = max(0.0, min(1.0, float(field_cfg["opacity"])))
        alpha = int(opacity_value * 255)
    else:
        alpha = random.randint(int(a0), int(a1))

    text = text.replace(" ", " " * max(1, space_w))

    # bbox-режим: рисуем в прямоугольник и переносим строки
    bbox = field_cfg.get("bbox")
    if bbox:
        w = max(1, int(bbox["x2"] - bbox["x1"]))
        h = max(1, int(bbox["y2"] - bbox["y1"]))
        avg_char_w = max(1, (font.getbbox("M")[2] - font.getbbox("M")[0]))
        chars_per_line = max(1, int(w / avg_char_w))
        lines = textwrap.wrap(text, width=chars_per_line)
        text = "\n".join(lines)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Рисуем построчно по baseline, чтобы bbox совпадали 1:1 с отрисовкой
        dlayer = ImageDraw.Draw(layer)
        ascent, _ = font.getmetrics()
        # Вычисляем высоту строки: от top до bottom для тестовой строки
        line_bbox = font.getbbox("Ag")
        line_height = line_bbox[3] - line_bbox[1]  # полная высота строки
        baseline_y = ascent
        for line in lines:
            dlayer.text((0, baseline_y), line, fill=fill_color, font=font, anchor="ls")
            baseline_y += line_height + line_sp  # следующая baseline с учетом интервала
        x0, y0 = int(bbox["x1"]), int(bbox["y1"])
        # bbox по строкам - используем ту же логику
        if boxes_out is not None and class_id is not None:
            baseline_y = int(bbox["y1"]) + ascent
            for line in lines:
                x1, y1, x2, y2 = draw0.textbbox(
                    (int(bbox["x1"]), baseline_y),
                    line,
                    font=font,
                    anchor="ls",
                )
                boxes_out.append((class_id, x1 - pad_l, y1 - pad_t, x2 + pad_r, y2 + pad_b))
                baseline_y += line_height + line_sp  # следующая baseline с учетом интервала
    else:
        # point-режим: координата (x,y) — baseline-left (нижний левый угол текста)
        bb = font.getbbox(text)
        ascent, _ = font.getmetrics()
        pad = 6
        w = max(1, int(bb[2] - bb[0] + pad * 2))
        h = max(1, int(bb[3] - bb[1] + pad * 2 + ascent))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Рисуем так, чтобы baseline текста был на y = pad + ascent
        ImageDraw.Draw(layer).text((pad, pad + ascent), text, fill=fill_color, font=font, anchor="ls")
        # Композитим так, чтобы baseline попал ровно в (x,y)
        x0 = int(field_cfg["x"]) - pad
        y0 = int(field_cfg["y"]) - (pad + ascent)
        if boxes_out is not None and class_id is not None:
            x1, y1, x2, y2 = draw0.textbbox(
                (int(field_cfg["x"]), int(field_cfg["y"])),
                text,
                font=font,
                anchor="ls",
            )
            boxes_out.append((class_id, x1 - pad_l, y1 - pad_t, x2 + pad_r, y2 + pad_b))

    # TRDG-подобная "печатность": тоньше/толще (мягко) + прозрачность + лёгкое размытие
    if bold_s:
        k = 3  # мягкий эффект
        filt = layer.filter((ImageFilter.MaxFilter if bold_s > 0 else ImageFilter.MinFilter)(k))
        layer = Image.blend(layer, filt, min(1.0, abs(bold_s)))
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    layer.putalpha(layer.split()[-1].point(lambda v: v * alpha // 255))

    img_text.alpha_composite(layer, dest=(x0, y0))
    return img_text.convert("RGB")


def generate_document(
    face_photo_path,
    config_path: str = "docs_generator/usa/config.json",
    row_index: int = 0,
    export_boxes: bool = False,
    draw_boxes_on_output: Optional[bool] = None,
):
    """Создает документ: вставляет фото, затем текст из CSV. При export_boxes возвращает (image, boxes)."""
    # Загружаем конфиг
    with open(config_path) as f:
        config = json.load(f)
    
    # Загружаем PSD
    psd = PSDImage.open(config["template_image"])
    base_img = psd.composite().convert("RGBA")
    
    # Загружаем данные из CSV для определения пола
    sex = None
    csv_path = config.get("personal_data_csv")
    if csv_path and Path(csv_path).exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if 0 <= row_index < len(rows):
                sex = rows[row_index].get("Sex", "").upper()
    
    # Выбираем фото в зависимости от пола
    face_src = face_photo_path
    if sex == "F":
        face_files = sorted(Path("dataset/generator_docs/face/F").glob("*.png"))
        if face_files:
            face_src = str(random.choice(face_files))
    elif sex == "M":
        face_files = sorted(Path("dataset/generator_docs/face/M").glob("*.png"))
        if face_files:
            face_src = str(random.choice(face_files))
    else:
        # Fallback: случайно из обеих папок или корня
        all_files = list(Path("dataset/generator_docs/face/F").glob("*.png")) + \
                   list(Path("dataset/generator_docs/face/M").glob("*.png"))
        if all_files:
            face_src = str(random.choice(all_files))

    # Вставляем mini_photo, mini1_photo, mini2_photo и photo
    for block_name in ["mini_photo", "mini1_photo", "mini2_photo", "photo"]:
        if block_name not in config: 
            continue
        block = config[block_name] 
        face_img = preprocess_face_photo(
            face_src,
            color_from=block.get("color_from"),
            color_to=block.get("color_to"),
            aspect_ratio=block.get("aspect_ratio"),
            grayscale=block.get("grayscale", False),
            opacity=block.get("opacity", 1.0),
            border_blur=block.get("border_blur", 0),
            border_opacity=block.get("border_opacity", 1.0),
            border_size=block.get("border_size", 0.15),
        )
        bbox = block["bbox"]
        face_img = face_img.resize((bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]), Image.LANCZOS)
        base_img.paste(face_img, (bbox["x1"], bbox["y1"]), face_img)
        if block.get("top_layer_name"): 
            for layer in psd.descendants():
                if layer.name == block["top_layer_name"]:
                    layer_img = layer.topil().convert("RGBA")
                    if layer.bbox:
                        base_img.paste(layer_img, (layer.bbox[0], layer.bbox[1]), layer_img)
                    break
    
    # Вставляем handwritten_signature
    sig_files = list(Path("dataset/generator_docs/handwritten_signatures").glob("*.png"))
    if "handwritten_signature" in config and sig_files: # это
        block = config["handwritten_signature"]
        sig_img = preprocess_face_photo(random.choice(sig_files))
        bbox = block["bbox"]
        sig_img = sig_img.resize((bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]), Image.LANCZOS)
        base_img.paste(sig_img, (bbox["x1"], bbox["y1"]), sig_img)
    
    # Вставляем текст из CSV
    csv_path = config.get("personal_data_csv")
    if csv_path and Path(csv_path).exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            if 0 <= row_index < len(rows):
                # Базовые данные из CSV
                data = rows[row_index]
                # Вычисляем дополнительные поля по шаблонам из конфига (derived_fields)
                derived_cfg = config.get("derived_fields", {})
                data = apply_derived_fields(data, derived_cfg)
                font_path = config.get("font_main")
                boxes = [] if export_boxes else None
                if boxes is not None:
                    # bbox для блоков-картинок (photo/mini_photo/signature)
                    for cid, name in enumerate(IMAGE_FIELDS):
                        if name in config and "bbox" in config[name]:
                            b = config[name]["bbox"]
                            boxes.append((cid, b["x1"], b["y1"], b["x2"], b["y2"]))

                base_cid = len(IMAGE_FIELDS)
                for cid, field_name in enumerate(TEXT_FIELDS, start=base_cid):
                    base_img = insert_text(
                        base_img,
                        field_name,
                        font_path,
                        config,
                        data,
                        boxes_out=boxes,
                        class_id=cid,
                    )

    out = base_img.convert("RGB")

    if export_boxes:
        should_draw = draw_boxes_on_output if draw_boxes_on_output is not None else config.get("draw_boxes")
        if should_draw:
            w, h = out.size
            draw = ImageDraw.Draw(out)
            label_font = ImageFont.load_default()
            id_to_name = {i: n for i, n in enumerate(IMAGE_FIELDS)}
            base_cid = len(IMAGE_FIELDS)
            id_to_name.update({base_cid + i: n for i, n in enumerate(TEXT_FIELDS)})
            for cid, x1, y1, x2, y2 in boxes:
                draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                name = id_to_name.get(cid, str(cid))
                tx = max(0, int(x1))
                ty = max(0, int(y1) - 12)
                draw.text((tx, ty), name, fill="red", font=label_font)

        return out, boxes or []

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генератор документов Idaho")
    parser.add_argument("--config", default="docs_generator/usa/idaho.json", help="Путь к конфигу")
    parser.add_argument("--face", default=DEFAULT_INPUT, help="Путь к фото лица")
    parser.add_argument("--export-labels", action="store_true", help="Сохранять YOLO bbox рядом с изображением")
    args = parser.parse_args()
    
    # Загружаем конфиг
    with open(args.config) as f:
        config = json.load(f)
    
    # Читаем CSV
    csv_path = config.get("personal_data_csv")
    if not csv_path or not Path(csv_path).exists():
        print(f"Ошибка: CSV файл не найден: {csv_path}")
        exit(1)
    
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    
    # Определяем имя папки из имени конфига (например, arizona.json -> arizona)
    config_path = Path(args.config)
    dataset_name = config_path.stem  # убираем расширение .json
    
    # Определяем пути для raw и cropped
    crop_cfg = config.get("crop", {})
    corner_cfg = crop_cfg.get("corner_crop")  # новый формат: обрезка по углам
    keep_w = float(crop_cfg.get("keep_w", 1.0))
    keep_h = float(crop_cfg.get("keep_h", 1.0))
    has_crop = corner_cfg is not None or (keep_w < 1.0 or keep_h < 1.0)
    use_corner_crop = corner_cfg is not None
    augment_cropped = config.get("augment_cropped", False)

    try:
        belarus_dir = _project_root / "docs_generator" / "belarus"
        if str(belarus_dir) not in sys.path:
            sys.path.insert(0, str(belarus_dir))
        from augment_photoreal import pipeline as augment_pipeline
    except Exception:
        augment_pipeline = None
        if augment_cropped:
            print("Warning: augment_photoreal не найден, augment_cropped отключён")
            augment_cropped = False
    
    # Создаём директории для вывода
    base_output_dir = Path(f"dataset/usa/{dataset_name}")
    raw_dir = base_output_dir / "raw"
    cropped_dir = base_output_dir / "cropped"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    if has_crop:
        cropped_dir.mkdir(parents=True, exist_ok=True)
    
    # Директории для лейблов
    raw_labels_dir = raw_dir / "labels"
    cropped_labels_dir = cropped_dir / "labels" if has_crop else None
    
    if args.export_labels:
        raw_labels_dir.mkdir(parents=True, exist_ok=True)
        if has_crop and cropped_labels_dir:
            cropped_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Генерируем документы для всех строк
    print(f"Генерация {len(rows)} документов...")
    for i, row in enumerate(rows):
        if args.export_labels:
            # При corner_crop не рисуем в generate_document — иначе боксы попадут в перспективу и будут дублироваться
            skip_draw = use_corner_crop
            img_raw, boxes_raw = generate_document(
                args.face, args.config, row_index=i, export_boxes=True,
                draw_boxes_on_output=False if skip_draw else None,
            )
        else:
            img_raw = generate_document(args.face, args.config, row_index=i)
            boxes_raw = []

        # Сохраняем raw версию (без кропа)
        raw_path = raw_dir / f"{dataset_name}_{i+1:04d}.png"
        if use_corner_crop and config.get("draw_boxes") and args.export_labels and boxes_raw:
            out_raw = img_raw.convert("RGB")
            draw = ImageDraw.Draw(out_raw)
            id_to_name = {i: n for i, n in enumerate(IMAGE_FIELDS)}
            id_to_name.update({len(IMAGE_FIELDS) + i: n for i, n in enumerate(TEXT_FIELDS)})
            label_font = ImageFont.load_default()
            for cid, x1, y1, x2, y2 in boxes_raw:
                draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                name = id_to_name.get(cid, str(cid))
                draw.text((max(0, int(x1)), max(0, int(y1) - 12)), name, fill="red", font=label_font)
            out_raw.save(raw_path)
        else:
            img_raw.save(raw_path)

        if args.export_labels:
            w, h = img_raw.size
            raw_label_path = raw_labels_dir / f"{dataset_name}_{i+1:04d}.txt"
            with open(raw_label_path, "w", encoding="utf-8") as f:
                for cid, x1, y1, x2, y2 in boxes_raw:
                    x1 = max(0.0, min(float(w), float(x1)))
                    x2 = max(0.0, min(float(w), float(x2)))
                    y1 = max(0.0, min(float(h), float(y1)))
                    y2 = max(0.0, min(float(h), float(y2)))
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)
                    if bw <= 0 or bh <= 0:
                        continue
                    xc = (x1 + x2) / 2.0 / w
                    yc = (y1 + y2) / 2.0 / h
                    ww = bw / w
                    hh = bh / h
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        # Сохраняем cropped версию (с кропом), если нужно
        if has_crop:
            if use_corner_crop:
                img_cropped, boxes_cropped = corner_crop(
                    img_raw.copy(), corner_cfg,
                    [(cid, x1, y1, x2, y2) for cid, x1, y1, x2, y2 in boxes_raw],
                )
            else:
                img_cropped, (dx, dy) = center_crop(img_raw.copy(), keep_w, keep_h)
                boxes_cropped = [
                    (cid, float(x1) - dx, float(y1) - dy, float(x2) - dx, float(y2) - dy)
                    for cid, x1, y1, x2, y2 in boxes_raw
                ]

            cropped_path = cropped_dir / f"{dataset_name}_{i+1:04d}.png"

            # Аугментация только для cropped
            if augment_cropped and augment_pipeline is not None:
                aug_seed = (hash(f"{dataset_name}_{i}") % (2**32)) if i is not None else None
                img_cropped = augment_pipeline(img_cropped, seed=aug_seed, skip_crop=True)

            # Рисуем bbox на cropped если draw_boxes и export_labels
            if config.get("draw_boxes") and args.export_labels and boxes_cropped:
                out_draw = img_cropped.convert("RGB")
                draw = ImageDraw.Draw(out_draw)
                w_c, h_c = img_cropped.size
                id_to_name = {i: n for i, n in enumerate(IMAGE_FIELDS)}
                id_to_name.update({len(IMAGE_FIELDS) + i: n for i, n in enumerate(TEXT_FIELDS)})
                label_font = ImageFont.load_default()
                for cid, x1, y1, x2, y2 in boxes_cropped:
                    draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
                    name = id_to_name.get(cid, str(cid))
                    draw.text((max(0, int(x1)), max(0, int(y1) - 12)), name, fill="red", font=label_font)
                out_draw.save(cropped_path)
            else:
                img_cropped.save(cropped_path)

            if args.export_labels and cropped_labels_dir:
                w_c, h_c = img_cropped.size
                cropped_label_path = cropped_labels_dir / f"{dataset_name}_{i+1:04d}.txt"
                with open(cropped_label_path, "w", encoding="utf-8") as f:
                    for cid, x1, y1, x2, y2 in boxes_cropped:
                        x1_c = max(0.0, min(float(w_c), float(x1)))
                        x2_c = max(0.0, min(float(w_c), float(x2)))
                        y1_c = max(0.0, min(float(h_c), float(y1)))
                        y2_c = max(0.0, min(float(h_c), float(y2)))
                        bw = max(0.0, x2_c - x1_c)
                        bh = max(0.0, y2_c - y1_c)
                        if bw <= 0 or bh <= 0:
                            continue
                        xc = (x1_c + x2_c) / 2.0 / w_c
                        yc = (y1_c + y2_c) / 2.0 / h_c
                        ww = bw / w_c
                        hh = bh / h_c
                        f.write(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

        print(f"  [{i+1}/{len(rows)}] Сохранено: {raw_path}" + (f" и {cropped_path}" if has_crop else ""))

    print(f"\nГотово! Создано {len(rows)} документов:")
    print(f"  - raw: {raw_dir}")
    if has_crop:
        print(f"  - cropped: {cropped_dir}")

