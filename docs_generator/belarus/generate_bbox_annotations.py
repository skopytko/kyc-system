import argparse
import csv
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Параметры обрезки из augment_photoreal.py (для out/out_aug)
CROP_LEFT = 30
CROP_TOP = 25
CROP_RIGHT = 20
CROP_BOTTOM = 70

# Оригинальный размер изображения (до обрезки) — обычный паспорт
ORIGINAL_WIDTH = 1240
ORIGINAL_HEIGHT = 1754

# Размер после обрезки
CROPPED_WIDTH = ORIGINAL_WIDTH - CROP_LEFT - CROP_RIGHT  # 1190
CROPPED_HEIGHT = ORIGINAL_HEIGHT - CROP_TOP - CROP_BOTTOM  # 1659

# Параметры шрифта из config.json (для точного измерения текста)
DEFAULT_FONT_PATH = "docs_generator/fonts/OCRBPro.TTF"
CHARACTER_SPACING = -3
SPACE_WIDTH = 0.88

# Маппинг полей из config.json к классам для детекции
FIELD_MAPPING = {
    "type": "type",
    "code_of_issuing": "code_of_issuing",
    "passport_no": "passport_no",
    "surname": "surname",
    "names": "names",
    "nationality": "nationality",
    "date_of_birth": "date_of_birth",
    "identification_no": "identification_no",
    "sex": "sex",
    "place_of_birth": "place_of_birth",
    "date_of_issue": "date_of_issue",
    "date_of_expiry": "date_of_expiry",
    "authority": "authority",
    "authority2": "authority2",
    "authority_code": "authority_code",
    "photo": "photo",
    "mini_photo": "mini_photo",
    "signature": "signature",
    "signature2": "signature2",
    "mrz_line1": "mrz_line1",
    "mrz_line2": "mrz_line2",
}

# Высота строки для текстовых полей (приблизительно, на основе font_size=32)
# Теперь вычисляется динамически на основе реального текста
TEXT_LINE_HEIGHT = 45

# Диапазоны обрезки углов (пиксели от края): ((x_min, x_max), (y_min, y_max))
# out: левый верх, правый верх (x от правого края), левый низ (y от нижнего), правый низ
CORNER_CROP_OUT = {
    "lt": ((25, 60), (20, 50)),   # left-top: x от левого, y от верхнего
    "rt": ((25, 45), (20, 50)),   # right-top: x от правого, y от верхнего
    "lb": ((25, 60), (60, 120)),  # left-bottom: x от левого, y от нижнего
    "rb": ((25, 45), (60, 120)),  # right-bottom: x от правого, y от нижнего
}
# out_bio: диапазоны для data/out_bio_annotated
CORNER_CROP_OUT_BIO = {
    "lt": ((225, 300), (100, 200)),   # left-top: x от левого, y от верхнего
    "rt": ((225, 330), (100, 200)),   # right-top: x от правого, y от верхнего
    "lb": ((225, 300), (60, 100)),    # left-bottom: x от левого, y от нижнего
    "rb": ((225, 330), (60, 100)),    # right-bottom: x от правого, y от нижнего
}


def get_random_corner_points(w: int, h: int, is_bio: bool) -> np.ndarray:
    """
    Генерирует 4 точки углов (четырехугольник) для обрезки изображения.
    Точки в порядке: левый верх, правый верх, правый низ, левый низ.
    """
    cfg = CORNER_CROP_OUT_BIO if is_bio else CORNER_CROP_OUT
    lt = (random.randint(cfg["lt"][0][0], cfg["lt"][0][1]), random.randint(cfg["lt"][1][0], cfg["lt"][1][1]))
    rt = (w - random.randint(cfg["rt"][0][0], cfg["rt"][0][1]), random.randint(cfg["rt"][1][0], cfg["rt"][1][1]))
    rb = (w - random.randint(cfg["rb"][0][0], cfg["rb"][0][1]), h - random.randint(cfg["rb"][1][0], cfg["rb"][1][1]))
    lb = (random.randint(cfg["lb"][0][0], cfg["lb"][0][1]), h - random.randint(cfg["lb"][1][0], cfg["lb"][1][1]))
    return np.array([lt, rt, rb, lb], dtype=np.float32)


def perspective_crop_image(img: np.ndarray, src_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Вырезает четырехугольник и выправляет перспективу в прямоугольник.

    Args:
        img: изображение (H, W, C) в BGR или RGB
        src_pts: 4 точки [[x,y], ...] в порядке LT, RT, RB, LB

    Returns:
        (cropped_img, transform_matrix, dst_w, dst_h)
    """
    lt, rt, rb, lb = src_pts
    top_w = np.linalg.norm(rt - lt)
    bottom_w = np.linalg.norm(rb - lb)
    left_h = np.linalg.norm(lb - lt)
    right_h = np.linalg.norm(rb - rt)
    dst_w = int((top_w + bottom_w) / 2)
    dst_h = int((left_h + right_h) / 2)
    dst_w = max(1, dst_w)
    dst_h = max(1, dst_h)
    dst_pts = np.array([[0, 0], [dst_w, 0], [dst_w, dst_h], [0, dst_h]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped = cv2.warpPerspective(img, M, (dst_w, dst_h), flags=cv2.INTER_LINEAR)
    return cropped, M, dst_w, dst_h


def transform_bbox_perspective(
    x1: int, y1: int, x2: int, y2: int,
    M: np.ndarray, dst_w: int, dst_h: int,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Трансформирует bbox через матрицу перспективы. Возвращает новый bbox в пространстве dst
    или None если bbox полностью выпал за пределы.
    """
    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    pts = pts.reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, M)
    xs = transformed[:, 0, 0]
    ys = transformed[:, 0, 1]
    nx1 = max(0, int(np.floor(xs.min())))
    ny1 = max(0, int(np.floor(ys.min())))
    nx2 = min(dst_w, int(np.ceil(xs.max())))
    ny2 = min(dst_h, int(np.ceil(ys.max())))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def load_config(config_path: Path) -> dict:
    """Загружает конфигурацию из JSON файла."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_csv_data(csv_path: Path) -> dict:
    """Загружает данные из CSV и возвращает словарь по passport_no."""
    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['passport_no']] = row
    return data


def measure_text_size(text: str, font_size: int, font_path: str = None) -> tuple:
    """
    Измеряет реальный размер текста с учетом параметров шрифта.
    Симулирует поведение TRDG с character_spacing и space_width.
    
    Args:
        text: Текст для измерения
        font_size: Размер шрифта
        font_path: Путь к файлу шрифта
        
    Returns:
        Кортеж (width, height) в пикселях
    """
    # Пробуем использовать TRDG напрямую, если доступен
    try:
        from trdg.generators import GeneratorFromStrings
        args = {
            "skewing_angle": 0,
            "random_skew": False,
            "fit": True,
            "margins": (0, 0, 0, 0),
            "size": font_size,
            "text_color": "#000000",
            "background_type": 1,
            "blur": 1,
            "random_blur": False,
            "character_spacing": CHARACTER_SPACING,
            "space_width": SPACE_WIDTH,
        }
        if font_path and Path(font_path).exists():
            args["fonts"] = [font_path]
        
        generator = GeneratorFromStrings([text], **args)
        img, _ = next(iter(generator))
        return (img.width, img.height)
    except (ImportError, TypeError, Exception):
        # Fallback на PIL если TRDG недоступен или не работает
        pass
    
    # Fallback: используем PIL для измерения
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Создаем временное изображение для измерения
    tmp = Image.new("RGB", (1, 1), color=(255, 255, 255))
    draw = ImageDraw.Draw(tmp)
    
    if len(text) == 0:
        return (0, font_size)
    
    # Измеряем каждый символ отдельно для точного учета character_spacing
    total_width = 0
    max_height = 0
    char_count = len(text)
    
    for i, char in enumerate(text):
        if char == ' ':
            # Для пробелов учитываем space_width
            space_bbox = draw.textbbox((0, 0), ' ', font=font)
            char_width = (space_bbox[2] - space_bbox[0]) * SPACE_WIDTH
        else:
            # Для обычных символов
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]
            max_height = max(max_height, char_height)
        
        total_width += char_width
        
        # Добавляем character_spacing между символами (но не после последнего)
        # character_spacing = -3 означает сжатие, но мы учитываем его как есть
        if i < len(text) - 1:
            total_width += CHARACTER_SPACING
    
    # Учитываем, что character_spacing может быть отрицательным (сжатие)
    # Добавляем запас, который увеличивается с количеством символов
    # Для одного символа запас не нужен, для нескольких - 3 пикселя на каждый символ
    # Это компенсирует накопление погрешностей при большом количестве символов
    # и разницу между PIL и TRDG
    if char_count == 1:
        padding = 0  # Для одного символа запас не нужен
    else:
        # Запас = 3 пикселя * количество символов
        padding = 3 * char_count
    
    width = max(1, int(total_width + padding))
    height = max(1, int(max_height) if max_height > 0 else font_size)
    
    return (width, height)


def adjust_coordinates(
    x: int, y: int, text: str, max_width: int,
    font_size: int = 32, font_path: str = None,
    crop_left: int = CROP_LEFT, crop_top: int = CROP_TOP,
    img_width: int = CROPPED_WIDTH, img_height: int = CROPPED_HEIGHT,
) -> tuple:
    """
    Корректирует координаты с учетом обрезки изображения и реального размера текста.
    Расширяет bbox на 2 пикселя во все стороны.
    """
    x1 = max(0, x - crop_left)
    y1 = max(0, y - crop_top)

    text_width, text_height = measure_text_size(text, font_size, font_path)
    width = min(text_width, max_width, img_width - x1)
    height = text_height

    x2 = x1 + width
    y2 = y1 + height

    padding = 2
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_width, x2 + padding)
    y2 = min(img_height, y2 + padding)

    return (x1, y1, x2, y2)


def get_bbox_for_field(
    field_name: str, field_config: dict,
    text: str = None, font_path: str = None,
    crop_left: int = CROP_LEFT, crop_top: int = CROP_TOP,
    img_width: int = CROPPED_WIDTH, img_height: int = CROPPED_HEIGHT,
) -> tuple:
    """Вычисляет bbox для поля. Возвращает (x1, y1, x2, y2, class_name) или None."""
    if "text" in field_config:
        x = field_config["position"]["x"]
        y = field_config["position"]["y"]
        max_width = field_config.get("max_width", 300)
        font_size = field_config.get("font_size", 32)

        if text is None:
            text = field_config.get("text", "")

        # authority_code — только 3 цифры
        if field_name == "authority_code":
            digits = "".join(c for c in str(text) if c.isdigit())
            text = (digits[-3:] if len(digits) >= 3 else digits.zfill(3))[:3]

        if field_name in ["authority", "authority2", "authority_code"]:
            text_width, text_height = measure_text_size(text, font_size, font_path)
            x1 = max(0, x - crop_left)
            y1 = max(0, y - crop_top)
            width = min(text_width, max_width, img_width - x1)
            x2 = x1 + width
            y2 = y1 + text_height
            padding = 2
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_width, x2 + padding)
            y2 = min(img_height, y2 + padding)
        else:
            x1, y1, x2, y2 = adjust_coordinates(
                x, y, text, max_width, font_size, font_path,
                crop_left, crop_top, img_width, img_height,
            )
        return (x1, y1, x2, y2, FIELD_MAPPING.get(field_name, field_name))

    elif "bbox" in field_config:
        bbox = field_config["bbox"]
        x1 = max(0, bbox["x1"] - crop_left)
        y1 = max(0, bbox["y1"] - crop_top)
        x2 = min(img_width, bbox["x2"] - crop_left)
        y2 = min(img_height, bbox["y2"] - crop_top)
        padding = 2
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_width, x2 + padding)
        y2 = min(img_height, y2 + padding)
        return (x1, y1, x2, y2, FIELD_MAPPING.get(field_name, field_name))

    return None


def draw_bboxes_on_image(img_path: Path, bboxes: list, output_path: Path, img: Optional[Image.Image] = None):
    """
    Рисует bbox на изображении и сохраняет результат.

    Args:
        img_path: Путь к исходному изображению (используется если img не передан)
        bboxes: Список bbox в формате (x1, y1, x2, y2, class_name)
        output_path: Путь для сохранения результата
        img: Опционально предзагруженное изображение (например после perspective crop)
    """
    if img is not None:
        img = img.convert("RGB") if img.mode != "RGB" else img
    else:
        img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    colors = {
        "type": (255, 0, 0),
        "code_of_issuing": (0, 255, 0),
        "passport_no": (0, 0, 255),
        "surname": (255, 255, 0),
        "names": (255, 0, 255),
        "nationality": (0, 255, 255),
        "date_of_birth": (128, 0, 0),
        "identification_no": (0, 128, 0),
        "sex": (0, 0, 128),
        "place_of_birth": (128, 128, 0),
        "date_of_issue": (128, 0, 128),
        "date_of_expiry": (0, 128, 128),
        "authority": (192, 192, 192),
        "authority2": (128, 128, 128),
        "authority_code": (192, 192, 192),
        "photo": (255, 165, 0),
        "mini_photo": (255, 165, 0),
        "signature": (255, 20, 147),
        "signature2": (255, 20, 147),
        "mrz_line1": (0, 200, 0),
        "mrz_line2": (0, 200, 0),
    }
    
    # Рисуем каждый bbox
    for x1, y1, x2, y2, class_name in bboxes:
        color = colors.get(class_name, (255, 255, 255))  # Белый по умолчанию
        
        # Рисуем прямоугольник
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Добавляем подпись класса
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        text = class_name
        bbox_text = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(bbox_text, fill=color, outline=color)
        draw.text((x1, y1 - 20), text, fill=(0, 0, 0), font=font)
    
    # Сохраняем результат
    img.save(output_path)
    print(f"Saved: {output_path}")


def get_mrz_bboxes(mrz_config: dict, crop_left: int, crop_top: int, img_width: int, img_height: int) -> list:
    """Возвращает список bbox для строк МЧЗ: [(x1,y1,x2,y2,'mrz_line1'), ...]."""
    if not mrz_config:
        return []
    mx = int(mrz_config.get("x", 90))
    my1 = int(mrz_config.get("y1", 1530))
    my2 = int(mrz_config.get("y2", 1590))
    mw = int(mrz_config.get("width", 1080))
    font_size = int(mrz_config.get("font_size", 32))
    line_h = int(font_size * 1.3)
    out = []
    for y, label in [(my1, "mrz_line1"), (my2, "mrz_line2")]:
        x1 = max(0, mx - crop_left)
        y1 = max(0, y - crop_top)
        x2 = min(img_width, mx + mw - crop_left)
        y2 = min(img_height, y + line_h - crop_top)
        if x2 > x1 and y2 > y1:
            out.append((x1, y1, x2, y2, label))
    return out


def generate_yolo_annotation(bboxes: list, img_width: int, img_height: int) -> str:
    """
    Генерирует аннотацию в формате YOLO.
    
    Args:
        bboxes: Список bbox в формате (x1, y1, x2, y2, class_name)
        img_width: Ширина изображения
        img_height: Высота изображения
        
    Returns:
        Строка с аннотациями в формате YOLO
    """
    # Маппинг классов к индексам
    class_names = sorted(set(FIELD_MAPPING.values()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    lines = []
    for x1, y1, x2, y2, class_name in bboxes:
        # Нормализуем координаты (центр и размер относительно изображения)
        center_x = ((x1 + x2) / 2) / img_width
        center_y = ((y1 + y2) / 2) / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        class_idx = class_to_idx.get(class_name, 0)
        lines.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    
    return "\n".join(lines)


FIELD_TO_CSV = {
    "type": "type",
    "code_of_issuing": "code_of_issuing",
    "passport_no": "passport_no",
    "surname": "surname",
    "names": "names",
    "nationality": "nationality",
    "date_of_birth": "date_of_birth",
    "identification_no": "identification_no",
    "sex": "sex",
    "place_of_birth": "place_of_birth",
    "date_of_issue": "date_of_issue",
    "date_of_expiry": "date_of_expiry",
    "authority": "authority",
    "authority2": "authority2",
    "authority_code": "authority_code",
}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Генерация bbox разметки для паспортов (out / out_bio)")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Папка с PNG (data/out или data/out_bio)")
    parser.add_argument("--config", type=Path, default=None,
                        help="config.json или config_bio.json")
    parser.add_argument("--csv", type=Path, default=None, help="CSV с данными (для подстановки текста)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Папка для изображений с нарисованными bbox")
    parser.add_argument("--labels-dir", type=Path, default=None,
                        help="Папка для YOLO .txt аннотаций")
    parser.add_argument("--no-crop", action="store_true",
                        help="Без обрезки (для out, out_bio — изображения полного размера)")
    parser.add_argument("--no-corner-crop", action="store_true",
                        help="Отключить обрезку углов с коррекцией перспективы")
    parser.add_argument("--no-augment", action="store_true",
                        help="Отключить photorealistic augmentation (augment_photoreal)")
    parser.add_argument("--no-draw-bboxes", action="store_true",
                        help="Не рисовать bbox на выходных изображениях")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed для воспроизводимой случайной обрезки углов")
    parser.add_argument("--limit", type=int, default=None, help="Макс. кол-во изображений для обработки")
    args = parser.parse_args()

    # Определяем режим по input-dir
    if args.input_dir is None:
        if (root / "data" / "out_bio").exists() and list((root / "data" / "out_bio").glob("*.png")):
            args.input_dir = root / "data" / "out_bio"
            args.config = args.config or root / "config_bio.json"
        else:
            args.input_dir = root / "data" / "out"
            args.config = args.config or root / "config.json"
    else:
        args.input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
        args.config = args.config or (root / "config_bio.json" if "bio" in str(args.input_dir) else root / "config.json")

    args.config = args.config if args.config.is_absolute() else root / args.config
    csv_path = args.csv or root / "data" / "data.csv"
    csv_path = csv_path if csv_path.is_absolute() else root / csv_path

    use_crop = not args.no_crop
    if "bio" in str(args.input_dir) or args.config.name == "config_bio.json":
        use_crop = False

    output_dir = args.output_dir or (args.input_dir.parent / (args.input_dir.name + "_annotated"))
    labels_dir = args.labels_dir or (args.input_dir.parent / (args.input_dir.name + "_labels"))
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    labels_dir = labels_dir if labels_dir.is_absolute() else root / labels_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    csv_data = load_csv_data(csv_path) if csv_path.exists() else {}

    font_candidates = [
        root / "fonts" / "OCRBPro.TTF",
        root.parent / "dataset" / "generator_docs" / "belarus" / "fonts" / "OCRBPro.TTF",
        root.parent / "dataset" / "generator_docs" / "belarus" / "fonts" / "OCRBPro.ttf",
    ]
    cfg_font = config.get("default_font_path")
    if cfg_font:
        font_candidates.insert(0, Path(cfg_font) if Path(cfg_font).is_absolute() else root.parent / cfg_font)
    font_path = next((p for p in font_candidates if p.exists()), None)

    image_files = sorted(args.input_dir.glob("*.png"))
    if args.limit:
        image_files = image_files[: args.limit]
    if not image_files:
        print(f"No images in {args.input_dir}")
        return

    crop_left = 0 if use_crop is False else CROP_LEFT
    crop_top = 0 if use_crop is False else CROP_TOP
    use_corner_crop = not args.no_corner_crop
    use_augment = not args.no_augment
    is_bio = "bio" in str(args.input_dir)

    if args.seed is not None:
        random.seed(args.seed)

    try:
        from augment_photoreal import pipeline as augment_pipeline
    except ImportError:
        augment_pipeline = None
        if use_augment:
            print("Warning: augment_photoreal not found, --no-augment implied")
            use_augment = False

    print(f"Processing {len(image_files)} images from {args.input_dir} (crop={use_crop}, corner_crop={use_corner_crop}, augment={use_augment})...")

    for idx, img_path in enumerate(image_files):
        passport_no = img_path.stem
        passport_data = csv_data.get(passport_no, {})

        img = Image.open(img_path)
        img_width, img_height = img.size

        if use_crop:
            cropped_w = img_width
            cropped_h = img_height
        else:
            cropped_w = img_width
            cropped_h = img_height

        bboxes = []
        fields = config.get("fields", {})

        for field_name, field_config in fields.items():
            if not isinstance(field_config, dict):
                continue
            text = None
            if field_name in FIELD_TO_CSV and FIELD_TO_CSV[field_name] in passport_data:
                text = passport_data[FIELD_TO_CSV[field_name]]
            if text is None and "text" in field_config:
                text = field_config["text"]

            bbox = get_bbox_for_field(
                field_name, field_config, text=text, font_path=str(font_path) if font_path else None,
                crop_left=crop_left, crop_top=crop_top, img_width=cropped_w, img_height=cropped_h,
            )
            if bbox:
                bboxes.append(bbox)

        mrz_cfg = config.get("mrz", {})
        if mrz_cfg:
            bboxes.extend(get_mrz_bboxes(mrz_cfg, crop_left, crop_top, cropped_w, cropped_h))

        # При use_crop bbox-и в «cropped» координатах (вычтены crop_left/crop_top).
        # Конвертируем в координаты изображения для perspective transform и отрисовки.
        if use_crop and (crop_left != 0 or crop_top != 0):
            bboxes = [
                (x1 + crop_left, y1 + crop_top, x2 + crop_left, y2 + crop_top, cls)
                for x1, y1, x2, y2, cls in bboxes
            ]

        out_img = img
        out_w, out_h = img_width, img_height

        if use_corner_crop:
            src_pts = get_random_corner_points(img_width, img_height, is_bio)
            img_np = np.array(img.convert("RGB"))
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            cropped, M, dst_w, dst_h = perspective_crop_image(img_bgr, src_pts)
            cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            out_img = Image.fromarray(cropped_rgb)
            out_w, out_h = dst_w, dst_h
            new_bboxes = []
            for x1, y1, x2, y2, cls in bboxes:
                t = transform_bbox_perspective(x1, y1, x2, y2, M, dst_w, dst_h)
                if t:
                    new_bboxes.append((*t, cls))
            bboxes = new_bboxes

        if use_augment and augment_pipeline is not None:
            aug_seed = (args.seed + idx * 1000) if args.seed is not None else None
            out_img = augment_pipeline(out_img, seed=aug_seed, skip_crop=True)

        if args.no_draw_bboxes:
            out_path = output_dir / img_path.name
            out_img.convert("RGB").save(out_path)
            print(f"Saved: {out_path}")
        else:
            draw_bboxes_on_image(img_path, bboxes, output_dir / img_path.name, img=out_img)

        yolo_text = generate_yolo_annotation(bboxes, out_w, out_h)
        (labels_dir / f"{img_path.stem}.txt").write_text(yolo_text, encoding="utf-8")

    class_names = sorted(set(FIELD_MAPPING.values()))
    (labels_dir / "classes.txt").write_text("\n".join(class_names), encoding="utf-8")

    print(f"Processed {len(image_files)} images")
    print(f"Annotated images: {output_dir}")
    print(f"YOLO labels: {labels_dir}")


if __name__ == "__main__":
    main()

