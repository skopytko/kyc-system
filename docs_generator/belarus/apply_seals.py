import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
from PIL import Image, ImageFilter, ImageChops

import cv2


def _list_image_files_recursive(root: Path) -> List[Path]:
    """Находит все изображения в директории рекурсивно."""
    supported = {".png", ".jpg", ".jpeg", ".webp"}
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported:
            files.append(p)
    return files


def _remove_white_background_rgba(img: Image.Image, tolerance: int = 15, soften: float = 1.5) -> Image.Image:
    """Удаляет белый фон из изображения."""
    # Make sure RGBA
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[..., :3].astype(np.int16)
    alpha = arr[..., 3].astype(np.uint8)
    # Chebyshev distance from white (255,255,255)
    dist = np.max(255 - rgb, axis=2)
    # Foreground where distance > tolerance (i.e., not near white)
    fg = dist > int(max(0, min(255, tolerance)))
    mask = np.where(fg, 255, 0).astype(np.uint8)

    mask_img = Image.fromarray(mask, mode="L")
    if soften and soften > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=float(soften)))

    # Combine with original alpha (keep transparent if it was already transparent)
    final_alpha = Image.fromarray(np.minimum(alpha, np.array(mask_img, dtype=np.uint8)))
    out = rgba.copy()
    out.putalpha(final_alpha)
    return out


def _apply_otsu_alpha(
    img: Image.Image,
    soften: float = 1.0,
    open_kernel: int = 3,
    dilate_iter: int = 0,
) -> Image.Image:
    """Применяет бинаризацию Otsu для создания альфа-канала."""
    rgb = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.8)
    # Invert so strokes (dark) become bright for THRESH_BINARY
    inv = 255 - gray
    _, mask = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    k = max(1, int(open_kernel))
    if k % 2 == 0:
        k += 1
    if k >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if int(dilate_iter) > 0:
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel_d, iterations=int(dilate_iter))
    if soften and soften > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), float(soften))
    rgba = img.convert("RGBA")
    alpha_existing = np.array(rgba.split()[3], dtype=np.uint8)
    final_alpha = np.minimum(alpha_existing, mask.astype(np.uint8))
    out = rgba.copy()
    out.putalpha(Image.fromarray(final_alpha))
    return out


def _apply_random_seal_color(img: Image.Image, strength: float = 0.7) -> Image.Image:
    """
    Применяет случайный цветовой оттенок к печати в диапазоне темно-синих, синих, 
    фиолетовых и зеленоватых оттенков для реалистичного вида.
    
    Args:
        img: Изображение печати (RGBA)
        strength: Сила эффекта (0.0-1.0)
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Определяем диапазоны цветов для разных оттенков печатей
    color_presets = [
        # Темно-синие оттенки
        (15, 25, 80),
        (20, 32, 96),
        (25, 40, 110),
        (30, 50, 120),
        # Синие оттенки
        (40, 60, 150),
        (50, 80, 180),
        (60, 100, 200),
        (70, 110, 210),
        # Фиолетовые оттенки
        (80, 50, 120),
        (100, 60, 150),
        (120, 70, 180),
        (130, 80, 190),
        # Зеленовато-синие оттенки
        (30, 80, 100),
        (40, 100, 120),
        (50, 120, 140),
        (60, 130, 150),
    ]
    
    # Выбираем случайный цвет из пресетов
    base_color = random.choice(color_presets)
    
    # Добавляем небольшую случайную вариацию (±10-20 для каждого канала)
    variation = random.randint(-15, 15)
    r = max(10, min(200, base_color[0] + variation))
    g = max(10, min(200, base_color[1] + variation))
    b = max(10, min(220, base_color[2] + variation))
    
    color = (r, g, b)
    
    # Применяем цветовое умножение
    r_ch, g_ch, b_ch, a_ch = img.split()
    rgb = Image.merge("RGB", (r_ch, g_ch, b_ch))
    color_layer = Image.new("RGB", rgb.size, color)
    multiplied = ImageChops.multiply(rgb, color_layer)
    
    # Смешиваем оригинал с окрашенной версией для контроля интенсивности
    s = max(0.0, min(1.0, float(strength)))
    toned = Image.blend(rgb, multiplied, alpha=s)
    
    out = Image.merge("RGBA", (*toned.split(), a_ch))
    return out


def _apply_stamp_defect(
    img: Image.Image,
    intensity_range: tuple[float, float] = (0.08, 0.30),
    ink_loss_range: tuple[float, float] = (0.05, 0.20),
) -> Image.Image:
    """Накладывает зернистый дефект, имитируя неполное проштамповывание."""
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.array(img).astype(np.float32)
    rgb = arr[..., :3]
    alpha = arr[..., 3] / 255.0

    h, w = alpha.shape
    noise = np.random.rand(h, w).astype(np.float32)

    # Добавляем несколько раундов блюра, чтобы получить зернистость
    blur_passes = random.randint(1, 3)
    for _ in range(blur_passes):
        sigma = random.uniform(0.6, 1.8)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma)

    threshold = random.uniform(0.4, 0.7)
    defect_map = (noise > threshold).astype(np.float32)

    intensity = random.uniform(*intensity_range)
    ink_loss = random.uniform(*ink_loss_range)

    alpha = np.clip(alpha * (1.0 - defect_map * intensity), 0.0, 1.0)
    rgb = np.clip(rgb * (1.0 - defect_map[..., None] * ink_loss), 0.0, 255.0)

    arr[..., :3] = rgb
    arr[..., 3] = (alpha * 255.0).astype(np.uint8)
    out = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
    return out


def _rect_overlap_ratio(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Возвращает долю пересечения прямоугольников относительно площади меньшего прямоугольника."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    area1 = max(0, w1) * max(0, h1)
    area2 = max(0, w2) * max(0, h2)
    if area1 == 0 or area2 == 0:
        return 0.0

    left = max(x1, x2)
    right = min(x1 + w1, x2 + w2)
    top = max(y1, y2)
    bottom = min(y1 + h1, y2 + h2)

    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    min_area = min(area1, area2)
    if min_area <= 0:
        return 0.0
    return intersection / float(min_area)


def _generate_for_template(
    template_idx: int,
    template_cfg: dict[str, Any],
    seal_files: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    total_saved: int,
) -> tuple[int, int, int, int]:
    """Генерирует изображения печатей для одного шаблона."""

    label = template_cfg.get("label", f"template_{template_idx:02d}")
    base_image_path = Path(template_cfg["base_image"]).resolve()

    if not base_image_path.exists():
        print(f"[{label}] Пропуск: базовое изображение не найдено: {base_image_path}")
        return total_saved, 0, 0

    x_min = int(template_cfg["x_min"])
    x_max = int(template_cfg["x_max"])
    y_min = int(template_cfg["y_min"])
    y_max = int(template_cfg["y_max"])

    print(f"\n=== Шаблон {label}: {base_image_path}")
    print(f"    Область размещения: x[{x_min},{x_max}], y[{y_min},{y_max}]")

    base_image = Image.open(str(base_image_path)).convert("RGBA")

    area_width = x_max - x_min
    area_height = y_max - y_min
    if area_width <= 0 or area_height <= 0:
        print(f"[{label}] Ошибка: неверные границы области (ширина/высота <= 0)")
        return total_saved, 0, 0

    min_width = max(1, int(area_width * 0.4))
    min_height = max(1, int(area_height * 0.4))
    max_width = area_width
    max_height = area_height

    singles_saved = 0

    for seal_path in seal_files:
        try:
            seal_image = Image.open(str(seal_path)).convert("RGBA")

            prepared_seal = prepare_seal(
                seal_image=seal_image,
                min_width=min_width,
                min_height=min_height,
                max_width=max_width,
                max_height=max_height,
                remove_bg=args.remove_bg,
                binarize=args.binarize,
                tolerance=args.tolerance,
                soften=args.soften,
            )

            max_x = x_max - prepared_seal.width
            max_y = y_max - prepared_seal.height

            if max_x < x_min or max_y < y_min:
                continue

            x = random.randint(x_min, max_x)
            y = random.randint(y_min, max_y)

            result = apply_seal_to_image(
                base_image=base_image,
                seal_image=prepared_seal,
                position=(x, y),
            )

            total_saved += 1
            singles_saved += 1
            output_filename = f"seal_{total_saved:03d}.png"
            output_path = output_dir / output_filename
            result.save(str(output_path))
            print(f"[{label}] Сохранено: {output_path}")

        except Exception as exc:
            print(f"[{label}] Ошибка при обработке {seal_path.name}: {exc}")

    duo_target = max(0, int(args.duo_count))
    duo_created = 0
    duo_attempts = 0
    max_total_attempts = 50

    if duo_target > 0:
        print(f"[{label}] Генерация {duo_target} изображений с двумя печатями...")

    while duo_created < duo_target and duo_attempts < max_total_attempts:
        duo_attempts += 1
        try:
            duo_canvas = base_image.copy()
            placements: list[Tuple[int, int, int, int]] = []

            for _ in range(2):
                placed = False
                for _ in range(40):
                    seal_path = random.choice(seal_files)
                    seal_image = Image.open(str(seal_path)).convert("RGBA")

                    prepared_seal = prepare_seal(
                        seal_image=seal_image,
                        min_width=min_width,
                        min_height=min_height,
                        max_width=max_width,
                        max_height=max_height,
                        remove_bg=args.remove_bg,
                        binarize=args.binarize,
                        tolerance=args.tolerance,
                        soften=args.soften,
                    )

                    max_x = x_max - prepared_seal.width
                    max_y = y_max - prepared_seal.height
                    if max_x < x_min or max_y < y_min:
                        continue

                    x = random.randint(x_min, max_x)
                    y = random.randint(y_min, max_y)
                    candidate_box = (x, y, prepared_seal.width, prepared_seal.height)

                    overlap_ok = all(_rect_overlap_ratio(candidate_box, box) <= 0.35 for box in placements)
                    if not overlap_ok:
                        continue

                    duo_canvas = apply_seal_to_image(
                        base_image=duo_canvas,
                        seal_image=prepared_seal,
                        position=(x, y),
                    )
                    placements.append(candidate_box)
                    placed = True
                    break

                if not placed:
                    break

            if len(placements) == 2:
                duo_created += 1
                total_saved += 1
                duo_filename = f"seal_{total_saved:03d}.png"
                duo_path = output_dir / duo_filename
                duo_canvas.save(str(duo_path))
                print(f"[{label}] Сохранено (2 печати): {duo_path}")

        except Exception as exc:
            print(f"[{label}] Ошибка при создании двойного изображения: {exc}")

    multi_layouts = template_cfg.get("multi_layouts")
    multi_created = 0
    if multi_layouts:
        multi_target = int(template_cfg.get("multi_target", args.multi_count))
        multi_overlap_limit = float(template_cfg.get("multi_overlap", 0.35))
        multi_attempts = 0
        max_multi_attempts = max(50, multi_target * 10)
        print(f"[{label}] Генерация {multi_target} изображений с {len(multi_layouts)} зонами (3-4 печати)...")

        while multi_created < multi_target and multi_attempts < max_multi_attempts:
            multi_attempts += 1
            try:
                multi_canvas = base_image.copy()
                placements: list[Tuple[int, int, int, int]] = []
                success = True

                for layout in multi_layouts:
                    region_x_min = int(layout["x_min"])
                    region_x_max = int(layout["x_max"])
                    region_y_min = int(layout["y_min"])
                    region_y_max = int(layout["y_max"])
                    region_w = region_x_max - region_x_min
                    region_h = region_y_max - region_y_min
                    if region_w <= 0 or region_h <= 0:
                        success = False
                        break

                    min_count = int(layout.get("min_count", 1))
                    max_count = int(layout.get("max_count", min_count))
                    desired_count = random.randint(min_count, max_count)

                    for _ in range(desired_count):
                        placed = False
                        for _ in range(60):
                            seal_path = random.choice(seal_files)
                            seal_image = Image.open(str(seal_path)).convert("RGBA")
                            prepared_seal = prepare_seal(
                                seal_image=seal_image,
                                min_width=max(1, int(region_w * 0.3)),
                                min_height=max(1, int(region_h * 0.3)),
                                max_width=region_w,
                                max_height=region_h,
                                remove_bg=args.remove_bg,
                                binarize=args.binarize,
                                tolerance=args.tolerance,
                                soften=args.soften,
                            )

                            max_x = region_x_max - prepared_seal.width
                            max_y = region_y_max - prepared_seal.height
                            if max_x < region_x_min or max_y < region_y_min:
                                continue

                            x = random.randint(region_x_min, max_x)
                            y = random.randint(region_y_min, max_y)
                            candidate_box = (x, y, prepared_seal.width, prepared_seal.height)

                            overlap_ok = all(
                                _rect_overlap_ratio(candidate_box, existing_box) <= multi_overlap_limit
                                for existing_box in placements
                            )
                            if not overlap_ok:
                                continue

                            multi_canvas = apply_seal_to_image(
                                base_image=multi_canvas,
                                seal_image=prepared_seal,
                                position=(x, y),
                            )
                            placements.append(candidate_box)
                            placed = True
                            break

                        if not placed:
                            success = False
                            break

                    if not success:
                        break

                if success and len(placements) >= sum(int(layout.get("min_count", 1)) for layout in multi_layouts):
                    multi_created += 1
                    total_saved += 1
                    multi_filename = f"seal_{total_saved:03d}.png"
                    multi_path = output_dir / multi_filename
                    multi_canvas.save(str(multi_path))
                    print(f"[{label}] Сохранено (3+ печати): {multi_path}")

            except Exception as exc:
                print(f"[{label}] Ошибка при создании многоштампового изображения: {exc}")

    return total_saved, singles_saved, duo_created, multi_created


def prepare_seal(
    seal_image: Image.Image,
    min_width: int,
    min_height: int,
    max_width: int,
    max_height: int,
    remove_bg: bool = True,
    binarize: bool = True,
    tolerance: int = 15,
    soften: float = 1.5,
) -> Image.Image:
    """
    Подготавливает печать: удаляет фон и масштабирует до случайного размера в заданном диапазоне.
    
    Args:
        seal_image: Изображение печати
        min_width: Минимальная ширина
        min_height: Минимальная высота
        max_width: Максимальная ширина
        max_height: Максимальная высота
        remove_bg: Удалять ли фон
        binarize: Использовать ли бинаризацию Otsu
        tolerance: Допуск для удаления белого фона
        soften: Смягчение краев
    """
    # Конвертируем в RGBA
    seal = seal_image.convert("RGBA")
    
    # Удаляем фон печати, если нужно
    if remove_bg:
        if binarize:
            seal = _apply_otsu_alpha(
                seal,
                soften=soften,
                open_kernel=3,
                dilate_iter=0,
            )
        else:
            seal = _remove_white_background_rgba(
                seal,
                tolerance=tolerance,
                soften=soften,
            )
    
    # Применяем случайную цветовую коррекцию для реалистичного вида печати
    # Сила эффекта случайная от 0.6 до 0.85 для разнообразия
    color_strength = random.uniform(0.6, 0.85)
    seal = _apply_random_seal_color(seal, strength=color_strength)
    
    # Накладываем зернистый дефект печати, чтобы имитировать неполное проштамповывание
    seal = _apply_stamp_defect(seal)

    # Сначала масштабируем, чтобы печать поместилась в максимальный размер
    if seal.width > max_width or seal.height > max_height:
        scale_w = max_width / seal.width if seal.width > 0 else 1.0
        scale_h = max_height / seal.height if seal.height > 0 else 1.0
        scale = min(scale_w, scale_h)
        new_size = (max(1, int(round(seal.width * scale))), max(1, int(round(seal.height * scale))))
        seal = seal.resize(new_size, Image.LANCZOS)
    
    # Теперь выбираем случайный размер в диапазоне [min, max], сохраняя пропорции
    current_width = seal.width
    current_height = seal.height
    
    # Вычисляем возможные масштабы для случайного размера
    scale_min_w = min_width / current_width if current_width > 0 else 1.0
    scale_min_h = min_height / current_height if current_height > 0 else 1.0
    scale_min = max(scale_min_w, scale_min_h)  # Берем больший, чтобы вписаться в минимум
    
    scale_max_w = max_width / current_width if current_width > 0 else 1.0
    scale_max_h = max_height / current_height if current_height > 0 else 1.0
    scale_max = min(scale_max_w, scale_max_h)  # Берем меньший, чтобы не выйти за максимум
    
    # Ограничиваем scale_min, чтобы не был больше scale_max
    scale_min = min(scale_min, scale_max)
    
    # Выбираем случайный масштаб
    if scale_min < scale_max:
        random_scale = random.uniform(scale_min, scale_max)
    else:
        random_scale = scale_min
    
    # Применяем случайный масштаб
    new_width = max(1, int(round(current_width * random_scale)))
    new_height = max(1, int(round(current_height * random_scale)))
    seal = seal.resize((new_width, new_height), Image.LANCZOS)
    
    # Применяем случайный поворот от -10 до +10 градусов
    rotation_angle = random.uniform(-10.0, 10.0)
    if abs(rotation_angle) > 0.01:  # Поворачиваем только если угол значимый
        seal = seal.rotate(rotation_angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))
    
    # После поворота размер может увеличиться, проверяем и масштабируем если нужно
    # Но учитываем, что мы хотим размер в диапазоне [min, max]
    # Если после поворота превышает max - масштабируем до max
    if seal.width > max_width or seal.height > max_height:
        scale_w = max_width / seal.width if seal.width > 0 else 1.0
        scale_h = max_height / seal.height if seal.height > 0 else 1.0
        scale = min(scale_w, scale_h)
        new_size = (max(1, int(round(seal.width * scale))), max(1, int(round(seal.height * scale))))
        seal = seal.resize(new_size, Image.LANCZOS)
    
    return seal


def apply_seal_to_image(
    base_image: Image.Image,
    seal_image: Image.Image,
    position: tuple[int, int],
) -> Image.Image:
    """
    Накладывает подготовленную печать на базовое изображение в указанной позиции.
    
    Args:
        base_image: Базовое изображение
        seal_image: Подготовленное изображение печати (уже обработанное и масштабированное)
        position: Позиция (x, y) для размещения печати
    """
    # Конвертируем в RGBA
    base = base_image.convert("RGBA")
    seal = seal_image.convert("RGBA")
    
    # Создаем новый слой для печати
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    
    # Размещаем печать
    x, y = position
    overlay.paste(seal, (x, y), seal)
    
    # Композитируем с базовым изображением
    result = Image.alpha_composite(base, overlay)
    return result


def main():
    root = Path(__file__).resolve().parent
    
    parser = argparse.ArgumentParser(
        description="Накладывает печати из папки seal на изображение 24_25.png в случайных местах"
    )
    parser.add_argument(
        "--base-image",
        type=Path,
        default=root / "data" / "24_25.png",
        help="Путь к базовому изображению (по умолчанию: data/24_25.png)",
    )
    parser.add_argument(
        "--seal-dir",
        type=Path,
        default=root / "data" / "seal",
        help="Путь к папке с печатями (по умолчанию: data/seal)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "out_seal",
        help="Папка для сохранения результатов (по умолчанию: data/out_seal)",
    )
    parser.add_argument(
        "--x-min",
        type=int,
        default=600,
        help="Минимальная координата X (по умолчанию: 600)",
    )
    parser.add_argument(
        "--x-max",
        type=int,
        default=1030,
        help="Максимальная координата X (по умолчанию: 1030)",
    )
    parser.add_argument(
        "--y-min",
        type=int,
        default=180,
        help="Минимальная координата Y (по умолчанию: 180)",
    )
    parser.add_argument(
        "--y-max",
        type=int,
        default=730,
        help="Максимальная координата Y (по умолчанию: 730)",
    )
    parser.add_argument(
        "--remove-bg",
        action="store_true",
        default=True,
        help="Удалять белый фон у печатей (по умолчанию: True)",
    )
    parser.add_argument(
        "--no-remove-bg",
        dest="remove_bg",
        action="store_false",
        help="Не удалять белый фон у печатей",
    )
    parser.add_argument(
        "--binarize",
        action="store_true",
        default=True,
        help="Использовать бинаризацию Otsu для удаления фона (по умолчанию: True)",
    )
    parser.add_argument(
        "--no-binarize",
        dest="binarize",
        action="store_false",
        help="Не использовать бинаризацию Otsu",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=15,
        help="Допуск для удаления белого фона (по умолчанию: 15)",
    )
    parser.add_argument(
        "--soften",
        type=float,
        default=1.5,
        help="Смягчение краев (по умолчанию: 1.5)",
    )
    parser.add_argument(
        "--duo-count",
        type=int,
        default=10,
        help="Сколько изображений с двумя печатями генерировать для каждого шаблона (по умолчанию: 10)",
    )
    parser.add_argument(
        "--multi-count",
        type=int,
        default=10,
        help="Сколько изображений с тремя и более печатями генерировать (по умолчанию: 10, только для шаблонов с multi_layouts)",
    )
    parser.add_argument(
        "--skip-secondary-template",
        action="store_true",
        help="Не генерировать для стандартного дополнительного шаблона (26_27.png)",
    )
    
    args = parser.parse_args()
    
    # Разрешаем пути
    base_image_path = Path(args.base_image) if os.path.isabs(str(args.base_image)) else (root / args.base_image).resolve()
    seal_dir = Path(args.seal_dir) if os.path.isabs(str(args.seal_dir)) else (root / args.seal_dir).resolve()
    output_dir = Path(args.output_dir) if os.path.isabs(str(args.output_dir)) else (root / args.output_dir).resolve()
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все печати
    seal_files = _list_image_files_recursive(seal_dir)
    if not seal_files:
        print(f"Ошибка: печати не найдены в директории: {seal_dir}")
        return
    print(f"Найдено печатей: {len(seal_files)}")

    templates: list[dict[str, Any]] = [
        {
            "label": Path(base_image_path).stem,
            "base_image": str(base_image_path),
            "x_min": args.x_min,
            "x_max": args.x_max,
            "y_min": args.y_min,
            "y_max": args.y_max,
        }
    ]

    if not args.skip_secondary_template:
        templates.append(
            {
                "label": "26_27",
                "base_image": str((root / "data" / "26_27.png").resolve()),
                "x_min": 150,
                "x_max": 600,
                "y_min": 150,
                "y_max": 750,
                "multi_layouts": [
                    {
                        "x_min": 150,
                        "x_max": 600,
                        "y_min": 150,
                        "y_max": 750,
                        "min_count": 2,
                        "max_count": 3,
                    },
                    {
                        "x_min": 600,
                        "x_max": 1030,
                        "y_min": 180,
                        "y_max": 730,
                        "min_count": 1,
                        "max_count": 1,
                    },
                ],
                "multi_overlap": 0.35,
                "multi_target": 10,
            }
        )

    total_saved = 0
    summary: list[tuple[str, int, int, int]] = []

    for idx, template_cfg in enumerate(templates, 1):
        total_saved, singles, duo_created, multi_created = _generate_for_template(
            template_idx=idx,
            template_cfg=template_cfg,
            seal_files=seal_files,
            output_dir=output_dir,
            args=args,
            total_saved=total_saved,
        )
        summary.append((template_cfg["label"], singles, duo_created, multi_created))

    print("\nИтоги по шаблонам:")
    for label, singles, duo_created, multi_created in summary:
        extra = f", мульти={multi_created}" if multi_created else ""
        print(f"  {label}: одиночных={singles}, двойных={duo_created}{extra}")
    print(f"Всего изображений сохранено: {total_saved}. Папка: {output_dir}")


if __name__ == "__main__":
    main()

