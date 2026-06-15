"""
Определение угла поворота документа без обучения.

Две ступени:
1. Свободный угол (skew) ±45° — Hough lines / minAreaRect по бинаризованному
   изображению. Алгоритмический метод, не требует ML.
2. Квадрант (0/90/180/270) — Tesseract OSD (zero-shot), если установлен.
   Если pytesseract нет — fallback на старый ONNX-классификатор Angle90.

Класс сохраняет имя `Angle90` и интерфейс (`predict`, `predict_transform`,
ключи `angle`, `confidence`, `warped_img` в meta) для совместимости с
существующим pipeline.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from ..base_module import BaseModule

try:
    import pytesseract
    _HAS_TESS = True
except Exception:
    pytesseract = None
    _HAS_TESS = False


def _prep_gray(img: np.ndarray, max_side: int = 1500) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
    return gray


def _detect_skew_minarea(img: np.ndarray, max_abs_deg: float = 30.0) -> Optional[float]:
    """Через minAreaRect: угол верхнего ребра bounding-box текстового блока.

    Возвращает correction angle: знак согласован с `_rotate_image_keep_size`
    (положительный = поворот против часовой, отрицательный = по часовой).
    """
    gray = _prep_gray(img)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 200]
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    edges = []
    for i in range(4):
        p1, p2 = box[i], box[(i + 1) % 4]
        if p2[0] < p1[0]:
            p1, p2 = p2, p1
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        if math.hypot(dx, dy) < 5:
            continue
        a = math.degrees(math.atan2(dy, dx))
        if abs(a) > 45:
            continue
        mid_y = (p1[1] + p2[1]) / 2.0
        edges.append((mid_y, a))
    if not edges:
        return None
    edges.sort(key=lambda x: x[0])
    angle = edges[0][1]
    if abs(angle) > max_abs_deg:
        return None
    return float(angle)


def _detect_skew_hough(img: np.ndarray, max_abs_deg: float = 30.0) -> Optional[float]:
    """Через HoughLinesP: робастная медиана углов горизонтальных линий."""
    gray = _prep_gray(img)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3, L2gradient=True)
    min_line = int(min(edges.shape) * 0.25)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 720, threshold=120,
                            minLineLength=max(40, min_line), maxLineGap=15)
    if lines is None:
        return None
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = float(x2 - x1), float(y2 - y1)
        if math.hypot(dx, dy) < 5:
            continue
        a = math.degrees(math.atan2(dy, dx))
        if abs(a) <= max_abs_deg:
            angles.append(a)
    if not angles:
        return None
    arr = np.array(angles, dtype=np.float32)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) or 0.5
    keep = arr[np.abs(arr - med) <= 3 * mad]
    return float(np.median(keep)) if keep.size else med


def _projection_score(binary: np.ndarray, angle: float) -> float:
    """Дисперсия 1-й производной горизонтальной проекции — мера ровности строк.

    Чем строки текста точнее параллельны горизонтали, тем сильнее «полосатая»
    структура горизонтальной проекции — у её diff больше дисперсия.
    """
    rotated = _rotate_image_keep_size(binary, angle)
    proj = np.sum(rotated, axis=1, dtype=np.float64)
    if proj.size < 4:
        return 0.0
    return float(np.var(np.diff(proj)))


def _detect_skew_projection(img: np.ndarray,
                            coarse_range: Tuple[float, float] = (-15.0, 15.0),
                            coarse_step: float = 1.0,
                            fine_window: float = 1.5,
                            fine_step: float = 0.1) -> Tuple[float, float]:
    """Skew через projection profile (2 этапа: грубо + уточнение)."""
    gray = _prep_gray(img, max_side=700)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)),
                          iterations=1)

    angles = np.arange(coarse_range[0], coarse_range[1] + coarse_step, coarse_step)
    scores = np.array([_projection_score(th, a) for a in angles])
    if scores.max() <= 0:
        return 0.0, 0.0
    best_idx = int(scores.argmax())
    best_angle = float(angles[best_idx])

    fine_angles = np.arange(best_angle - fine_window,
                            best_angle + fine_window + fine_step, fine_step)
    fine_scores = np.array([_projection_score(th, a) for a in fine_angles])
    best_idx = int(fine_scores.argmax())
    final_angle = float(fine_angles[best_idx])

    s_sorted = np.sort(scores)[::-1]
    margin = (s_sorted[0] - s_sorted[1]) / s_sorted[0] if len(s_sorted) >= 2 and s_sorted[0] > 0 else 0.0
    confidence = float(np.clip(margin * 5 + 0.4, 0.0, 1.0))
    return final_angle, confidence


def _detect_skew(img: np.ndarray, max_abs_deg: float = 30.0) -> Tuple[float, float]:
    """Главный детектор: projection profile + Hough как sanity check."""
    angle, conf = _detect_skew_projection(img,
                                          coarse_range=(-max_abs_deg, max_abs_deg))
    hough = _detect_skew_hough(img, max_abs_deg=max_abs_deg)
    if hough is not None and abs(angle - hough) < 2.0:
        conf = float(min(1.0, conf + 0.2))
    return angle, conf


def _rotate_image_keep_size(img: np.ndarray, angle: float) -> np.ndarray:
    """Поворот вокруг центра с расширением канвы (без обрезки)."""
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _tesseract_quadrant(img: np.ndarray) -> Optional[Tuple[int, float]]:
    """Tesseract OSD → (нужный поворот в градусах из {0,90,180,270}, conf)."""
    if not _HAS_TESS:
        return None
    try:
        if img.ndim == 3:
            inp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            inp = img
        osd = pytesseract.image_to_osd(inp, config="--psm 0")
        m = re.search(r"Rotate:\s*(\d+)", osd)
        c = re.search(r"Orientation confidence:\s*([\d.]+)", osd)
        if not m:
            return None
        rotate = int(m.group(1)) % 360
        conf = float(c.group(1)) if c else 1.0
        return rotate, min(conf / 5.0, 1.0)
    except Exception:
        return None


def _apply_quadrant(img: np.ndarray, rotate_deg: int) -> np.ndarray:
    """rotate_deg ∈ {0, 90, 180, 270} — поворот по часовой стрелке."""
    if rotate_deg == 0:
        return img
    if rotate_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotate_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class Angle90(BaseModule):
    """Алгоритмическое определение угла + поворот документа.

    Совместим со старым `Angle90` по интерфейсу. Сначала находит skew ±45°
    (Hough + minAreaRect, без обучения), затем при необходимости квадрант
    через Tesseract OSD (zero-shot). Если pytesseract отсутствует — старая
    ONNX-модель используется только для определения квадранта.
    """

    def __init__(self, model_format: str = "ONNX", device: str = "cpu",
                 verbose: bool = False, use_skew: bool = True,
                 use_quadrant: bool = True, quadrant_conf_threshold: float = 0.6):
        self.model_name = "Angle90"
        super().__init__(self.model_name, model_format=model_format,
                         device=device, verbose=verbose)
        self.use_skew = use_skew
        self.use_quadrant = use_quadrant
        self.quadrant_conf_threshold = quadrant_conf_threshold

    def _quadrant_classifier(self, img: np.ndarray) -> Tuple[int, float]:
        """Старая ONNX-модель — даёт класс из {0,90,180,270}."""
        tensor = self.model.preprocessing(img)
        tensor = tensor.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        logits = self.model.inference_model.predict(tensor)[0]
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        label = self.model_info["Labels"][int(probs.argmax())]
        return int(label), float(probs.max())

    def _detect_full_angle(self, img: np.ndarray) -> Tuple[float, float, dict]:
        """Возвращает (total_deg, confidence, debug_info)."""
        debug: dict = {}
        rotated = img
        total = 0.0

        if self.use_skew:
            skew_deg, skew_conf = _detect_skew(rotated, max_abs_deg=30.0)
            debug["skew"] = {"angle": skew_deg, "confidence": skew_conf}
            if abs(skew_deg) >= 0.3:
                rotated = _rotate_image_keep_size(rotated, skew_deg)
                total += skew_deg

        quadrant_conf = 1.0
        if self.use_quadrant:
            tess = _tesseract_quadrant(rotated)
            if tess is not None:
                q_rotate, q_conf = tess
                source = "tesseract"
            else:
                q_label, q_conf = self._quadrant_classifier(rotated)
                q_rotate = int(q_label)
                source = "Angle90 ONNX"
            applied = q_rotate if q_conf >= self.quadrant_conf_threshold else 0
            debug["quadrant"] = {"source": source, "rotate": q_rotate,
                                 "applied": applied, "confidence": q_conf}
            if applied:
                rotated = _apply_quadrant(rotated, applied)
                total += applied
            quadrant_conf = q_conf

        skew_conf = debug.get("skew", {}).get("confidence", 1.0)
        confidence = float(min(skew_conf, quadrant_conf))
        total = ((total + 180) % 360) - 180
        debug["rotated_img"] = rotated
        return total, confidence, debug

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        loaded = self.load_img(img)
        angle, conf, debug = self._detect_full_angle(loaded)
        return {
            self.model_name: {
                "angle": angle,
                "confidence": conf,
                "skew_deg": debug.get("skew", {}).get("angle", 0.0),
                "quadrant_deg": debug.get("quadrant", {}).get("rotate", 0),
                "quadrant_source": debug.get("quadrant", {}).get("source", "skip"),
            }
        }

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        loaded = self.load_img(img)
        angle, conf, debug = self._detect_full_angle(loaded)
        return {
            self.model_name: {
                "angle": angle,
                "confidence": conf,
                "warped_img": debug["rotated_img"],
                "skew_deg": debug.get("skew", {}).get("angle", 0.0),
                "quadrant_deg": debug.get("quadrant", {}).get("rotate", 0),
                "quadrant_source": debug.get("quadrant", {}).get("source", "skip"),
            }
        }
