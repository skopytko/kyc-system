import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional

def determine_page_side(seal_center: np.ndarray, img_shape: Tuple[int, int]) -> str:
    """Определяет, на какой стороне страницы находится печать.
    
    Args:
        seal_center: Центр печати (x, y)
        img_shape: Размеры изображения (height, width)
        
    Returns:
        'left' или 'right' в зависимости от расположения
    """
    h, w = img_shape[:2]
    center_x = seal_center[0]
    
    # Если центр печати находится в левой половине изображения
    if center_x < w / 2:
        return 'left'
    else:
        return 'right'

def calculate_seal_priority(seal_center: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """Вычисляет приоритет печати для определения "последней".
    
    Логика: левая страница → вниз, затем правая страница → вниз
    Чем меньше приоритет, тем "последнее" печать
    
    Args:
        seal_center: Центр печати (x, y)
        img_shape: Размеры изображения (height, width)
        
    Returns:
        Приоритет печати (меньше = последнее)
    """
    h, w = img_shape[:2]
    x, y = seal_center
    
    # Определяем сторону страницы
    side = determine_page_side(seal_center, img_shape)
    
    # Нормализуем координаты (0-1)
    norm_x = x / w
    norm_y = y / h
    
    if side == 'left':
        # Левая страница: приоритет = Y координата (сверху вниз)
        # Добавляем небольшой сдвиг, чтобы левая страница была "раньше"
        priority = norm_y
    else:
        # Правая страница: приоритет = Y координата + 1 (чтобы была "позже" левой)
        priority = norm_y + 1.0
    
    return priority

def process_seals(seals: List[Dict], img_shape: Tuple[int, int]) -> List[Dict]:
    """Обрабатывает список печатей, добавляя приоритеты и сортируя их.
    
    Args:
        seals: Список обнаруженных печатей
        img_shape: Размеры изображения
        
    Returns:
        Обработанный список печатей с приоритетами
    """
    processed_seals = []
    
    for seal in seals:
        # Добавляем приоритет
        seal_copy = seal.copy()
        seal_copy['priority'] = calculate_seal_priority(seal['center'], img_shape)
        processed_seals.append(seal_copy)
    
    # Сортируем по приоритету (последняя печать будет первой)
    processed_seals.sort(key=lambda x: x['priority'], reverse=True)
    
    return processed_seals

def select_last_seal(processed_seals: List[Dict]) -> Optional[Dict]:
    """Выбирает самую последнюю печать из обработанного списка.
    
    Args:
        processed_seals: Обработанный список печатей
        
    Returns:
        Последняя печать или None если печатей нет
    """
    if not processed_seals:
        return None
    
    # Возвращаем печать с наивысшим приоритетом (первую в отсортированном списке)
    return processed_seals[0]

def approximate_seal_to_quad(mask: np.ndarray) -> Optional[np.ndarray]:
    """Аппроксимирует маску печати до 4-точечного контура.
    
    Args:
        mask: Бинарная маска печати
        
    Returns:
        4 точки контура или None если не удалось аппроксимировать
    """
    # Находим контуры в маске
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Берем самый большой контур
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Аппроксимируем до многоугольника
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    # Если получилось 4 точки, возвращаем их
    if len(approx) == 4:
        return approx.reshape(-1, 2)
    
    # Если точек больше или меньше, пытаемся получить 4 точки
    if len(approx) > 4:
        # Берем 4 точки с максимальным расстоянием между ними
        points = approx.reshape(-1, 2)
        # Простая эвристика: берем угловые точки
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        return box.astype(np.int32)
    else:
        # Если точек меньше 4, расширяем контур
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.int32)

def validate_seal_quad(quad: np.ndarray) -> bool:
    """Проверяет валидность 4-точечного контура печати.
    
    Args:
        quad: 4 точки контура
        
    Returns:
        True если контур валиден
    """
    if quad is None or len(quad) != 4:
        return False
    
    # Проверяем, что все точки разные
    unique_points = np.unique(quad, axis=0)
    if len(unique_points) < 4:
        return False
    
    # Проверяем, что контур не вырожден (площадь > 0)
    area = cv2.contourArea(quad.reshape(-1, 1, 2))
    if area <= 0:
        return False
    
    return True

def order_seal_points(pts: np.ndarray) -> np.ndarray:
    """Упорядочивает точки контура печати по часовой стрелке.
    
    Args:
        pts: 4 точки контура
        
    Returns:
        Упорядоченные точки
    """
    if pts is None or len(pts) != 4:
        return pts
    
    # Находим центр
    center = pts.mean(axis=0)
    
    # Вычисляем углы от центра
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Сортируем по углам
    indices = np.argsort(angles)
    
    return pts[indices]
