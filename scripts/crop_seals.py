#!/usr/bin/env python3
"""Обрезка изображений по границам печатей с исправлением перспективы"""

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

def order_points(pts):
    """Сортировка точек по часовой стрелке: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    """Исправление перспективы по 4 точкам"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Вычисляем ширину и высоту
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Точки назначения
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Матрица перспективы
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def approximate_polygon(points, epsilon=0.02):
    """Аппроксимация полигона до 4 точек"""
    if len(points) <= 4:
        return points
    
    # Конвертируем в формат для cv2.approxPolyDP
    points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    
    # Аппроксимируем с epsilon
    perimeter = cv2.arcLength(points_array, True)
    approx = cv2.approxPolyDP(points_array, epsilon * perimeter, True)
    
    # Если получилось больше 4 точек, увеличиваем epsilon
    while len(approx) > 4 and epsilon < 0.5:
        epsilon *= 1.5
        approx = cv2.approxPolyDP(points_array, epsilon * perimeter, True)
    
    # Если все еще больше 4, берем только углы
    if len(approx) > 4:
        # Находим bounding box и берем углы
        rect = cv2.minAreaRect(points_array.reshape(-1, 2))
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2)
    
    return approx.reshape(-1, 2).tolist()

def process_image(image_path, annotations, output_dir):
    """Обработка одного изображения"""
    # Загружаем изображение
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Не удалось загрузить {image_path}")
        return
    
    # Находим все печати для этого изображения
    image_id = None
    for img_info in annotations['images']:
        if img_info['file_name'] == image_path.name:
            image_id = img_info['id']
            break
    
    if image_id is None:
        print(f"⚠️ Изображение {image_path.name} не найдено в аннотациях")
        return
    
    # Получаем все печати для этого изображения
    seals = [ann for ann in annotations['annotations'] 
             if ann['image_id'] == image_id and ann['category_id'] == 2]  # category_id=2 для печатей
    
    if not seals:
        print(f"⚠️ Печати не найдены в {image_path.name}")
        return
    
    # Обрабатываем каждую печать
    for i, seal in enumerate(seals):
        # Получаем точки сегментации
        points = np.array(seal['segmentation'][0]).reshape(-1, 2)
        
        # Аппроксимируем до 4 точек
        if len(points) > 4:
            points = np.array(approximate_polygon(points))
        
        if len(points) < 4:
            print(f"⚠️ Недостаточно точек для печати {i} в {image_path.name}")
            continue
        
        # Обрезаем и исправляем перспективу
        try:
            cropped = four_point_transform(image, points)
            
            # Сохраняем результат
            output_name = f"{image_path.stem}_seal_{i:02d}.jpg"
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), cropped)
            print(f"✓ {output_name}")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке печати {i} в {image_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Обрезка печатей с исправлением перспективы')
    parser.add_argument('--input', default='../dataset/dataset_for_doc_detector/dataset_passport_pages',
                       help='Папка с изображениями')
    parser.add_argument('--annotations', default='../dataset/dataset_for_doc_detector/dataset_passport_pages/instances_default_seal.json',
                       help='Файл с аннотациями')
    parser.add_argument('--output', default='../dataset/dataset_passport_seal_ocr',
                       help='Папка для сохранения результатов')
    
    args = parser.parse_args()
    
    # Создаем выходную папку
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Загружаем аннотации
    with open(args.annotations, 'r') as f:
        annotations = json.load(f)
    
    # Обрабатываем все изображения
    input_dir = Path(args.input)
    image_files = list(input_dir.glob('*.jpg')) + list(input_dir.glob('*.png'))
    
    print(f"🔍 Найдено {len(image_files)} изображений")
    print(f"📁 Выходная папка: {output_dir}")
    
    for image_path in image_files:
        process_image(image_path, annotations, output_dir)
    
    print(f"\n✅ Готово! Результаты сохранены в {output_dir}")

if __name__ == '__main__':
    main()


