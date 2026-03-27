#!/usr/bin/env python3
"""
Скрипт для пересоздания поврежденных изображений через скриншот
Убирает все проблемы с Corrupt JPEG data
"""

import cv2
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

def fix_corrupted_images(input_dir: str, output_dir: str = None):
    """Пересоздает изображения, убирая проблемы с JPEG"""
    
    input_path = Path(input_dir)
    if output_dir is None:
        output_dir = input_dir + "_fixed"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Находим все изображения
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Обрабатываем каждое изображение
    for img_file in tqdm(image_files, desc="Пересоздание изображений"):
        try:
            # Загружаем изображение
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Не удалось загрузить: {img_file}")
                continue
            
            # Сохраняем с тем же именем файла
            output_file = output_path / img_file.name
            
            # Создаем папки если нужно
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Сохраняем как новый JPEG с высоким качеством, сохраняя оригинальное имя
            cv2.imwrite(str(output_file), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
    
    print(f"Готово! Исправленные изображения сохранены в: {output_path}")
    return str(output_path)

def main():
    # Обрабатываем все папки с изображениями
    dataset_dirs = [
        "dataset/dataset_for_doc_detector/yolo_dataset/train/images",
        "dataset/dataset_for_doc_detector/yolo_dataset/val/images", 
        "dataset/dataset_for_doc_detector/yolo_dataset/test/images"
    ]
    
    for dataset_dir in dataset_dirs:
        if Path(dataset_dir).exists():
            print(f"\nОбрабатываем: {dataset_dir}")
            fixed_dir = fix_corrupted_images(dataset_dir)
            
            # Заменяем старые изображения новыми
            print(f"Заменяем старые изображения на исправленные...")
            os.system(f"rm -rf {dataset_dir}")
            os.system(f"mv {fixed_dir} {dataset_dir}")
            print(f"Готово! {dataset_dir} обновлен")

if __name__ == "__main__":
    main()
