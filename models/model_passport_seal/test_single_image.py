#!/usr/bin/env python3
"""
Скрипт для тестирования обученной модели на одном изображении
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def test_single_image():
    # Загружаем обученную модель
    model_path = 'runs/segment/train2/weights/best.pt'
    model = YOLO(model_path)
    
    # Путь к тестовому изображению
    test_image_path = '../../dataset/dataset_for_seal_detector/test/images/passport_pages_034.jpg'
    
    # Загружаем изображение
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Не удалось загрузить изображение: {test_image_path}")
        return
    
    print(f"Тестирую изображение: {test_image_path}")
    print(f"Размер изображения: {img.shape}")
    
    # Инференс
    results = model(img, verbose=False)[0]
    
    # Анализируем результаты
    if hasattr(results, 'boxes') and results.boxes is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy()
        
        print(f"\nНайдено детекций: {len(boxes)}")
        
        for i, (box, conf, cl) in enumerate(zip(boxes, confs, cls)):
            print(f"  Детекция {i+1}:")
            print(f"    Класс: {cl} (seal)")
            print(f"    Уверенность: {conf:.3f}")
            print(f"    Bounding box: {box}")
    
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        print(f"\nНайдено масок: {len(masks)}")
        
        # Визуализируем результаты
        viz_img = img.copy()
        
        for i, mask in enumerate(masks):
            # Преобразуем маску в бинарную
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
            
            # Находим контуры маски
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Рисуем контур
                cv2.drawContours(viz_img, contours, 0, (0, 255, 0), 2)
                
                # Подписываем номер детекции
                if len(boxes) > i:
                    box = boxes[i]
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.putText(viz_img, f"Seal{i+1}", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Сохраняем результат
        output_path = 'test_result.jpg'
        cv2.imwrite(output_path, viz_img)
        print(f"\nРезультат сохранен в: {output_path}")
        
        # Показываем статистику масок
        for i, mask in enumerate(masks):
            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)
            
            # Вычисляем площадь маски
            area = np.sum(mask_bin)
            area_percent = (area / (img.shape[0] * img.shape[1])) * 100
            
            print(f"  Маска {i+1}:")
            print(f"    Площадь: {area} пикселей ({area_percent:.2f}% от изображения)")
            print(f"    Размер маски: {mask.shape}")
    else:
        print("Маски не найдены")

if __name__ == "__main__":
    test_single_image()
