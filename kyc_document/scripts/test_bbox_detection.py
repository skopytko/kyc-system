"""
Скрипт для тестирования точности детекции bounding boxes текстовых полей.

Сравнивает результаты детектора текстовых полей с ground truth разметкой
и вычисляет метрики: IoU, Precision, Recall, F1-score по классам и в целом.

Поддерживает форматы разметки:
- YOLO формат (.txt файлы)
- JSON формат с bbox координатами

Пример использования:
    python test_bbox_detection.py --verbose
    
    python test_bbox_detection.py --images_dir /path/to/images --labels_dir /path/to/labels
    
    python test_bbox_detection.py --limit 10 --iou_threshold 0.5 --format ONNX

Формат разметки YOLO (.txt):
    class_id x_center y_center width height
    (координаты нормализованы относительно размера изображения)

Формат разметки JSON:
    {
      "image_id": "image.jpg",
      "bbox_annotations": [
        {
          "class": "Last_name_ru",
          "bbox": [x1, y1, x2, y2]  или [x, y, width, height]
        }
      ]
    }
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
import cv2

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent.parent.parent))

from kyc_document.document_processing.pipeline_modules.textfields_detector.russia.passport.textfields_detector import TextFieldsDetector


# Имена классов из модели (из model.json)
CLASS_NAMES = [
    "Face",
    "Last_name_ru",
    "First_name_ru",
    "Last_name_en",
    "First_name_en",
    "Licence_number",
    "Issue_date",
    "Expiration_date",
    "Driver_class",
    "Signature",
    "Birth_date",
    "Birth_place_ru",
    "Birth_place_en",
    "Issue_organization_ru",
    "Issue_organization_en",
    "Living_region_ru",
    "Living_region_en",
    "Middle_name_ru",
    "Sex_ru",
    "Sex_en",
    "Issue_organisation_code",
    "Middle_name_en"
]


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Вычисляет Intersection over Union (IoU) между двумя bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value между 0 и 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Вычисляем пересечение
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Вычисляем объединение
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_yolo_boxes(label_path: Path, img_shape: Tuple[int, int]) -> List[Dict]:
    """
    Загружает YOLO bounding boxes из файла.
    
    Args:
        label_path: Путь к файлу с разметкой в формате YOLO
        img_shape: (height, width) изображения
        
    Returns:
        Список словарей с ключами 'class', 'bbox' [x1, y1, x2, y2]
    """
    h, w = img_shape[:2]
    boxes = []
    
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                
                # Конвертируем из формата (center_x, center_y, width, height) в (x1, y1, x2, y2)
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                boxes.append({
                    'class': class_id,
                    'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f'class_{class_id}',
                    'bbox': [x1, y1, x2, y2]
                })
            except (ValueError, IndexError) as e:
                continue
    
    return boxes


def load_json_boxes(annotation_path: Path, img_shape: Tuple[int, int]) -> List[Dict]:
    """
    Загружает bounding boxes из JSON файла.
    
    Ожидаемый формат JSON:
    {
      "image_id": "image.jpg",
      "bbox_annotations": [
        {
          "class": "Last_name_ru",
          "bbox": [x1, y1, x2, y2]  или [x, y, width, height]
        },
        ...
      ]
    }
    
    Args:
        annotation_path: Путь к JSON файлу с разметкой
        img_shape: (height, width) изображения
        
    Returns:
        Список словарей с ключами 'class', 'class_name', 'bbox' [x1, y1, x2, y2]
    """
    boxes = []
    
    if not annotation_path.exists():
        return boxes
    
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        annotations = data.get('bbox_annotations', [])
        if not annotations:
            # Пробуем альтернативный формат
            annotations = data.get('annotations', [])
        
        for ann in annotations:
            class_name = ann.get('class', '')
            if not class_name:
                continue
            
            # Находим индекс класса
            class_id = -1
            if class_name in CLASS_NAMES:
                class_id = CLASS_NAMES.index(class_name)
            
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                # Проверяем формат: [x1, y1, x2, y2] или [x, y, width, height]
                # Если width и height меньше размеров изображения, это вероятно [x, y, w, h]
                if bbox[2] < img_shape[1] and bbox[3] < img_shape[0] and bbox[2] > 0 and bbox[3] > 0:
                    # Формат [x, y, width, height]
                    x1, y1, w, h = bbox
                    x2 = x1 + w
                    y2 = y1 + h
                else:
                    # Формат [x1, y1, x2, y2]
                    x1, y1, x2, y2 = bbox
                
                boxes.append({
                    'class': class_id,
                    'class_name': class_name,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })
    except Exception as e:
        print(f"Ошибка при загрузке JSON разметки {annotation_path}: {e}")
    
    return boxes


def match_boxes(
    pred_boxes: List[Dict],
    gt_boxes: List[Dict],
    iou_threshold: float = 0.5,
    match_by_class: bool = True
) -> Tuple[List[float], List[bool], List[int], List[int]]:
    """
    Сопоставляет предсказанные bbox с ground truth по максимальному IoU.
    
    Args:
        pred_boxes: Список предсказанных bbox с ключами 'bbox', 'class', 'class_name'
        gt_boxes: Список ground truth bbox с ключами 'bbox', 'class', 'class_name'
        iou_threshold: Минимальный IoU для совпадения
        match_by_class: Если True, сопоставляет только bbox одного класса
        
    Returns:
        (matched_ious, matched_flags, matched_gt_indices, matched_pred_indices)
    """
    matched_gt = [False] * len(gt_boxes)
    matched_pred = [False] * len(pred_boxes)
    matched_ious = []
    matched_gt_indices = []
    matched_pred_indices = []
    
    # Создаем список всех возможных пар (pred_idx, gt_idx, iou)
    pairs = []
    for pred_idx, pred_box in enumerate(pred_boxes):
        for gt_idx, gt_box in enumerate(gt_boxes):
            if match_by_class:
                # Сопоставляем только если класс совпадает
                pred_class = pred_box.get('class', -1)
                gt_class = gt_box.get('class', -1)
                if pred_class != gt_class and pred_class != -1 and gt_class != -1:
                    # Также проверяем по имени класса
                    pred_class_name = pred_box.get('class_name', '')
                    gt_class_name = gt_box.get('class_name', '')
                    if pred_class_name != gt_class_name:
                        continue
            
            iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
            if iou >= iou_threshold:
                pairs.append((pred_idx, gt_idx, iou))
    
    # Сортируем по IoU (от большего к меньшему)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Жадное сопоставление
    for pred_idx, gt_idx, iou in pairs:
        if not matched_gt[gt_idx] and not matched_pred[pred_idx]:
            matched_gt[gt_idx] = True
            matched_pred[pred_idx] = True
            matched_ious.append(iou)
            matched_gt_indices.append(gt_idx)
            matched_pred_indices.append(pred_idx)
    
    return matched_ious, matched_gt, matched_gt_indices, matched_pred_indices


def process_image(
    image_path: Path,
    annotation_path: Path,
    detector: TextFieldsDetector,
    annotation_format: str = 'auto',
    iou_threshold: float = 0.5,
    verbose: bool = False
) -> Dict:
    """
    Обрабатывает одно изображение и сравнивает детекцию с разметкой.
    
    Returns:
        Словарь с результатами сравнения
    """
    image_name = image_path.name
    
    # Загружаем изображение
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {
                'image_id': image_name,
                'error': 'Failed to load image',
                'fields': {}
            }
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
    except Exception as e:
        return {
            'image_id': image_name,
            'error': f'Failed to load image: {e}',
            'fields': {}
        }
    
    # Загружаем разметку
    if annotation_format == 'auto':
        # Определяем формат по расширению
        if annotation_path.suffix == '.txt':
            annotation_format = 'yolo'
        elif annotation_path.suffix == '.json':
            annotation_format = 'json'
        else:
            return {
                'image_id': image_name,
                'error': f'Unknown annotation format: {annotation_path.suffix}',
                'fields': {}
            }
    
    if annotation_format == 'yolo':
        gt_boxes = load_yolo_boxes(annotation_path, (h, w))
    elif annotation_format == 'json':
        gt_boxes = load_json_boxes(annotation_path, (h, w))
    else:
        return {
            'image_id': image_name,
            'error': f'Unsupported annotation format: {annotation_format}',
            'fields': {}
        }
    
    if not gt_boxes:
        return {
            'image_id': image_name,
            'error': 'No ground truth boxes found',
            'fields': {}
        }
    
    # Запускаем детектор
    try:
        result = detector.predict(str(image_path))
        pred_boxes_raw = result.get('TextFieldsDetectorRussia', {}).get('bbox', [])
        
        # Конвертируем предсказанные bbox в нужный формат
        pred_boxes = []
        for box in pred_boxes_raw:
            if len(box) >= 4:
                # Формат: [x1, y1, x2, y2, confidence, class_id, class_name, ...]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                conf = float(box[4]) if len(box) > 4 else 1.0
                class_id = int(box[5]) if len(box) > 5 else -1
                class_name = box[6] if len(box) > 6 else (CLASS_NAMES[class_id] if class_id >= 0 and class_id < len(CLASS_NAMES) else 'unknown')
                
                pred_boxes.append({
                    'bbox': [x1, y1, x2, y2],
                    'class': class_id,
                    'class_name': class_name,
                    'confidence': conf
                })
    except Exception as e:
        if verbose:
            print(f"Ошибка при обработке {image_name}: {e}")
        return {
            'image_id': image_name,
            'error': str(e),
            'fields': {}
        }
    
    # Сопоставляем bbox
    matched_ious, matched_gt_flags, matched_gt_indices, matched_pred_indices = match_boxes(
        pred_boxes, gt_boxes, iou_threshold=iou_threshold, match_by_class=True
    )
    
    # Группируем результаты по классам
    class_results = defaultdict(lambda: {
        'gt_count': 0,
        'pred_count': 0,
        'matched_count': 0,
        'ious': []
    })
    
    # Подсчитываем GT по классам
    for gt_box in gt_boxes:
        class_name = gt_box.get('class_name', 'unknown')
        class_results[class_name]['gt_count'] += 1
    
    # Подсчитываем предсказания по классам
    for pred_box in pred_boxes:
        class_name = pred_box.get('class_name', 'unknown')
        class_results[class_name]['pred_count'] += 1
    
    # Подсчитываем совпадения по классам
    for i, gt_idx in enumerate(matched_gt_indices):
        gt_box = gt_boxes[gt_idx]
        class_name = gt_box.get('class_name', 'unknown')
        class_results[class_name]['matched_count'] += 1
        class_results[class_name]['ious'].append(matched_ious[i])
    
    # Конвертируем в обычный словарь
    class_results_dict = {}
    for class_name, stats in class_results.items():
        precision = stats['matched_count'] / stats['pred_count'] if stats['pred_count'] > 0 else 0.0
        recall = stats['matched_count'] / stats['gt_count'] if stats['gt_count'] > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_iou = np.mean(stats['ious']) if stats['ious'] else 0.0
        
        class_results_dict[class_name] = {
            'gt_count': stats['gt_count'],
            'pred_count': stats['pred_count'],
            'matched_count': stats['matched_count'],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_iou': mean_iou
        }
    
    return {
        'image_id': image_name,
        'error': None,
        'total_gt_boxes': len(gt_boxes),
        'total_pred_boxes': len(pred_boxes),
        'total_matched_boxes': len(matched_ious),
        'mean_iou': np.mean(matched_ious) if matched_ious else 0.0,
        'class_results': class_results_dict
    }


def calculate_metrics(results: List[Dict]) -> Dict:
    """Вычисляет общие метрики по всем результатам."""
    # Агрегируем статистику по всем изображениям
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_matched_boxes = 0
    all_ious = []
    
    # Статистика по классам
    class_stats = defaultdict(lambda: {
        'gt_count': 0,
        'pred_count': 0,
        'matched_count': 0,
        'ious': []
    })
    
    valid_results = [r for r in results if r.get('error') is None]
    
    for result in valid_results:
        total_gt_boxes += result.get('total_gt_boxes', 0)
        total_pred_boxes += result.get('total_pred_boxes', 0)
        total_matched_boxes += result.get('total_matched_boxes', 0)
        
        if result.get('mean_iou', 0) > 0:
            all_ious.append(result['mean_iou'])
        
        # Агрегируем по классам
        for class_name, class_result in result.get('class_results', {}).items():
            class_stats[class_name]['gt_count'] += class_result['gt_count']
            class_stats[class_name]['pred_count'] += class_result['pred_count']
            class_stats[class_name]['matched_count'] += class_result['matched_count']
            # Здесь можно добавить ious, но для упрощения используем mean_iou из результата
    
    # Общие метрики
    precision = total_matched_boxes / total_pred_boxes if total_pred_boxes > 0 else 0.0
    recall = total_matched_boxes / total_gt_boxes if total_gt_boxes > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    # Метрики по классам
    class_metrics = {}
    for class_name, stats in class_stats.items():
        if stats['gt_count'] > 0 or stats['pred_count'] > 0:
            precision = stats['matched_count'] / stats['pred_count'] if stats['pred_count'] > 0 else 0.0
            recall = stats['matched_count'] / stats['gt_count'] if stats['gt_count'] > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = {
                'gt_count': stats['gt_count'],
                'pred_count': stats['pred_count'],
                'matched_count': stats['matched_count'],
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
    
    return {
        'total_images': len(valid_results),
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'total_matched_boxes': total_matched_boxes,
        'overall_precision': precision,
        'overall_recall': recall,
        'overall_f1': f1_score,
        'mean_iou': mean_iou,
        'class_metrics': class_metrics
    }


def print_report(metrics: Dict, verbose: bool = False):
    """Выводит отчет о результатах тестирования."""
    print("\n" + "="*80)
    print("ОТЧЕТ О ТЕСТИРОВАНИИ ДЕТЕКЦИИ BBOX")
    print("="*80)
    print(f"\nВсего обработано изображений: {metrics['total_images']}")
    print(f"Всего GT bbox: {metrics['total_gt_boxes']}")
    print(f"Всего предсказанных bbox: {metrics['total_pred_boxes']}")
    print(f"Всего совпавших bbox: {metrics['total_matched_boxes']}")
    
    print(f"\nОбщие метрики:")
    print(f"  Precision: {metrics['overall_precision']:.4f} ({metrics['overall_precision']:.2%})")
    print(f"  Recall: {metrics['overall_recall']:.4f} ({metrics['overall_recall']:.2%})")
    print(f"  F1-score: {metrics['overall_f1']:.4f} ({metrics['overall_f1']:.2%})")
    print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    
    print("\n" + "-"*80)
    print("МЕТРИКИ ПО КЛАССАМ:")
    print("-"*80)
    
    # Сортируем классы по F1-score
    sorted_classes = sorted(
        metrics['class_metrics'].items(),
        key=lambda x: x[1]['f1'],
        reverse=True
    )
    
    for class_name, class_metric in sorted_classes:
        print(f"\n{class_name}:")
        print(f"  GT: {class_metric['gt_count']}, Pred: {class_metric['pred_count']}, Matched: {class_metric['matched_count']}")
        print(f"  Precision: {class_metric['precision']:.4f} ({class_metric['precision']:.2%})")
        print(f"  Recall: {class_metric['recall']:.4f} ({class_metric['recall']:.2%})")
        print(f"  F1-score: {class_metric['f1']:.4f} ({class_metric['f1']:.2%})")
    
    print("\n" + "="*80)


def save_detailed_report(metrics: Dict, output_path: Path):
    """Сохраняет детальный отчет в JSON файл."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nДетальный отчет сохранен в: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Тестирование точности детекции bbox текстовых полей'
    )
    # Корень репозитория: kyc_document/scripts/ -> parents[2]
    project_root = Path(__file__).resolve().parent.parent.parent
    
    parser.add_argument(
        '--images_dir',
        type=Path,
        default=project_root / 'dataset' / 'russia' / 'passport' / 'data_page' / 'cropped',
        help='Путь к директории с изображениями'
    )
    parser.add_argument(
        '--labels_dir',
        type=Path,
        default=project_root / 'dataset' / 'russia' / 'passport' / 'data_page' / 'cropped' / 'labels',
        help='Путь к директории с разметкой (YOLO .txt или JSON файлы)'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='ONNX',
        choices=['ONNX', 'OpenVINO', 'TFlite'],
        help='Формат моделей'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='Устройство для инференса'
    )
    parser.add_argument(
        '--annotation_format',
        type=str,
        default='auto',
        choices=['auto', 'yolo', 'json'],
        help='Формат разметки (auto определяет по расширению файла)'
    )
    parser.add_argument(
        '--iou_threshold',
        type=float,
        default=0.5,
        help='Минимальный IoU для совпадения bbox'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Путь для сохранения детального отчета (JSON)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Выводить детальную информацию'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Ограничить количество обрабатываемых изображений (для тестирования)'
    )
    
    args = parser.parse_args()
    
    # Преобразуем относительные пути в абсолютные
    if not args.images_dir.is_absolute():
        args.images_dir = project_root / args.images_dir
    if not args.labels_dir.is_absolute():
        args.labels_dir = project_root / args.labels_dir
    
    # Проверяем существование директорий
    if not args.images_dir.exists():
        print(f"Ошибка: директория с изображениями не найдена: {args.images_dir}")
        return
    
    if not args.labels_dir.exists():
        print(f"Ошибка: директория с разметкой не найдена: {args.labels_dir}")
        print(f"Создайте директорию и добавьте файлы с разметкой")
        return
    
    # Инициализируем детектор
    print("Инициализация детектора текстовых полей...")
    detector = TextFieldsDetector(model_format=args.format, device=args.device, verbose=False)
    
    # Находим все изображения
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in args.images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if args.limit:
        image_files = image_files[:args.limit]
    
    print(f"Найдено {len(image_files)} изображений для обработки")
    
    # Обрабатываем каждое изображение
    results = []
    for i, image_path in enumerate(image_files, 1):
        # Ищем соответствующий файл разметки
        # Пробуем сначала .txt (YOLO), потом .json
        annotation_path = None
        annotation_format = args.annotation_format
        
        if annotation_format == 'auto' or annotation_format == 'yolo':
            txt_path = args.labels_dir / (image_path.stem + '.txt')
            if txt_path.exists():
                annotation_path = txt_path
                annotation_format = 'yolo'
        
        if annotation_path is None and (annotation_format == 'auto' or annotation_format == 'json'):
            json_path = args.labels_dir / (image_path.stem + '.json')
            if json_path.exists():
                annotation_path = json_path
                annotation_format = 'json'
        
        if annotation_path is None or not annotation_path.exists():
            if args.verbose:
                print(f"[{i}/{len(image_files)}] Пропущено {image_path.name}: разметка не найдена")
            continue
        
        print(f"[{i}/{len(image_files)}] Обработка {image_path.name}...", end=' ', flush=True)
        
        result = process_image(
            image_path,
            annotation_path,
            detector,
            annotation_format=annotation_format,
            iou_threshold=args.iou_threshold,
            verbose=args.verbose
        )
        results.append(result)
        
        if result.get('error'):
            print(f"ОШИБКА: {result['error']}")
        else:
            matched = result.get('total_matched_boxes', 0)
            gt = result.get('total_gt_boxes', 0)
            pred = result.get('total_pred_boxes', 0)
            mean_iou = result.get('mean_iou', 0.0)
            print(f"✓ (GT: {gt}, Pred: {pred}, Matched: {matched}, IoU: {mean_iou:.3f})")
    
    # Вычисляем метрики
    print("\nВычисление метрик...")
    metrics = calculate_metrics(results)
    
    # Выводим отчет
    print_report(metrics, args.verbose)
    
    # Сохраняем детальный отчет если указан путь
    if args.output:
        save_detailed_report(metrics, args.output)


if __name__ == '__main__':
    main()

