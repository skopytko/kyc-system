"""
Скрипт для тестирования точности OCR распознавания на датасете с разметкой.

Сравнивает результаты OCR с ground truth разметкой и вычисляет метрики точности.

Пример использования:
    python test_ocr_accuracy.py --verbose
    
    python test_ocr_accuracy.py --images_dir /path/to/images --labels_dir /path/to/labels
    
    python test_ocr_accuracy.py --limit 10 --format ONNX --device cpu

Формат разметки (JSON файлы в директории labels):
    {
      "image_id": "passport_centerfold_011.jpg",
      "text_annotations": {
        "last_name": "ТИМОНИН",
        "first_name": "СЕРГЕЙ",
        "patronymic": "КОНСТАНТИНОВИЧ",
        "birth_date": "09.05.1976",
        "gender": "M",
        "birth_place": "г. Москва",
        "passport_series_number": "63 10 258741",
        "issue_date": "05.04.2006",
        "issue_authority": "ОУФМС РОССИИ ПО ГОР. МОСКВЕ",
        "issue_code": "640-004"
      }
    }
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import re

# Добавляем путь к модулю
sys.path.append(str(Path(__file__).parent.parent.parent))

from kyc_document.document_processing import Pipeline


# Маппинг полей из разметки в поля OCR
FIELD_MAPPING = {
    'last_name': 'Last_name_ru',
    'first_name': 'First_name_ru',
    'patronymic': 'Middle_name_ru',
    'birth_date': 'Birth_date',
    'gender': 'Sex_ru',
    'birth_place': 'Birth_place_ru',
    'passport_series_number': 'Licence_number',
    'issue_date': 'Issue_date',
    'issue_authority': 'Issue_organization_ru',
    'issue_code': 'Issue_organisation_code'
}


def normalize_text(text: str) -> str:
    """Нормализует текст для сравнения: убирает лишние пробелы, приводит к верхнему регистру."""
    if not text:
        return ""
    # Убираем множественные пробелы, обрезаем пробелы по краям
    text = re.sub(r'\s+', ' ', text.strip())
    return text.upper()


def normalize_date(date_str: str) -> str:
    """Нормализует дату для сравнения."""
    if not date_str:
        return ""
    # Убираем все пробелы и точки, оставляем только цифры
    date_clean = re.sub(r'[^\d]', '', date_str)
    # Если формат DDMMYYYY, преобразуем в DD.MM.YYYY
    if len(date_clean) == 8:
        return f"{date_clean[:2]}.{date_clean[2:4]}.{date_clean[4:]}"
    return date_str.strip()


def compare_fields(ground_truth: str, ocr_result: str, field_name: str) -> bool:
    """
    Сравнивает поле из разметки с результатом OCR.
    
    Args:
        ground_truth: Значение из разметки
        ocr_result: Результат OCR
        field_name: Название поля
        
    Returns:
        True если поля совпадают, False иначе
    """
    if not ground_truth and not ocr_result:
        return True
    if not ground_truth or not ocr_result:
        return False
    
    # Для дат используем специальную нормализацию
    if 'date' in field_name.lower():
        gt_norm = normalize_date(ground_truth)
        ocr_norm = normalize_date(ocr_result)
        return gt_norm == ocr_norm
    
    # Для остальных полей используем обычную нормализацию
    gt_norm = normalize_text(ground_truth)
    ocr_norm = normalize_text(ocr_result)
    return gt_norm == ocr_norm


def load_annotation(annotation_path: Path) -> Optional[Dict]:
    """Загружает разметку из JSON файла."""
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('text_annotations', {})
    except Exception as e:
        print(f"Ошибка при загрузке разметки {annotation_path}: {e}")
        return None


def process_image(
    image_path: Path,
    annotation_path: Path,
    pipeline: Pipeline,
    verbose: bool = False
) -> Dict:
    """
    Обрабатывает одно изображение и сравнивает с разметкой.
    
    Returns:
        Словарь с результатами сравнения
    """
    image_name = image_path.name
    
    # Загружаем разметку
    annotations = load_annotation(annotation_path)
    if annotations is None:
        return {
            'image_id': image_name,
            'error': 'Failed to load annotation',
            'fields': {}
        }
    
    # Запускаем OCR
    try:
        result = pipeline(
            str(image_path),
            ocr=True,
            get_doc_borders=True,
            find_text_fields=True,
            check_quality=False,
            low_quality=True
        )
        ocr_results = result.ocr if result.ocr else {}
    except Exception as e:
        if verbose:
            print(f"Ошибка при обработке {image_name}: {e}")
        return {
            'image_id': image_name,
            'error': str(e),
            'fields': {}
        }
    
    # Сравниваем поля
    field_comparisons = {}
    for gt_field, ocr_field in FIELD_MAPPING.items():
        gt_value = annotations.get(gt_field, '')
        ocr_value = ocr_results.get(ocr_field, '')
        
        is_match = compare_fields(gt_value, ocr_value, gt_field)
        
        field_comparisons[gt_field] = {
            'ground_truth': gt_value,
            'ocr_result': ocr_value,
            'match': is_match
        }
    
    return {
        'image_id': image_name,
        'error': None,
        'fields': field_comparisons
    }


def calculate_metrics(results: List[Dict]) -> Dict:
    """Вычисляет метрики точности по всем результатам."""
    # Подсчитываем статистику по каждому полю
    field_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': []})
    
    total_images = 0
    images_with_errors = 0
    
    for result in results:
        if result.get('error'):
            images_with_errors += 1
            continue
        
        total_images += 1
        all_fields_match = True
        
        for field_name, comparison in result['fields'].items():
            field_stats[field_name]['total'] += 1
            if comparison['match']:
                field_stats[field_name]['correct'] += 1
            else:
                all_fields_match = False
                field_stats[field_name]['errors'].append({
                    'image_id': result['image_id'],
                    'ground_truth': comparison['ground_truth'],
                    'ocr_result': comparison['ocr_result']
                })
        
        if not all_fields_match:
            images_with_errors += 1
    
    # Вычисляем метрики для каждого поля
    field_metrics = {}
    for field_name, stats in field_stats.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total']
            field_metrics[field_name] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total'],
                'errors': stats['errors']
            }
        else:
            field_metrics[field_name] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'errors': []
            }
    
    # Общая точность (процент полностью правильных изображений)
    overall_accuracy = 0.0
    if total_images > 0:
        fully_correct = total_images - images_with_errors
        overall_accuracy = fully_correct / total_images
    
    # Средняя точность по полям
    avg_field_accuracy = 0.0
    if field_metrics:
        avg_field_accuracy = sum(m['accuracy'] for m in field_metrics.values()) / len(field_metrics)
    
    return {
        'total_images': total_images,
        'images_with_errors': images_with_errors,
        'fully_correct_images': total_images - images_with_errors,
        'overall_accuracy': overall_accuracy,
        'average_field_accuracy': avg_field_accuracy,
        'field_metrics': field_metrics
    }


def print_report(metrics: Dict, verbose: bool = False):
    """Выводит отчет о результатах тестирования."""
    print("\n" + "="*80)
    print("ОТЧЕТ О ТЕСТИРОВАНИИ OCR")
    print("="*80)
    print(f"\nВсего обработано изображений: {metrics['total_images']}")
    print(f"Полностью правильных: {metrics['fully_correct_images']}")
    print(f"С ошибками: {metrics['images_with_errors']}")
    print(f"\nОбщая точность (полностью правильные): {metrics['overall_accuracy']:.2%}")
    print(f"Средняя точность по полям: {metrics['average_field_accuracy']:.2%}")
    
    print("\n" + "-"*80)
    print("ТОЧНОСТЬ ПО ПОЛЯМ:")
    print("-"*80)
    
    # Сортируем поля по точности
    sorted_fields = sorted(
        metrics['field_metrics'].items(),
        key=lambda x: x[1]['accuracy'],
        reverse=True
    )
    
    for field_name, field_metric in sorted_fields:
        accuracy = field_metric['accuracy']
        correct = field_metric['correct']
        total = field_metric['total']
        print(f"\n{field_name}:")
        print(f"  Точность: {accuracy:.2%} ({correct}/{total})")
        
        if verbose and field_metric['errors']:
            print(f"  Ошибки ({len(field_metric['errors'])}):")
            for error in field_metric['errors'][:10]:  # Показываем первые 10 ошибок
                print(f"    {error['image_id']}:")
                print(f"      Ожидалось: '{error['ground_truth']}'")
                print(f"      Получено:  '{error['ocr_result']}'")
            if len(field_metric['errors']) > 10:
                print(f"    ... и еще {len(field_metric['errors']) - 10} ошибок")
    
    print("\n" + "="*80)


def save_detailed_report(metrics: Dict, output_path: Path):
    """Сохраняет детальный отчет в JSON файл."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nДетальный отчет сохранен в: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Тестирование точности OCR на датасете с разметкой'
    )
    # Определяем корень проекта (на уровень выше kyc_document)
    project_root = Path(__file__).parent.parent.parent.parent
    
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
        help='Путь к директории с разметкой (JSON файлы)'
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
        '--output',
        type=Path,
        help='Путь для сохранения детального отчета (JSON)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Выводить детальную информацию об ошибках'
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
        print(f"Создайте директорию и добавьте JSON файлы с разметкой")
        return
    
    # Инициализируем pipeline
    print("Инициализация OCR pipeline...")
    pipeline = Pipeline(model_format=args.format, device=args.device, verbose=False)
    
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
        annotation_name = image_path.stem + '.json'
        annotation_path = args.labels_dir / annotation_name
        
        if not annotation_path.exists():
            print(f"[{i}/{len(image_files)}] Пропущено {image_path.name}: разметка не найдена")
            continue
        
        print(f"[{i}/{len(image_files)}] Обработка {image_path.name}...", end=' ', flush=True)
        
        result = process_image(image_path, annotation_path, pipeline, args.verbose)
        results.append(result)
        
        if result.get('error'):
            print(f"ОШИБКА: {result['error']}")
        else:
            # Подсчитываем количество совпадений
            matches = sum(1 for f in result['fields'].values() if f['match'])
            total = len(result['fields'])
            print(f"✓ ({matches}/{total} полей совпадают)")
    
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

