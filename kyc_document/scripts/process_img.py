import sys
sys.path.append('..')
from kyc_document.document_processing import Pipeline
from pathlib import Path
import pprint
import argparse
import cv2
import numpy as np
import os


def save_intermediate_results(result, output_dir='intermediate_results'):
    """Сохраняет промежуточные результаты обработки в указанную директорию.
    
    Args:
        result: Результат работы Pipeline
        output_dir: Директория для сохранения результатов
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем оригинальное изображение
    if result.meta_results.get('original_img') is not None:
        cv2.imwrite(os.path.join(output_dir, 'original.jpg'), cv2.cvtColor(result.meta_results['original_img'], cv2.COLOR_RGB2BGR))
    
    # Сохраняем развернутое изображение
    if result.rotated_image is not None:
        img = result.rotated_image.copy()
        # Получаем показатель уверенности (например, DocConf)
        doc_conf = None
        if hasattr(result, 'full_report') and isinstance(result.full_report, dict):
            doc_conf = result.full_report.get('Quality', {}).get('DocConf', None)
        if doc_conf is not None:
            cv2.putText(
                img,
                f"DocConf: {doc_conf:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
        cv2.imwrite(os.path.join(output_dir, 'rotated.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Сохраняем изображение с обнаруженными границами документа
    if result.meta_results.get('DocDetector') and result.meta_results['DocDetector'].get('border_img') is not None:
        cv2.imwrite(os.path.join(output_dir, 'doc_detection.jpg'), cv2.cvtColor(result.meta_results['DocDetector']['border_img'], cv2.COLOR_RGB2BGR))
    
    # Сохраняем изображение с исправленной перспективой
    if result.img_with_fixed_perspective is not None:
        cv2.imwrite(os.path.join(output_dir, 'fixed_perspective.jpg'), cv2.cvtColor(result.img_with_fixed_perspective, cv2.COLOR_RGB2BGR))
    
    # Сохраняем изображение с обнаруженными печатями
    if result.meta_results.get('PassportSealDetector') and result.meta_results['PassportSealDetector'].get('seals_img') is not None:
        cv2.imwrite(os.path.join(output_dir, 'seal_detection.jpg'), cv2.cvtColor(result.meta_results['PassportSealDetector']['seals_img'], cv2.COLOR_RGB2BGR))
    
    # Сохраняем обрезанное изображение печати
    if result.meta_results.get('PassportSealDetector') and result.meta_results['PassportSealDetector'].get('fixed_seal_img') is not None:
        cv2.imwrite(os.path.join(output_dir, 'fixed_seal.jpg'), cv2.cvtColor(result.meta_results['PassportSealDetector']['fixed_seal_img'], cv2.COLOR_RGB2BGR))
    
    # Сохраняем изображение с отмеченными текстовыми полями
    if result.text_fields is not None and result.img_with_fixed_perspective is not None:
        coords, _ = result.text_fields
        img_with_boxes = result.img_with_fixed_perspective.copy()
        for box in coords:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if len(box) > 6:
                label = box[-1]
                conf = box[4]
                label_text = f'{label} {conf:.2f}'
                (w, h), b = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_with_boxes, (x1, y1 - h - b - 5), (x1 + w, y1), (0, 255, 0), -1)
                cv2.putText(img_with_boxes, label_text, (x1, y1 - b - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(output_dir, 'text_fields.jpg'), cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))
    
    # Сохраняем отдельные патчи слов
    if result.words_patches is not None:
        for i, (field_name, patches) in enumerate(result.words_patches.items()):
            for j, patch_info in enumerate(patches):
                # patch_info может быть np.ndarray или кортежем/списком
                if isinstance(patch_info, np.ndarray):
                    patch = patch_info
                elif isinstance(patch_info, (list, tuple)):
                    patch = patch_info[0]  # первый элемент — изображение
                else:
                    continue
                cv2.imwrite(os.path.join(output_dir, f'word_{field_name}_{j}.jpg'), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


def process_img(**kwargs) -> dict:
    """Runs pipeline inference on an image.

    Runs a provided document analysis pipeline on a single
    input image. Prints and returns a detailed output report.

    Args:
        img_path: Path to input image
        img_size: Resize image to this size before inference
        check_q: Whether to run pipeline quality checks
        pipeline: Configured Pipeline object
        show_intermediate: Whether to save intermediate results

    Returns:
        dict: Pipeline output report for image
    """

    assert kwargs.get('img_path') is not None, "Missing path to image"
    img_path = kwargs.get('img_path')
    img_size = kwargs.get('img_size')
    check_q = kwargs.get('check_quality')
    show_intermediate = kwargs.get('show_intermediate', False)

    pipeline = kwargs.get('pipeline')

    result = pipeline(img_path, check_quality=check_q, img_size=img_size)
    
    # Выводим полный отчет
    pp = pprint.PrettyPrinter(depth=4, indent=4)
    pp.pprint(result.full_report)
    
    # Сохраняем промежуточные результаты если нужно
    if show_intermediate:
        save_intermediate_results(result)

    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark pipeline')
    parser.add_argument('-i', '--img_path', help='Image path', type=Path)
    parser.add_argument('-f', '--format', help='Select model format TFlite, ONNX, OpenVINO', type=str,
                        default='OpenVINO')
    parser.add_argument('-d', '--device', help='On which device to run - cpu or gpu', default='cpu', type=str)
    parser.add_argument('--check_quality',
                        help='Is there need to check quality?',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        type=bool)
    parser.add_argument('--img_size', help='To which max size reshape image', required=False, default=1500, type=int)
    parser.add_argument('--show_intermediate',
                        help='Save intermediate processing results',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        type=bool)
    args = parser.parse_args()
    params = vars(args)

    pipeline = Pipeline(model_format=params['format'], device=params['device'], )
    params['pipeline'] = pipeline
    process_img(**params)


if __name__ == '__main__':
    main()






