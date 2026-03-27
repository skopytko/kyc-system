"""
Тестирование YOLO TextFields Detector для Belarus passport_1996.
Датасет: dataset/belarus/passport_1996/data_page/cropped
"""

import os
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../../dataset/belarus/passport_1996/data_page/cropped'
)


def prepare_test_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Подготавливает тестовый датасет (10% по умолчанию)."""
    images_dir = dataset_path
    labels_dir = os.path.join(dataset_path, 'labels')

    image_files = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
            label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(label_path):
                image_files.append(f)

    generator = torch.Generator().manual_seed(random_state)
    n = len(image_files)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    test_files = [image_files[i] for i in indices[n_train + n_val:]]

    test_images = []
    test_labels = []
    for img_file in test_files:
        test_images.append(os.path.join(images_dir, img_file))
        test_labels.append(os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt'))

    return test_images, test_labels


def load_yolo_boxes(label_path, img_shape):
    """Загружает YOLO bounding boxes из файла."""
    h, w = img_shape[:2]
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append({
                'class': class_id,
                'bbox': [x1, y1, x2, y2]
            })
    return boxes


def calculate_iou(box1, box2):
    """Вычисляет IoU между двумя bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def find_best_model(runs_dir):
    """Ищет best.pt в runs/train, runs/train2, ..."""
    for name in sorted(os.listdir(runs_dir)) if os.path.exists(runs_dir) else []:
        if name.startswith('train'):
            cand = os.path.join(runs_dir, name, 'weights', 'best.pt')
            if os.path.exists(cand):
                return cand
    return None


def main():
    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    model_path = os.environ.get(
        'TEXTFIELDS_MODEL_PATH',
        find_best_model(runs_dir) or os.path.join(runs_dir, 'train', 'weights', 'best.pt')
    )
    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))
    conf_threshold = float(os.environ.get('TEXTFIELDS_CONF', 0.25))
    iou_threshold = float(os.environ.get('TEXTFIELDS_IOU', 0.45))
    dataset_path = os.environ.get('TEXTFIELDS_DATASET', DATASET_PATH)

    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Run training first: python train.py')
        return

    print(f'Loading model from: {model_path}')
    model = YOLO(model_path)

    test_images, test_labels = prepare_test_dataset(dataset_path, random_state=random_state)

    print(f'Testing on {len(test_images)} images (passport_1996 data_page cropped)')
    print(f'Confidence threshold: {conf_threshold}, IoU threshold: {iou_threshold}')

    all_ious = []
    all_precisions = []
    all_recalls = []
    failed_images = []

    for img_path, label_path in tqdm(zip(test_images, test_labels), desc='Testing', total=len(test_images)):
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_boxes = load_yolo_boxes(label_path, img.shape)
        if len(gt_boxes) == 0:
            continue

        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        pred_boxes = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            for box in boxes_data:
                x1, y1, x2, y2 = box[:4]
                conf = box[4] if len(box) > 4 else 1.0
                cls = int(box[5]) if len(box) > 5 else 0
                pred_boxes.append({
                    'class': cls,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(conf)
                })

        if len(pred_boxes) == 0:
            failed_images.append({
                'path': img_path,
                'reason': 'No predictions'
            })
            continue

        # Сопоставляем предсказания с GT по максимальному IoU
        matched_gt = [False] * len(gt_boxes)
        ious = []

        for pred_box in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]:
                    continue
                if pred_box['class'] != gt_box['class']:
                    continue
                iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_idx >= 0 and best_iou >= iou_threshold:
                matched_gt[best_idx] = True
                ious.append(best_iou)

        if len(ious) > 0:
            all_ious.extend(ious)
            precision = len(ious) / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
            recall = len(ious) / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
            all_precisions.append(precision)
            all_recalls.append(recall)

            if recall < 0.5 or precision < 0.5:
                failed_images.append({
                    'path': img_path,
                    'precision': precision,
                    'recall': recall,
                    'avg_iou': np.mean(ious) if ious else 0.0
                })

    results = []
    results.append('=' * 60)
    results.append('belarus_passport_1996 TextFields Detector - TEST RESULTS')
    results.append('=' * 60)

    if len(all_ious) > 0:
        mean_iou = np.mean(all_ious)
        mean_precision = np.mean(all_precisions)
        mean_recall = np.mean(all_recalls)
        mean_f1 = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0.0

        results.append(f'\nOverall Metrics:')
        results.append(f'  Mean IoU: {mean_iou:.4f}')
        results.append(f'  Precision: {mean_precision:.4f}')
        results.append(f'  Recall: {mean_recall:.4f}')
        results.append(f'  F1-score: {mean_f1:.4f}')
        results.append(f'  Total matched boxes: {len(all_ious)}')
    else:
        results.append('\nNo successful predictions found!')

    if failed_images:
        results.append(f'\nFailed/Low Quality Predictions ({len(failed_images)}/{len(test_images)}):')
        results.append('=' * 60)
        for i, item in enumerate(failed_images[:20], 1):
            if 'reason' in item:
                results.append(f"{i}. {os.path.basename(item['path'])}: {item['reason']}")
            else:
                results.append(f"{i}. {os.path.basename(item['path'])}: P={item['precision']:.3f}, R={item['recall']:.3f}, IoU={item['avg_iou']:.3f}")

    results_text = '\n'.join(results)
    print(results_text)

    output_path = os.path.join(runs_dir, 'test_results.txt')
    os.makedirs(runs_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(results_text)

    print(f'\nResults saved to: {output_path}')


if __name__ == '__main__':
    main()
