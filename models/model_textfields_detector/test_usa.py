"""
Тестирование YOLO TextFields Detector для одного или всех штатов USA.
Датасет: dataset/usa/<state>/cropped
Модель: runs/usa_<state>/weights/best.pt

Запуск одного штата: USA_STATE=alabama python test_usa.py
Все штаты: python test_usa.py all
"""

import os
import sys
import torch
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATASET_USA_ROOT = os.path.join(REPO_ROOT, 'dataset', 'usa')
RUNS_DIR = os.path.join(os.path.dirname(__file__), 'runs')


def prepare_test_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Тестовый сплит (10%) — те же индексы, что при обучении."""
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
    """Загружает YOLO bbox из файла."""
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
            boxes.append({'class': class_id, 'bbox': [x1, y1, x2, y2]})
    return boxes


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    a1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    a2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def find_usa_model(state):
    """Ищет best.pt для штата: usa_<state>, usa_<state>2, ..."""
    prefix = f'usa_{state}'
    if not os.path.exists(RUNS_DIR):
        return None
    for name in sorted(os.listdir(RUNS_DIR)):
        if name == prefix or (name.startswith(prefix) and len(name) > len(prefix)):
            cand = os.path.join(RUNS_DIR, name, 'weights', 'best.pt')
            if os.path.exists(cand):
                return cand
    return None


def run_test_for_state(state, conf_threshold=0.25, iou_threshold=0.45, random_state=42):
    """Запускает тест для одного штата. Возвращает dict с метриками или None при ошибке."""
    dataset_path = os.path.join(DATASET_USA_ROOT, state, 'cropped')
    model_path = find_usa_model(state)

    if not os.path.exists(dataset_path):
        print(f'  [{state}] Dataset not found: {dataset_path}')
        return None
    if not model_path:
        print(f'  [{state}] Model not found (runs/usa_{state}*/weights/best.pt)')
        return None

    model = YOLO(model_path)
    test_images, test_labels = prepare_test_dataset(dataset_path, random_state=random_state)

    all_ious = []
    all_precisions = []
    all_recalls = []

    for img_path, label_path in zip(test_images, test_labels):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gt_boxes = load_yolo_boxes(label_path, img.shape)
        if len(gt_boxes) == 0:
            continue

        results = model(img, conf=conf_threshold, iou=iou_threshold, verbose=False)
        pred_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                conf = float(box[4]) if len(box) > 4 else 1.0
                cls = int(box[5]) if len(box) > 5 else 0
                pred_boxes.append({'class': cls, 'bbox': [float(x1), float(y1), float(x2), float(y2)], 'conf': conf})

        if len(pred_boxes) == 0:
            continue

        matched_gt = [False] * len(gt_boxes)
        ious = []
        for pred_box in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i] or pred_box['class'] != gt_box['class']:
                    continue
                iou = calculate_iou(pred_box['bbox'], gt_box['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou >= iou_threshold:
                matched_gt[best_idx] = True
                ious.append(best_iou)

        if ious:
            all_ious.extend(ious)
            all_precisions.append(len(ious) / len(pred_boxes))
            all_recalls.append(len(ious) / len(gt_boxes))

    if not all_ious:
        return {
            'state': state,
            'n_test': len(test_images),
            'mean_iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'matched': 0,
        }

    mean_iou = np.mean(all_ious)
    mean_p = np.mean(all_precisions)
    mean_r = np.mean(all_recalls)
    f1 = 2 * mean_p * mean_r / (mean_p + mean_r) if (mean_p + mean_r) > 0 else 0.0

    return {
        'state': state,
        'n_test': len(test_images),
        'mean_iou': mean_iou,
        'precision': mean_p,
        'recall': mean_r,
        'f1': f1,
        'matched': len(all_ious),
    }


def main():
    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))
    conf = float(os.environ.get('TEXTFIELDS_CONF', 0.25))
    iou = float(os.environ.get('TEXTFIELDS_IOU', 0.45))

    if len(sys.argv) > 1 and sys.argv[1].lower() == 'all':
        states = []
        for d in sorted(os.listdir(RUNS_DIR)):
            if d.startswith('usa_'):
                state = d[4:]
                if os.path.isdir(os.path.join(DATASET_USA_ROOT, state)):
                    states.append(state)
        if not states:
            print('No usa_* runs or dataset/usa/<state> found.')
            sys.exit(1)
    else:
        state = (os.environ.get('USA_STATE') or (sys.argv[1] if len(sys.argv) > 1 else '')).strip().lower()
        if not state:
            print('Usage: USA_STATE=alabama python test_usa.py')
            print('   or: python test_usa.py alabama')
            print('   or: python test_usa.py all')
            sys.exit(1)
        states = [state]

    print(f'Testing USA TextFields Detector (conf={conf}, iou={iou})')
    print('=' * 70)

    all_results = []
    for s in states:
        print(f'  {s}...', end=' ', flush=True)
        r = run_test_for_state(s, conf_threshold=conf, iou_threshold=iou, random_state=random_state)
        if r is None:
            print('skip')
            continue
        all_results.append(r)
        print(f"IoU={r['mean_iou']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} n={r['n_test']}")

    if not all_results:
        sys.exit(1)

    # Сводка
    print('=' * 70)
    print(f'{"State":<14} {"IoU":>8} {"P":>8} {"R":>8} {"F1":>8} {"N_test":>8}')
    print('-' * 70)
    for r in all_results:
        print(f"{r['state']:<14} {r['mean_iou']:>8.3f} {r['precision']:>8.3f} {r['recall']:>8.3f} {r['f1']:>8.3f} {r['n_test']:>8}")
    if len(all_results) > 1:
        avg_iou = np.mean([x['mean_iou'] for x in all_results])
        avg_p = np.mean([x['precision'] for x in all_results])
        avg_r = np.mean([x['recall'] for x in all_results])
        avg_f1 = np.mean([x['f1'] for x in all_results])
        print('-' * 70)
        print(f'{"(avg)":<14} {avg_iou:>8.3f} {avg_p:>8.3f} {avg_r:>8.3f} {avg_f1:>8.3f}')

    out_path = os.path.join(RUNS_DIR, 'test_results_usa.txt')
    with open(out_path, 'w') as f:
        f.write('USA TextFields Detector - Test Results\n')
        f.write('=' * 70 + '\n')
        for r in all_results:
            f.write(f"{r['state']}\t{r['mean_iou']:.4f}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['n_test']}\n")
    print(f'\nResults saved to: {out_path}')


if __name__ == '__main__':
    main()
