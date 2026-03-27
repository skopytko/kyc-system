"""
Визуализация bbox на изображениях для belarus_passport_1996 TextFields Detector.
Сохраняет картинки с нарисованными боксами в runs/test_visualizations.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../../dataset/belarus/passport_1996/data_page/cropped'
)

CLASS_NAMES = [
    'authority', 'authority2', 'authority_code', 'code_of_issuing', 'date_of_birth',
    'date_of_expiry', 'date_of_issue', 'identification_no', 'mini_photo', 'mrz_line1',
    'mrz_line2', 'names', 'nationality', 'passport_no', 'photo', 'place_of_birth',
    'sex', 'signature', 'signature2', 'surname', 'type'
]


def prepare_test_images(dataset_path, train_ratio=0.8, val_ratio=0.1, max_images=None, random_state=42):
    """Подготавливает список тестовых изображений."""
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
    if max_images:
        test_files = test_files[:max_images]

    return [os.path.join(images_dir, f) for f in test_files]


def find_best_model(runs_dir):
    """Ищет best.pt в runs/train, runs/train2, ..."""
    if not os.path.exists(runs_dir):
        return None
    for name in sorted(os.listdir(runs_dir)):
        if name.startswith('train'):
            cand = os.path.join(runs_dir, name, 'weights', 'best.pt')
            if os.path.exists(cand):
                return cand
    return None


def draw_boxes(img, results, class_names, conf_threshold=0.25):
    """Рисует bbox на изображении. Возвращает img с боксами."""
    result = img.copy()
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return result

    boxes_data = results[0].boxes.data.cpu().numpy()
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100),
        (255, 100, 255), (100, 255, 255), (200, 150, 100), (100, 200, 150),
        (150, 100, 200), (200, 100, 150), (100, 150, 200), (200, 200, 100),
        (100, 200, 200), (200, 100, 200), (150, 200, 100), (100, 150, 200),
        (180, 120, 80), (80, 180, 120), (120, 80, 180), (180, 80, 120), (80, 120, 180),
    ]

    for box in boxes_data:
        x1, y1, x2, y2 = box[:4].astype(int)
        conf = float(box[4]) if len(box) > 4 else 1.0
        cls = int(box[5]) if len(box) > 5 else 0

        if conf < conf_threshold:
            continue

        color = colors[cls % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        name = class_names[cls] if cls < len(class_names) else f'cls_{cls}'
        label = f'{name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def main():
    import sys
    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    model_path = os.environ.get(
        'TEXTFIELDS_MODEL_PATH',
        find_best_model(runs_dir) or os.path.join(runs_dir, 'train', 'weights', 'best.pt')
    )
    output_dir = os.path.join(runs_dir, 'test_visualizations')
    dataset_path = os.environ.get('TEXTFIELDS_DATASET', DATASET_PATH)
    conf_threshold = float(os.environ.get('TEXTFIELDS_CONF', 0.25))
    max_images = int(os.environ.get('TEXTFIELDS_VIS_MAX', 20))  # по умолчанию 20 картинок
    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))

    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f'Model: {model_path}')
    model = YOLO(model_path)

    # Один файл передан аргументом — тест только на нём
    if len(sys.argv) > 1:
        single_path = os.path.abspath(sys.argv[1])
        if not os.path.exists(single_path):
            print(f'Image not found: {single_path}')
            return
        test_images = [single_path]
        print(f'Processing: {single_path}')
    else:
        test_images = prepare_test_images(dataset_path, max_images=max_images, random_state=random_state)
        print(f'Processing {len(test_images)} images -> {output_dir}')

    for img_path in tqdm(test_images, desc='Visualize'):
        img = cv2.imread(img_path)
        if img is None:
            continue

        results = model(img, conf=conf_threshold, verbose=False)
        result_img = draw_boxes(img, results, CLASS_NAMES, conf_threshold)

        name = os.path.basename(img_path)
        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, result_img)

    print(f'\nSaved to: {output_dir}')


if __name__ == '__main__':
    main()
