"""
Обучение YOLO TextFields Detector для одного штата USA.
Датасет: dataset/usa/<STATE>/cropped (images + labels/).
Запуск: USA_STATE=alabama python train_usa.py
Или из скрипта по очереди для всех штатов.
"""

import os
import sys
import yaml
import torch
from ultralytics import YOLO

# Корень репозитория (на два уровня выше этой папки)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DATASET_USA_ROOT = os.path.join(REPO_ROOT, 'dataset', 'usa')


def get_max_class_id(labels_dir):
    """Определяет максимальный class id по всем .txt в labels_dir."""
    max_id = -1
    for f in os.listdir(labels_dir):
        if not f.endswith('.txt') or f == 'classes.txt':
            continue
        path = os.path.join(labels_dir, f)
        with open(path, 'r') as fp:
            for line in fp:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        max_id = max(max_id, int(parts[0]))
                    except ValueError:
                        pass
    return max_id


def prepare_yolo_dataset(dataset_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Подготавливает датасет для YOLO. Split: 80% train, 10% val, 10% test."""
    images_dir = dataset_path
    labels_dir = os.path.join(dataset_path, 'labels')

    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")

    image_files = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('.'):
            label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(label_path):
                image_files.append(f)

    if len(image_files) == 0:
        raise ValueError("No images with labels found!")

    print(f"Found {len(image_files)} images with labels")

    generator = torch.Generator().manual_seed(random_state)
    n = len(image_files)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    train_files = [image_files[i] for i in indices[:n_train]]
    val_files = [image_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [image_files[i] for i in indices[n_train + n_val:]]

    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    splits = {'train': train_files, 'val': val_files, 'test': test_files}
    for split, files in splits.items():
        for img_file in files:
            img_src = os.path.abspath(os.path.join(images_dir, img_file))
            label_src = os.path.abspath(os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt'))
            img_link = os.path.join(output_dir, 'images', split, img_file)
            label_link = os.path.join(output_dir, 'labels', split, os.path.splitext(img_file)[0] + '.txt')
            for dst in (img_link, label_link):
                if os.path.exists(dst) or os.path.islink(dst):
                    os.remove(dst)
            os.symlink(img_src, img_link)
            os.symlink(label_src, label_link)

    # Классы: из classes.txt или по числу классов в разметке
    classes_file = os.path.join(labels_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r', encoding='utf-8') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        max_id = get_max_class_id(labels_dir)
        nc = max_id + 1
        # Совпадает с docs_generator/usa/generator.py:
        # IMAGE_FIELDS (0–2) + TEXT_FIELDS без dob_short (через dob) => 20 классов.
        default_names_20 = [
            "photo",
            "mini_photo",
            "handwritten_signature",
            "class",
            "end",
            "rest",
            "firstname",
            "lastname",
            "address",
            "sex",
            "hgt",
            "wgt",
            "eyes",
            "hair",
            "dd",
            "dln",
            "iss",
            "iss_duplicate",
            "exp",
            "dob",
        ]
        class_names = default_names_20[:nc] if nc <= len(default_names_20) else (default_names_20 + [f"class_{i}" for i in range(len(default_names_20), nc)])

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            'path': os.path.abspath(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }, f, default_flow_style=False, allow_unicode=True)

    print(f"Dataset prepared. Classes: {len(class_names)}")
    return yaml_path


def main():
    state = (os.environ.get('USA_STATE') or '').strip().lower()
    if not state:
        print('Usage: USA_STATE=alabama python train_usa.py')
        print('Or: python train_usa.py alabama')
        sys.exit(1)
    if len(sys.argv) > 1:
        state = sys.argv[1].strip().lower()

    dataset_path = os.path.join(DATASET_USA_ROOT, state, 'cropped')
    if not os.path.exists(dataset_path):
        print(f'Dataset not found: {dataset_path}')
        sys.exit(1)

    # Папки runs лежат рядом с train_usa.py (в model_textfields_detector)
    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    dataset_dir = os.path.join(runs_dir, 'dataset_usa', state)
    os.makedirs(dataset_dir, exist_ok=True)

    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))
    epochs = int(os.environ.get('TEXTFIELDS_EPOCHS', 100))
    batch_size = int(os.environ.get('TEXTFIELDS_BATCH', 8))
    imgsz = int(os.environ.get('TEXTFIELDS_IMGSZ', 640))
    model_size = os.environ.get('TEXTFIELDS_MODEL', 'yolo11n.pt')
    device = os.environ.get('TEXTFIELDS_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
    patience = int(os.environ.get('TEXTFIELDS_PATIENCE', 5))

    # Всегда готовим датасет заново для данного штата
    print(f"Preparing dataset from: {dataset_path}")
    data_yaml = prepare_yolo_dataset(dataset_path, dataset_dir, random_state=random_state)

    print(f"\nTraining parameters (USA {state}):")
    print(f"  Model: {model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {imgsz}")
    print(f"  Device: {device}")
    print(f"  Patience: {patience}")

    model = YOLO(model_size)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=runs_dir,
        name=f'usa_{state}',
        patience=patience,
        save=True,
        save_period=10,
        augment=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        scale=0.5,
        translate=0.1,
        shear=0.0,
        perspective=0.0,
        degrees=5.0,
        box=7.5,
        rect=False,
        verbose=True
    )

    print(f'\nBest model: {results.save_dir}/weights/best.pt')


if __name__ == '__main__':
    main()
