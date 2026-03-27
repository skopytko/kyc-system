import os
import random
import re
import shutil
from pathlib import Path

import mlflow
from ultralytics import YOLO


def clean_metric_name(name):
    """Очищает имя метрики от недопустимых символов для MLflow."""
    name = re.sub(r'\([^)]*\)', '', str(name))
    name = re.sub(r'[^a-zA-Z0-9_\-./: ]', '_', name)
    return name.strip('_')


def collect_border_images(dataset_root: Path):
    """
    Собирает изображения из dataset/*/raw/**/labels, у которых в labels есть разметка класса 0 (border).
    Возвращает список: [(img_path, label_content, unique_name), ...]
    """
    images = []
    for raw_dir in dataset_root.rglob('raw'):
        labels_dir = raw_dir / 'labels'
        if not labels_dir.exists():
            continue

        # Получаем относительный путь от dataset_root для уникальности (чтобы имена не конфликтовали)
        rel_path = raw_dir.relative_to(dataset_root).parent
        prefix = '_'.join(rel_path.parts).replace('/', '_').replace('\\', '_')

        jpgs = list(raw_dir.glob('*.jpg'))
        jpegs = list(raw_dir.glob('*.jpeg'))
        pngs = list(raw_dir.glob('*.png'))
        for img_path in jpgs + jpegs + pngs:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            with open(label_path, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip().startswith('0 ')]

            if not lines:
                continue

            # Создаем уникальное имя с учетом директории
            unique_name = f"{prefix}_{img_path.name}" if prefix else img_path.name
            images.append((img_path, ''.join(lines), unique_name))

    random.seed(42)
    random.shuffle(images)
    return images


def build_yolo_dataset(images, ds_dir: Path):
    """
    Создаёт структуру YOLO: train/test/val.
    Возвращает (n, train_end, test_end).
    """
    for d in [
        ds_dir / 'train' / 'images',
        ds_dir / 'train' / 'labels',
        ds_dir / 'val' / 'images',
        ds_dir / 'val' / 'labels',
        ds_dir / 'test' / 'images',
        ds_dir / 'test' / 'labels',
    ]:
        d.mkdir(parents=True, exist_ok=True)

    n = len(images)
    train_end = int(n * 0.7)
    test_end = train_end + int(n * 0.15)

    for i, (img_path, label_content, unique_name) in enumerate(images):
        if i < train_end:
            dst = ds_dir / 'train'
        elif i < test_end:
            dst = ds_dir / 'test'
        else:
            dst = ds_dir / 'val'

        dst_img = dst / 'images' / unique_name
        shutil.copy(img_path, dst_img)
        (dst / 'labels' / f"{Path(unique_name).stem}.txt").write_text(label_content)

    (ds_dir / 'data.yaml').write_text(
        f"path: {ds_dir.absolute()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"names:\n"
        f"  0: border\n"
    )

    return n, train_end, test_end


def setup_mlflow(project_root: Path):
    """Настраивает mlflow на уровне проекта."""
    mlruns_dir = project_root / 'mlruns'
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(mlruns_dir.resolve()))

    experiment_name = os.environ.get('BORDERS_EXPERIMENT', 'borders-segmentation')
    mlflow.set_experiment(experiment_name)


def run_training(ds_dir: Path, epochs: int, batch_size: int, model_name: str, run_name: str, imgsz: int = 640):
    """Запускает обучение YOLO-seg и логирует метрики/артефакты в MLflow."""
    patience = 5

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'model': model_name,
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'patience': patience,
            'augment': True,
            'mosaic': 0,
            'mixup': 0,
            'copy_paste': 0,
            'degrees': 7.0,
            'scale': 0.5,
            'translate': 0.1,
            'shear': float(os.environ.get('BORDERS_SHEAR', 2.0)),
            'perspective': float(os.environ.get('BORDERS_PERSPECTIVE', 0.0005)),
            'fliplr': float(os.environ.get('BORDERS_FLIPLR', 0.0)),
            'flipud': float(os.environ.get('BORDERS_FLIPUD', 0.0)),
            'hsv_h': float(os.environ.get('BORDERS_HSV_H', 0.02)),
            'hsv_s': float(os.environ.get('BORDERS_HSV_S', 0.9)),
            'hsv_v': float(os.environ.get('BORDERS_HSV_V', 0.6)),
            'rect': False,
        })

        model = YOLO(model_name)
        results = model.train(
            data=str(ds_dir / 'data.yaml'),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=str(Path(__file__).parent),
            name='runs',
            plots=True,
            patience=patience,
            augment=True,
            mosaic=0,
            mixup=0,
            copy_paste=0,
            degrees=7.0,
            scale=0.5,
            translate=0.1,
            shear=float(os.environ.get('BORDERS_SHEAR', 2.0)),
            perspective=float(os.environ.get('BORDERS_PERSPECTIVE', 0.0005)),
            fliplr=float(os.environ.get('BORDERS_FLIPLR', 0.0)),
            flipud=float(os.environ.get('BORDERS_FLIPUD', 0.0)),
            hsv_h=float(os.environ.get('BORDERS_HSV_H', 0.02)),
            hsv_s=float(os.environ.get('BORDERS_HSV_S', 0.9)),
            hsv_v=float(os.environ.get('BORDERS_HSV_V', 0.6)),
            rect=False,
        )

        if results and hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(clean_metric_name(key), value)

        runs_path = Path(__file__).parent / 'runs'
        if runs_path.exists():
            best_model = runs_path / 'segment' / 'train' / 'weights' / 'best.pt'
            if best_model.exists():
                mlflow.log_artifact(str(best_model), artifact_path='models')

            for plot_file in ['results.png', 'confusion_matrix.png', 'F1_curve.png', 'PR_curve.png']:
                plot_path = runs_path / 'segment' / 'train' / plot_file
                if plot_path.exists():
                    mlflow.log_artifact(str(plot_path), artifact_path='plots')

        return results


def print_batch_examples(ds_dir: Path, batch_size: int):
    """Печатает примеры батчей из train/images."""
    train_imgs = sorted((ds_dir / 'train' / 'images').glob('*.*'))
    print(f"\nПримеры батчей (batch_size={batch_size}):")
    for i in range(min(3, (len(train_imgs) + batch_size - 1) // batch_size)):
        batch_start = i * batch_size
        batch_imgs = train_imgs[batch_start:batch_start + batch_size]
        print(f"\nБатч {i+1} ({len(batch_imgs)} изображений):")
        for j, img_path in enumerate(batch_imgs[:5]):
            label_path = ds_dir / 'train' / 'labels' / f"{img_path.stem}.txt"
            ann_count = (
                len([l for l in label_path.read_text(encoding='utf-8').strip().split('\n') if l.strip().startswith('0 ')])
                if label_path.exists()
                else 0
            )
            print(f"  {j+1}. {img_path.name} - аннотаций класса 0: {ann_count}")


def main():
    dataset_root = Path(__file__).parent.parent.parent / 'dataset'
    ds_dir = Path(__file__).parent / 'dataset'

    images = collect_border_images(dataset_root)
    if not images:
        raise SystemExit('Нет изображений с разметкой класса 0 (border). Проверьте dataset/*/raw/**/labels.')

    n, train_end, test_end = build_yolo_dataset(images, ds_dir)
    _ = (n, train_end, test_end)  # оставлено как подсказка для будущих логов

    setup_mlflow(Path(__file__).parent.parent)

    epochs_1 = int(os.environ.get('BORDERS_EPOCHS_1', 200))
    batch_size_1 = int(os.environ.get('BORDERS_BATCH_1', 8))
    model_name_1 = os.environ.get('BORDERS_MODEL_1', 'yolo11s-seg.pt')
    run_name_1 = os.environ.get(
        'BORDERS_RUN_NAME_1',
        f'borders-{model_name_1.replace(".pt", "")}-batch{batch_size_1}',
    )

    run_training(ds_dir, epochs_1, batch_size_1, model_name_1, run_name_1, imgsz=640)
    print_batch_examples(ds_dir, batch_size_1)


if __name__ == '__main__':
    main()
