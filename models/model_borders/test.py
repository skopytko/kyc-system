import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def resolve_weights(model_dir: Path) -> Path:
    env = os.environ.get('BORDERS_WEIGHTS')
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise SystemExit(f'BORDERS_WEIGHTS задан, но файл не найден: {p}')

    candidates = [
        model_dir / 'runs' / 'weights' / 'best.pt',
        model_dir / 'runs' / 'segment' / 'train' / 'weights' / 'best.pt',
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise SystemExit(
        'Не найден best.pt. Обучите модель (python train.py) или задайте BORDERS_WEIGHTS=/path/to/best.pt'
    )


def _read_data_yaml_path_and_split_dir(data_yaml: Path, split: str) -> tuple[Path, Path]:
    """Корень датасета (path:) и каталог images для split (например test/images)."""
    root = None
    split_rel = None
    for line in data_yaml.read_text(encoding='utf-8').splitlines():
        s = line.strip()
        if s.startswith('path:'):
            root = Path(s.split(':', 1)[1].strip())
        elif s.startswith(f'{split}:'):
            split_rel = Path(s.split(':', 1)[1].strip())
    if root is None or split_rel is None:
        raise SystemExit(f'В {data_yaml} не найдены path: и {split}:')
    return root, root / split_rel


def _binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a > 0, b > 0).sum())
    union = int(np.logical_or(a > 0, b > 0).sum())
    if union == 0:
        return 1.0
    return inter / union


def _gt_masks_from_label(label_path: Path, h: int, w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if not label_path.is_file():
        return masks
    for line in label_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line.startswith('0 '):
            continue
        nums = list(map(float, line.split()[1:]))
        if len(nums) < 6:
            continue
        xs = nums[0::2]
        ys = nums[1::2]
        pts = np.stack([np.array(xs) * w, np.array(ys) * h], axis=1)
        m = np.zeros((h, w), dtype=np.uint8)
        pi = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pi], 1)
        masks.append(m)
    return masks


def _pred_masks_from_result(r) -> list[np.ndarray]:
    if r.masks is None:
        return []
    h, w = r.orig_shape
    out: list[np.ndarray] = []
    for xy in r.masks.xy:
        m = np.zeros((h, w), dtype=np.uint8)
        if len(xy) < 3:
            out.append(m)
            continue
        pi = np.round(xy).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pi], 1)
        out.append(m)
    return out


def _union_masks(masks: list[np.ndarray]) -> np.ndarray:
    u = masks[0].astype(bool)
    for m in masks[1:]:
        u = np.logical_or(u, m > 0)
    return u.astype(np.uint8)


def _instance_ious_greedy(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> list[float]:
    """По каждому GT — лучший неиспользованный pred (жадно), IoU по пересечению масок."""
    if not gt_masks:
        return []
    used: set[int] = set()
    order = sorted(range(len(gt_masks)), key=lambda i: int(gt_masks[i].sum()), reverse=True)
    ious: list[float] = []
    for gi in order:
        best = 0.0
        best_j: int | None = None
        for j, pm in enumerate(pred_masks):
            if j in used:
                continue
            v = _binary_iou(gt_masks[gi], pm)
            if v > best:
                best = v
                best_j = j
        if best_j is not None:
            used.add(best_j)
        ious.append(best)
    return ious


def compute_mask_miou(
    model: YOLO,
    data_yaml: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    conf: float,
    iou_nms: float,
    max_images: int | None,
) -> dict:
    _, images_dir = _read_data_yaml_path_and_split_dir(data_yaml, split)
    labels_dir = images_dir.parent / 'labels'
    exts = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG')
    paths: list[Path] = []
    for pat in exts:
        paths.extend(images_dir.glob(pat))
    paths = sorted({p.resolve() for p in paths})
    if max_images is not None:
        paths = paths[: max_images]

    all_ious: list[float] = []
    per_image_ious: list[float] = []
    n_skipped_no_gt = 0
    for r in model.predict(
        source=[str(p) for p in paths],
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou_nms,
        verbose=False,
        stream=True,
    ):
        p = Path(r.path).resolve()
        label_path = labels_dir / f'{p.stem}.txt'
        h, w = r.orig_shape
        gt = _gt_masks_from_label(label_path, h, w)
        if not gt:
            n_skipped_no_gt += 1
            continue
        pred = _pred_masks_from_result(r)
        all_ious.extend(_instance_ious_greedy(gt, pred))
        gt_u = _union_masks(gt)
        pred_u = _union_masks(pred) if pred else np.zeros((h, w), dtype=np.uint8)
        per_image_ious.append(_binary_iou(gt_u, pred_u))

    miou = float(np.mean(all_ious)) if all_ious else None
    miou_per_image = float(np.mean(per_image_ious)) if per_image_ious else None
    return {
        'miou': miou,
        'miou_per_image': miou_per_image,
        'miou_gt_instance_count': len(all_ious),
        'miou_images_scanned': len(paths),
        'miou_images_skipped_no_gt': n_skipped_no_gt,
        'miou_note': (
            'miou: средний IoU по каждому GT-инстансу класса 0 (жадное сопоставление с предсказаниями; '
            'нет пары — IoU=0). miou_per_image: для каждого изображения IoU(объединение GT-масок, '
            'объединение предсказанных масок), затем среднее по изображениям.'
        ),
    }


def _val_output_dir(model_dir: Path, metrics) -> Path:
    sd = Path(metrics.save_dir)
    return sd.resolve() if sd.is_absolute() else (model_dir / sd).resolve()


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, float):
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def main():
    model_dir = Path(__file__).parent.resolve()
    data_yaml = model_dir / 'dataset' / 'data.yaml'
    if not data_yaml.is_file():
        raise SystemExit(f'Нет {data_yaml} — сначала запустите train.py для сборки датасета.')

    weights = resolve_weights(model_dir)
    split = os.environ.get('BORDERS_VAL_SPLIT', 'test')
    imgsz = int(os.environ.get('BORDERS_IMGSZ', '640'))
    batch = int(os.environ.get('BORDERS_VAL_BATCH', '8'))
    device = os.environ.get('BORDERS_DEVICE', 'cpu')
    plots = os.environ.get('BORDERS_PLOTS', '1').lower() not in ('0', 'false', 'no')
    run_miou = os.environ.get('BORDERS_MIOU', '1').lower() not in ('0', 'false', 'no')
    miou_max = os.environ.get('BORDERS_MIOU_MAX_IMAGES')
    miou_max_i = int(miou_max) if miou_max else None
    conf = float(os.environ.get('BORDERS_CONF', '0.25'))
    iou_nms = float(os.environ.get('BORDERS_IOU_NMS', '0.7'))

    print(f'Веса: {weights}')
    print(f'data: {data_yaml}')
    print(f'split: {split}')

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=plots,
        project=str(model_dir / 'runs'),
        name=f'val_{split}',
        exist_ok=True,
    )

    out_dir = _val_output_dir(model_dir, metrics)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        'weights': str(weights),
        'data_yaml': str(data_yaml),
        'split': split,
    }

    if run_miou:
        print('Считаю mIoU по маскам (отдельный проход predict)...')
        miou_block = compute_mask_miou(
            model,
            data_yaml,
            split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            conf=conf,
            iou_nms=iou_nms,
            max_images=miou_max_i,
        )
        sanitized = _json_sanitize(miou_block)
        payload.update(sanitized)
        if miou_block.get('miou') is not None:
            print(f"mIoU (по GT-инстансам): {miou_block['miou']:.6f}")
        if miou_block.get('miou_per_image') is not None:
            print(f"mIoU (по изображениям, ∪GT vs ∪Pred): {miou_block['miou_per_image']:.6f}")

    payload['metrics'] = _json_sanitize(dict(metrics.results_dict))
    payload['speed'] = _json_sanitize(dict(metrics.speed))

    out_path = out_dir / 'metrics.json'
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Метрики записаны: {out_path}')


if __name__ == '__main__':
    main()
