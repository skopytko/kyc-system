import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

def yolo_segmentation_to_points(label_path, img_shape):
    h, w = img_shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 5:
                coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
                coords = coords.reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                return coords.astype(np.int32)
    return None

def calculate_iou(pred_mask, gt_contour, img_shape):
    h, w = img_shape[:2]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(gt_mask, [gt_contour], 0, 255, -1)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def match_masks_to_gt(pred_masks, gt_contours, img_shape):
    pairs = []
    used_pred = set()
    used_gt = set()
    ious = []
    for _ in range(min(len(pred_masks), len(gt_contours))):
        best_iou = -1
        best_pair = (None, None)
        for i, pred_mask in enumerate(pred_masks):
            if i in used_pred:
                continue
            mask_resized = cv2.resize(pred_mask, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
            for j, gt in enumerate(gt_contours):
                if j in used_gt:
                    continue
                iou = calculate_iou(mask_bin, gt, img_shape)
                if iou > best_iou:
                    best_iou = iou
                    best_pair = (i, j, mask_bin)
        if best_pair[0] is not None and best_pair[1] is not None:
            used_pred.add(best_pair[0])
            used_gt.add(best_pair[1])
            ious.append(best_iou)
            pairs.append((best_pair[0], best_pair[1], best_pair[2]))
    return ious, pairs

def main():
    test_images_dir = Path('../../dataset/dataset_for_seal_detector/test/images')
    test_labels_dir = Path('../../dataset/dataset_for_seal_detector/test/labels')
    viz_dir = Path('seals_viz_yolo_pt')
    viz_dir.mkdir(exist_ok=True)
    model_path = 'runs/segment/train2/weights/best.pt'
    model = YOLO(model_path)
    ious = []
    successful = 0
    total = 0
    for img_name in tqdm(os.listdir(test_images_dir)):
        if not (img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')):
            continue
        img_path = test_images_dir / img_name
        label_path = test_labels_dir / (os.path.splitext(img_name)[0] + '.txt')
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # Инференс через ultralytics YOLOv8
        results = model(img, verbose=False)[0]
        # Сортируем по conf и берём лучшие
        if hasattr(results, 'masks') and results.masks is not None:
            confs = results.boxes.conf.cpu().numpy()
            idxs = np.argsort(confs)[::-1][:5]  # Берём до 5 лучших печатей
            masks = results.masks.data.cpu().numpy()[idxs]
        else:
            continue
        # Получаем GT-контуры печатей
        gt_contours = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:
                    coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
                    coords = coords.reshape(-1, 2)
                    coords[:, 0] *= img.shape[1]
                    coords[:, 1] *= img.shape[0]
                    gt_contour = coords.astype(np.int32)
                    gt_contours.append(gt_contour)
        if len(gt_contours) == 0 or len(masks) == 0:
            continue
        # Жадное сопоставление масок и GT по максимальному IoU
        img_ious, pairs = match_masks_to_gt(masks, gt_contours, img.shape)
        viz_img = img.copy()
        colors = [(0,0,255), (255,0,0), (0,255,255), (255,255,0), (255,0,255)]  # Разные цвета для печатей
        for k, (pred_idx, gt_idx, mask_bin) in enumerate(pairs):
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(viz_img, contours, 0, colors[k%len(colors)], 2)
            # Подписываем IoU
            cv2.putText(viz_img, f"Seal{k+1} IoU: {img_ious[k]:.3f}", (10, 30 + k*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[k%len(colors)], 2)
        # Визуализируем GT-контуры
        for i, gt in enumerate(gt_contours):
            cv2.drawContours(viz_img, [gt], 0, (0,255,0), 2)
        if img_ious:
            ious.extend(img_ious)
            successful += 1
        cv2.imwrite(str(viz_dir / img_name), viz_img)
        total += 1
    if ious:
        print(f"\nСредний IoU: {np.mean(ious):.3f}")
        print(f"Медианный IoU: {np.median(ious):.3f}")
        print(f"Обработано изображений: {total}, успешно сравнили: {successful}")
        print(f"Визуализации сохранены в {viz_dir}/")
    else:
        print("Не удалось вычислить метрики - нет успешных детекций")

if __name__ == "__main__":
    main() 