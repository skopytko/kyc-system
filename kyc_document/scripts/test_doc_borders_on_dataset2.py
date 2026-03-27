from kyc_document.document_processing.pipeline_modules import DocDetector
import cv2
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path

# Пути к данным
JSON_PATH = '../../../Dataset2/rus_internalpassport_aug.json'
IMAGES_DIR = '../../../Dataset2/rus_internalpassport_aug'

def extract_doc_quad(region_list):
    """Извлекает doc_quad из списка регионов VIA."""
    for region in region_list:
        if region.get('region_attributes', {}).get('field_name') == 'doc_quad':
            shape = region.get('shape_attributes', {})
            if shape.get('name') == 'polygon':
                xs = shape['all_points_x']
                ys = shape['all_points_y']
                if len(xs) == 4 and len(ys) == 4:
                    return np.array(list(zip(xs, ys)), dtype=np.int32)
    return None

def extract_pred_quad(result):
    # Получаем сегмент (контур) документа
    docdet = result.get('DocDetector', {})
    segm = docdet.get('segm')
    if segm and len(segm) > 0:
        # Берём первый сегмент (основной контур)
        contour = np.array(segm[0], dtype=np.float32)
        # Аппроксимируем до 4 угловых точек
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2)
            return quad
        # Если не удалось аппроксимировать до 4 точек, возвращаем None
    return None

def mean_point_error(gt, pred):
    if gt is None or pred is None or len(gt) != 4 or len(pred) != 4:
        return None
    return float(np.mean(np.linalg.norm(gt - pred, axis=1)))

def draw_quad(img, quad, color, thickness=3):
    quad = quad.astype(int)
    for i in range(4):
        pt1 = tuple(quad[i])
        pt2 = tuple(quad[(i+1)%4])
        cv2.line(img, pt1, pt2, color, thickness)
    return img

def calculate_iou(pred_contour, gt_contour):
    """Вычисляет IoU между предсказанным и эталонным контуром."""
    # Определяем размер маски по максимуму координат
    all_points = np.vstack([pred_contour, gt_contour])
    max_x = int(np.max(all_points[:, 0])) + 10
    max_y = int(np.max(all_points[:, 1])) + 10
    pred_mask = np.zeros((max_y, max_x), dtype=np.uint8)
    gt_mask = np.zeros((max_y, max_x), dtype=np.uint8)
    cv2.drawContours(pred_mask, [pred_contour], 0, 255, -1)
    cv2.drawContours(gt_mask, [gt_contour], 0, 255, -1)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def main():
    # Абсолютные пути к данным
    annotations_file = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset/dataset_for_doc_detector/passport_aug.json")
    images_dir = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset/dataset_for_doc_detector/passport_aug")
    viz_dir = Path("borders_viz")
    viz_dir.mkdir(exist_ok=True)
    # Загружаем аннотации VIA
    with open(annotations_file, "r") as f:
        data = json.load(f)
    img_metadata = data['_via_img_metadata']
    detector = DocDetector()
    ious = []
    successful = 0
    total = 0
    for meta in tqdm(img_metadata.values()):
        filename = meta['filename']
        regions = meta['regions']
        gt_quad = extract_doc_quad(regions)
        if gt_quad is None:
            continue
        img_path = images_dir / filename
        if not img_path.exists():
            print(f"Изображение не найдено: {img_path}")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue
        result = detector.predict(img)
        if not result or "DocDetector" not in result:
            continue
        doc_result = result["DocDetector"]
        if not doc_result.get("segm"):
            continue
        pred_contours = doc_result["segm"]
        pred_contour = np.array(pred_contours[0], dtype=np.int32)
        pred_contour = cv2.approxPolyDP(pred_contour, 0.02 * cv2.arcLength(pred_contour, True), True)
        if len(pred_contour) != 4:
            continue
        # Приведение формы к (4,2)
        if pred_contour.ndim == 3:
            pred_contour = pred_contour[:, 0, :]
        if gt_quad.ndim == 3:
            gt_quad = gt_quad[:, 0, :]
        iou = calculate_iou(pred_contour, gt_quad)
        ious.append(iou)
        successful += 1
        # Визуализация
        viz_img = img.copy()
        cv2.drawContours(viz_img, [gt_quad], 0, (0, 255, 0), 2)
        cv2.drawContours(viz_img, [pred_contour], 0, (0, 0, 255), 2)
        cv2.putText(viz_img, f"IoU: {iou:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imwrite(str(viz_dir / filename), viz_img)
        total += 1
    if ious:
        print(f"\nСредний IoU: {np.mean(ious):.3f}")
        print(f"Медианный IoU: {np.median(ious):.3f}")
        print(f"Обработано изображений: {len(img_metadata)}, успешно сравнили: {successful}")
        print(f"Визуализации сохранены в {viz_dir}/")
    else:
        print("Не удалось вычислить метрики - нет успешных детекций")

if __name__ == "__main__":
    main() 