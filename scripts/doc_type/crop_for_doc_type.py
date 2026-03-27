import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO

def order_points(pts):
    # Упорядочить точки: [tl, tr, br, bl]
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def process_split(split_name):
    in_dir = Path(f'dataset/dataset_for_doc_detector/yolo_dataset/{split_name}/images')
    out_dir = Path(f'dataset/dataset_for_doc_type/{split_name}/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = 'russian_docs_ocr/document_processing/models/Borders/best.pt'
    model = YOLO(model_path)
    for img_name in tqdm(os.listdir(in_dir), desc=split_name):
        if not (img_name.endswith('.jpg') or img_name.endswith('.jpeg') or img_name.endswith('.png')):
            continue
        img_path = in_dir / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        results = model(img, verbose=False)[0]
        if hasattr(results, 'masks') and results.masks is not None:
            masks = results.masks.data.cpu().numpy()
        else:
            continue
        mask_sum = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for mask in masks:
            mask_bin = (cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
            mask_sum = np.logical_or(mask_sum, mask_bin)
        mask_sum = (mask_sum * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_sum, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest_contour, True)
        quad = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        if len(quad) != 4:
            continue
        quad = quad[:, 0, :] if quad.ndim == 3 else quad
        warped = four_point_transform(img, quad.astype(np.float32))
        out_path = out_dir / img_name
        cv2.imwrite(str(out_path), warped)

def main():
    for split in ['train', 'val', 'test']:
        process_split(split)

if __name__ == "__main__":
    main() 