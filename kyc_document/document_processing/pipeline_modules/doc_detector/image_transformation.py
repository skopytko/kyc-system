from typing import Optional

import numpy as np
import cv2



def iou(bbox1: np.ndarray, bbox2: np.ndarray):
    """Compute intersection over union between two bboxes.

    Args:
        bbox1 (ndarray): First bounding box
        bbox2 (ndarray): Second bounding box

    Returns:
        IoU ratio value
    """
    area1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    #intersection
    x1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
    x2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
    y1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
    y2 = np.minimum(bbox1[..., 3], bbox2[..., 3])

    intersection = np.maximum(0, (x2-x1)) * np.maximum(0, (y2-y1))

    ratio = intersection / (area1 + area2 - intersection)

    return ratio

def xywh2xyxy(x):
    """Convert bboxes from (x,y,w,h) to (x1,y1,x2,y2) format.

    Args:
        x (ndarray): Bounding boxes in x,y,w,h format

    Returns:
        Converted bboxes in x1,y1,x2,y2 format
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect


def _quad_from_hull(hull) -> Optional[np.ndarray]:
    """Четыре угла документа под перспективу: не прямоугольник minAreaRect, а четырёхугольник по контуру.

    Сначала approxPolyDP на выпуклой оболочке (впадины маски не тянут углы внутрь).
    minAreaRect — только запасной вариант, если 4 вершины получить не удалось.
    """
    hull_area = float(cv2.contourArea(hull))
    if hull_area < 1.0:
        return None
    peri = float(cv2.arcLength(hull, True))
    if peri < 1e-6:
        return None

    best_approx = None
    best_n = None
    for eps in [
        0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.22, 0.28, 0.35
    ]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        n = len(approx)
        if n == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            qa = float(cv2.contourArea(quad))
            if hull_area > 1.0 and qa < 0.75 * hull_area:
                continue
            return quad
        if n > 4:
            best_approx = approx
            best_n = n

    if best_approx is not None and best_n is not None and best_n > 4:
        for extra in [0.4, 0.45, 0.5, 0.55, 0.6]:
            approx = cv2.approxPolyDP(hull, extra * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    rect = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    return rect


def fix_perspective(img: np.ndarray, segments: np.ndarray):
    h, w = img.shape[:2]
    cnt_img = img.copy()
    warped_with_pos = []
    for i, segm in enumerate(segments):
        segm = np.clip(segm, [0, 0], [w - 1, h - 1])
        segm_int = segm.astype(np.int32).reshape(-1, 1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [segm_int], -1, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.drawContours(cnt_img, [cnt], -1, (0, 0, 255), 2)
            hull = cv2.convexHull(cnt)
            quad = _quad_from_hull(hull)
            if quad is None:
                continue
            poly_draw = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(cnt_img, [poly_draw], True, (255, 0, 0), 4)
            rect = order_points(quad)
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
            warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            center_y = quad[:, 1].mean()
            center_x = quad[:, 0].mean()
            warped_with_pos.append((center_y, center_x, warped))
    
    if warped_with_pos:
        # Определяем направление расположения областей
        if len(warped_with_pos) == 2:
            y1, x1, _ = warped_with_pos[0]
            y2, x2, _ = warped_with_pos[1]
            
            # Если разница по Y больше чем по X, то области расположены вертикально
            if abs(y2 - y1) > abs(x2 - x1):
                # Вертикальное расположение (сверху и снизу) - используем vstack
                warped_with_pos.sort(key=lambda x: x[0])  # сортируем по Y
                max_w = max(w.shape[1] for _, _, w in warped_with_pos)
                warped_imgs = [cv2.resize(w, (max_w, int(w.shape[0]*max_w/w.shape[1]))) for _, _, w in warped_with_pos]
                warpimg = np.vstack(warped_imgs)
            else:
                # Горизонтальное расположение (слева и справа) - используем hstack
                warped_with_pos.sort(key=lambda x: x[1])  # сортируем по X
                max_h = max(w.shape[0] for _, _, w in warped_with_pos)
                warped_imgs = [cv2.resize(w, (int(w.shape[1]*max_h/w.shape[0]), max_h)) for _, _, w in warped_with_pos]
                warpimg = np.hstack(warped_imgs)
        else:
            # Если областей больше 2, используем старую логику
            warped_with_pos.sort(key=lambda x: x[0])
            max_w = max(w.shape[1] for _, _, w in warped_with_pos)
            warped_imgs = [cv2.resize(w, (max_w, int(w.shape[0]*max_w/w.shape[1]))) for _, _, w in warped_with_pos]
            warpimg = np.vstack(warped_imgs)
    else:
        warpimg = img
    return warpimg, cnt_img



