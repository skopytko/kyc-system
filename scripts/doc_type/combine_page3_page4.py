import os
import cv2
from pathlib import Path
import numpy as np

def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_CUBIC)

page3_dir = Path('dataset/dataset_passport_page/page3')
page4_dir = Path('dataset/dataset_passport_page/page4')
out_dir = Path('dataset/dataset_for_doc_type/passport')
out_dir.mkdir(parents=True, exist_ok=True)

page3_files = sorted([f for f in os.listdir(page3_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
page4_files = sorted([f for f in os.listdir(page4_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

idx = 0
for f3 in page3_files:
    img3 = cv2.imread(str(page3_dir / f3))
    if img3 is None:
        continue
    for f4 in page4_files:
        img4 = cv2.imread(str(page4_dir / f4))
        if img4 is None:
            continue
        max_width = max(img3.shape[1], img4.shape[1])
        img3_resized = resize_to_width(img3, max_width)
        img4_resized = resize_to_width(img4, max_width)
        combined = np.vstack([img3_resized, img4_resized])
        out_name = f"passport_centerfold_{idx:04d}.png"
        cv2.imwrite(str(out_dir / out_name), combined)
        idx += 1
print(f"Готово! Сохранено {idx} файлов в {out_dir}") 