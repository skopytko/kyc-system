import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# --- Аугментации ---
def shift_image_and_quad(img, quad, max_shift=50):
    h, w = img.shape[:2]
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    shifted_quad = quad + np.array([dx, dy])
    return shifted_img, shifted_quad

def rotate_image_and_quad(img, quad, max_angle=15):
    h, w = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    quad_homo = np.hstack([quad, np.ones((4,1))])
    rotated_quad = (M @ quad_homo.T).T
    return rotated_img, rotated_quad.astype(int)

def blur_image(img, quad=None, max_ksize=7):
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0), quad

def scale_stretch_image_and_quad(img, quad, scale_range=(0.9, 1.1), stretch_range=(0.9, 1.1)):
    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)
    stretch_x = random.uniform(*stretch_range)
    stretch_y = random.uniform(*stretch_range)
    M = np.array([[scale*stretch_x, 0, 0], [0, scale*stretch_y, 0]], dtype=np.float32)
    scaled_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    quad_homo = np.hstack([quad, np.ones((4,1))])
    scaled_quad = (M @ quad_homo.T).T
    return scaled_img, scaled_quad.astype(int)

def change_brightness(img, quad=None, max_delta=40):
    delta = random.randint(-max_delta, max_delta)
    img = img.astype(np.int16) + delta
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, quad

def add_glare(img, quad=None, max_alpha=0.7):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    center = (random.randint(0, w), random.randint(0, h))
    axes = (random.randint(w//8, w//3), random.randint(h//8, h//3))
    angle = random.uniform(0, 360)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 1, -1)
    alpha = random.uniform(0.3, max_alpha)
    glare = np.full_like(img, 255)
    glare = cv2.addWeighted(img, 1, glare, alpha, 0)
    mask3 = np.stack([mask]*3, axis=-1)
    glare_img = img * (1-mask3) + glare * mask3
    return glare_img.astype(np.uint8), quad

def add_noise(img, quad=None, noise_level=20):
    noise = np.random.normal(0, noise_level, img.shape).astype(np.int16)
    noisy = img.astype(np.int16) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy, quad

# --- Обработка аннотаций VIA ---
def extract_doc_quad(region_list):
    for region in region_list:
        if region.get('region_attributes', {}).get('field_name') == 'doc_quad':
            shape = region.get('shape_attributes', {})
            if shape.get('name') == 'polygon':
                xs = shape['all_points_x']
                ys = shape['all_points_y']
                if len(xs) == 4 and len(ys) == 4:
                    return np.array(list(zip(xs, ys)), dtype=np.int32)
    return None

def update_doc_quad(region_list, new_quad):
    for region in region_list:
        if region.get('region_attributes', {}).get('field_name') == 'doc_quad':
            region['shape_attributes']['all_points_x'] = [int(x) for x, y in new_quad]
            region['shape_attributes']['all_points_y'] = [int(y) for x, y in new_quad]
            break

# --- Главная функция ---
def main():
    src_dir = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset2/rus_internalpassport")
    src_json = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset2/rus_internalpassport.json")
    out_dir = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset2/rus_internalpassport_aug")
    out_dir.mkdir(exist_ok=True)
    out_json = Path("/Users/kopytko/Yandex.Disk.localized/TRIUMPH/ocr-project/Dataset2/rus_internalpassport_aug.json")

    with open(src_json, "r") as f:
        data = json.load(f)
    img_metadata = data['_via_img_metadata']
    new_img_metadata = {}

    # Список аугментаций: (имя, функция, геометрическая ли)
    aug_list = [
        ('shift', shift_image_and_quad, True),
        ('rotate', rotate_image_and_quad, True),
        ('blur', blur_image, False),
        ('scale', scale_stretch_image_and_quad, True),
        ('brightness', change_brightness, False),
        ('glare', add_glare, False),
        ('noise', add_noise, False),
    ]
    N_COPIES = 4
    for meta_key, meta in tqdm(img_metadata.items()):
        filename = meta['filename']
        regions = meta['regions']
        quad = extract_doc_quad(regions)
        if quad is None:
            continue
        img_path = src_dir / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Не удалось загрузить {img_path}")
            continue
        for i in range(N_COPIES):
            aug_names = []
            aug_img = img.copy()
            aug_quad = quad.copy()
            # Случайно выбираем 3-4 разных аугментации
            n_augs = random.randint(3, 4)
            aug_choices = random.sample(aug_list, n_augs)
            for aug_name, aug_func, is_geom in aug_choices:
                if is_geom:
                    aug_img, aug_quad = aug_func(aug_img, aug_quad)
                else:
                    aug_img, _ = aug_func(aug_img, aug_quad)
                aug_names.append(aug_name)
            aug_filename = f"{os.path.splitext(filename)[0]}_aug_{'_'.join(aug_names)}_{i}.jpg"
            cv2.imwrite(str(out_dir / aug_filename), aug_img)
            # Копируем аннотацию и обновляем quad
            new_meta = json.loads(json.dumps(meta))  # глубокая копия
            new_meta['filename'] = aug_filename
            update_doc_quad(new_meta['regions'], aug_quad)
            new_key = f"{aug_filename}{random.randint(100000,999999)}"
            new_img_metadata[new_key] = new_meta
    # Сохраняем новый json
    new_data = data.copy()
    new_data['_via_img_metadata'] = new_img_metadata
    with open(out_json, "w") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"Готово! Аугментированные изображения сохранены в {out_dir}, аннотации — в {out_json}")

if __name__ == "__main__":
    main() 