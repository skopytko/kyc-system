import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt

def prepare_test_dataset(dataset_path, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Подготавливает тестовый датасет."""
    images_dir = dataset_path
    labels_dir = os.path.join(dataset_path, 'labels')
    
    image_files = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(label_path):
                image_files.append(f)
    
    generator = torch.Generator().manual_seed(random_state)
    n = len(image_files)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    
    test_files = [image_files[i] for i in indices[n_train + n_val:]]
    
    test_images = []
    for img_file in test_files:
        test_images.append(os.path.join(images_dir, img_file))
    
    return test_images

def load_class_names(labels_dir):
    """Загружает имена классов из classes.txt"""
    classes_file = os.path.join(labels_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = [
            'authority', 'authority2', 'code_of_issuing', 'date_of_birth',
            'date_of_expiry', 'date_of_issue', 'identification_no', 'names',
            'nationality', 'passport_no', 'photo', 'place_of_birth',
            'sex', 'signature', 'signature2', 'surname', 'type'
        ]
    return class_names

def process_image(model, img_path, output_path, class_names, conf_threshold=0.04):
    """Обрабатывает изображение и рисует bbox."""
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img, conf=conf_threshold, verbose=False)
    
    result_img = img_rgb.copy()
    box_count = 0
    detected_classes = {}
    
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        boxes_data = results[0].boxes.data.cpu().numpy()
        
        # Группируем по классам и выбираем лучший bbox для каждого класса
        best_boxes = {}  # {class_id: (box, conf)}
        
        for box in boxes_data:
            x1, y1, x2, y2 = box[:4].astype(int)
            conf = float(box[4]) if len(box) > 4 else 1.0
            cls = int(box[5]) if len(box) > 5 else 0
            
            # Сохраняем только лучший bbox для каждого класса
            if cls not in best_boxes or conf > best_boxes[cls][1]:
                best_boxes[cls] = (box, conf)
        
        # Цвета для разных классов
        colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
        
        # Рисуем только лучшие bbox для каждого класса
        for cls, (box, conf) in best_boxes.items():
            x1, y1, x2, y2 = box[:4].astype(int)
            
            # Получаем цвет для класса
            color = (colors[cls % len(colors)] * 255).astype(int).tolist()
            color = tuple(color[:3])  # RGB
            
            # Рисуем bbox
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # Подготовка текста
            class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            label = f'{class_name} {conf:.2f}'
            
            # Размер текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Фон для текста
            cv2.rectangle(
                result_img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Текст
            cv2.putText(
                result_img,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            # Статистика по классам
            detected_classes[class_name] = conf
            box_count += 1
    
    # Создаем визуализацию
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title(f'Original: {os.path.basename(img_path)}', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(result_img)
    title = f'Detected: {box_count} unique classes'
    if detected_classes:
        classes_str = ', '.join([f'{k}({v:.2f})' for k, v in sorted(detected_classes.items())])
        title += f'\n{classes_str}'
    axes[1].set_title(title, fontsize=10)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True

def main():
    model_path = os.path.join(os.path.dirname(__file__), 'runs', 'train', 'weights', 'best.pt')
    dataset_path = os.path.join(os.path.dirname(__file__), '../../../dataset/belarus/passport/data_page/cropped')
    output_dir = os.path.join(os.path.dirname(__file__), 'runs', 'train', 'test_visualizations')
    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))
    conf_threshold = float(os.environ.get('TEXTFIELDS_CONF', 0.04))
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Please train the model first using train.py')
        return
    
    print(f'Loading model from: {model_path}')
    model = YOLO(model_path)
    
    # Загружаем имена классов
    labels_dir = os.path.join(dataset_path, 'labels')
    class_names = load_class_names(labels_dir)
    print(f'Loaded {len(class_names)} classes: {class_names[:5]}...')
    
    # Собираем тестовые изображения
    test_images = prepare_test_dataset(dataset_path, random_state=random_state)
    
    print(f'Processing {len(test_images)} test images...')
    
    for img_path in tqdm(test_images, desc='Processing'):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(output_dir, f'{img_name}_detection.png')
        process_image(model, img_path, output_path, class_names, conf_threshold)
    
    print(f'\nAll results saved to: {output_dir}')

if __name__ == '__main__':
    main()

