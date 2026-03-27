import os
import yaml
import torch
from ultralytics import YOLO

def prepare_yolo_dataset(dataset_path, output_dir, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """Подготавливает датасет для обучения YOLO из папки cropped."""
    images_dir = dataset_path
    labels_dir = os.path.join(dataset_path, 'labels')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Собираем все изображения с соответствующими labels
    image_files = []
    for f in sorted(os.listdir(images_dir)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(label_path):
                image_files.append(f)
    
    if len(image_files) == 0:
        raise ValueError("No images with labels found!")
    
    print(f"Found {len(image_files)} images with labels")
    
    # Разделяем на train/val/test
    generator = torch.Generator().manual_seed(random_state)
    n = len(image_files)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    
    train_files = [image_files[i] for i in indices[:n_train]]
    val_files = [image_files[i] for i in indices[n_train:n_train + n_val]]
    test_files = [image_files[i] for i in indices[n_train + n_val:]]
    
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")
    
    # Создаем структуру папок
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # Создаем символические ссылки
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split, files in splits.items():
        for img_file in files:
            img_src = os.path.abspath(os.path.join(images_dir, img_file))
            label_src = os.path.abspath(os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt'))
            
            img_link = os.path.join(output_dir, 'images', split, img_file)
            label_link = os.path.join(output_dir, 'labels', split, os.path.splitext(img_file)[0] + '.txt')
            
            if os.path.exists(img_link) or os.path.islink(img_link):
                os.remove(img_link)
            if os.path.exists(label_link) or os.path.islink(label_link):
                os.remove(label_link)
            
            os.symlink(img_src, img_link)
            os.symlink(label_src, label_link)
    
    # Читаем классы из classes.txt
    classes_file = os.path.join(labels_dir, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: используем стандартные классы белорусского паспорта
        class_names = [
            'authority', 'authority2', 'code_of_issuing', 'date_of_birth',
            'date_of_expiry', 'date_of_issue', 'identification_no', 'names',
            'nationality', 'passport_no', 'photo', 'place_of_birth',
            'sex', 'signature', 'signature2', 'surname', 'type'
        ]
    
    # Создаем data.yaml
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path': os.path.abspath(output_dir),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }, f, default_flow_style=False)
    
    print(f"Dataset prepared. Classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    
    return yaml_path

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), '../../../dataset/belarus/passport_1996/data_page/cropped')
    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    dataset_dir = os.path.join(runs_dir, 'dataset')
    
    random_state = int(os.environ.get('TEXTFIELDS_RANDOM_STATE', 42))
    epochs = int(os.environ.get('TEXTFIELDS_EPOCHS', 100))
    batch_size = int(os.environ.get('TEXTFIELDS_BATCH', 8))
    imgsz = int(os.environ.get('TEXTFIELDS_IMGSZ', 640))
    model_size = os.environ.get('TEXTFIELDS_MODEL', 'yolo11n.pt')
    device = os.environ.get('TEXTFIELDS_DEVICE', '0' if torch.cuda.is_available() else 'cpu')
    patience = int(os.environ.get('TEXTFIELDS_PATIENCE', 20))
    
    if os.path.exists(dataset_dir):
        data_yaml = os.path.join(dataset_dir, 'data.yaml')
        print(f"Using existing dataset: {data_yaml}")
    else:
        os.makedirs(runs_dir, exist_ok=True)
        print(f"Preparing dataset from: {dataset_path}")
        data_yaml = prepare_yolo_dataset(dataset_path, dataset_dir, random_state=random_state)
    
    print(f"\nTraining parameters:")
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
        name='train',
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

