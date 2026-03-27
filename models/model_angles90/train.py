import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from tqdm import tqdm
import timm
import numpy as np
import matplotlib.pyplot as plt
import mlflow

class Angles90Dataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths) * 4

    def __getitem__(self, idx):
        img_idx = idx // 4
        angle_idx = idx % 4
        img_path = self.img_paths[img_idx]
        img = Image.open(img_path).convert('RGB')
        if angle_idx * 90 != 0:
            img = img.rotate(angle_idx * 90, expand=True)
        if self.transform:
            img = self.transform(img)
        return img, angle_idx

def find_raw_folders(dataset_root):
    raw_folders = []
    for root, dirs, files in os.walk(dataset_root):
        if 'raw' in dirs:
            raw_folders.append(os.path.join(root, 'raw'))
    return sorted(raw_folders)

def collect_images_from_folder(folder_path):
    return sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

def split_dataset_by_folder(img_paths, train_ratio=0.8, val_ratio=0.1, random_state=42):
    generator = torch.Generator().manual_seed(random_state)
    n = len(img_paths)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return (
        [img_paths[i] for i in indices[:n_train]],
        [img_paths[i] for i in indices[n_train:n_train + n_val]],
        [img_paths[i] for i in indices[n_train + n_val:]]
    )

def main():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset')
    runs_dir = current_dir / 'runs'
    random_state = int(os.environ.get('ANGLES90_RANDOM_STATE', 42))
    epochs = int(os.environ.get('ANGLES90_EPOCHS', 50))
    batch_size = int(os.environ.get('ANGLES90_BATCH', 8))
    lr = float(os.environ.get('ANGLES90_LR', 1e-3))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Настраиваем MLflow
    mlruns_dir = project_root / 'mlruns'
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(str(mlruns_dir.resolve()))
    experiment_name = os.environ.get('ANGLES90_EXPERIMENT', 'angles90-belarus-passport-all-pages')
    mlflow.set_experiment(experiment_name)
    run_name = os.environ.get('ANGLES90_RUN_NAME', f'angles90-batch{batch_size}-lr{lr}')

    # Находим все папки raw
    raw_folders = find_raw_folders(dataset_root)
    print(f'Found {len(raw_folders)} raw folders:')
    for folder in raw_folders:
        print(f'  {folder}')
    
    # Собираем все изображения
    all_images = []
    for folder in raw_folders:
        images = collect_images_from_folder(folder)
        all_images.extend(images)
    
    print(f'\nFound {len(all_images)} total images')
    
    if len(all_images) == 0:
        print('Error: No images found in raw folders!')
        return
    
    # Разделяем на train/val/test
    train_paths, val_paths, test_paths = split_dataset_by_folder(all_images, random_state=random_state)
    
    print(f'\nDataset split:')
    print(f'  Train: {len(train_paths)} images ({len(train_paths) * 4} samples with rotations)')
    print(f'  Val: {len(val_paths)} images ({len(val_paths) * 4} samples with rotations)')
    print(f'  Test: {len(test_paths)} images ({len(test_paths) * 4} samples with rotations)')
    
    # Трансформации
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Создаем датасеты (каждое изображение будет повернуто на 0°, 90°, 180°, 270°)
    train_dataset = Angles90Dataset(train_paths, transform)
    val_dataset = Angles90Dataset(val_paths, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Модель EfficientNet-B3 для 4 классов (0°, 90°, 180°, 270°)
    model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=4).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_wts = None
    
    print(f'\nTraining parameters:')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rate: {lr}')
    print(f'  Device: {device}')
    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Val samples: {len(val_dataset)}')
    print(f'  Model: EfficientNet-B3')
    print(f'  Classes: 0°, 90°, 180°, 270°')
    
    best_model_path = runs_dir / 'best_efficientnet_b3_angles90.pth'
    last_model_path = runs_dir / 'last_efficientnet_b3_angles90.pth'
    train_curves_path = runs_dir / 'train_curves.png'

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'optimizer': 'Adam',
            'scheduler': 'ReduceLROnPlateau',
            'model_name': 'efficientnet_b3',
            'train_images': len(train_paths),
            'val_images': len(val_paths),
            'test_images': len(test_paths),
            'total_raw_images': len(all_images),
            'device': str(device),
            'random_state': random_state
        })

        # Обучение
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} - train'):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds.cpu() == labels.cpu()).sum().item()
                total += imgs.size(0)
            train_loss /= len(train_loader.dataset)
            train_acc = correct / total if total > 0 else 0

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} - val'):
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = outputs.argmax(dim=1)
                    correct += (preds.cpu() == labels.cpu()).sum().item()
                    total += imgs.size(0)
            val_loss /= len(val_loader.dataset)
            val_acc = correct / total if total > 0 else 0

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            scheduler.step(val_loss)

            mlflow.log_metrics(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc
                },
                step=epoch + 1
            )

            print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_wts = model.state_dict().copy()
                torch.save(best_model_wts, best_model_path)
                print(f'  -> Best model saved!')
                mlflow.log_metric('best_val_loss', best_val_loss)
                mlflow.log_metric('best_epoch', epoch + 1)

        # Сохраняем последнюю модель
        torch.save(model.state_dict(), last_model_path)

        # Визуализация кривых обучения
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(train_curves_path, dpi=150)
        plt.close()

        # Логируем финальные метрики и артефакты
        mlflow.log_metric('final_train_loss', history['train_loss'][-1])
        mlflow.log_metric('final_val_loss', history['val_loss'][-1])
        mlflow.log_metric('final_train_acc', history['train_acc'][-1])
        mlflow.log_metric('final_val_acc', history['val_acc'][-1])

        if best_model_path.exists():
            mlflow.log_artifact(str(best_model_path), artifact_path='models')
        if last_model_path.exists():
            mlflow.log_artifact(str(last_model_path), artifact_path='models')
        if train_curves_path.exists():
            mlflow.log_artifact(str(train_curves_path), artifact_path='plots')

    print(f'\nBest model saved to: {best_model_path}')
    print(f'Last model saved to: {last_model_path}')
    print(f'Training curves saved to: {train_curves_path}')

if __name__ == '__main__':
    main()
