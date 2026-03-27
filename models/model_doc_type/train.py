import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V3_Small_Weights
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

class DocTypeDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def find_cropped_folders(dataset_root):
    """Находит все папки cropped в dataset (belarus, russia, usa)."""
    from classes_config import cropped_path_to_class
    cropped_folders = []
    for root, dirs, files in os.walk(dataset_root):
        if 'cropped' in dirs:
            cropped_path = os.path.join(root, 'cropped')
            if cropped_path_to_class(cropped_path) is not None:
                cropped_folders.append(cropped_path)
    return sorted(cropped_folders)

def collect_images_from_cropped(cropped_path):
    """Собирает все изображения из папки cropped."""
    images = []
    if not os.path.exists(cropped_path):
        return images
    
    for f in sorted(os.listdir(cropped_path)):
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(os.path.join(cropped_path, f))
    return images

def split_dataset_by_folder(img_paths, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Разделение выборки: 80% train, 10% val, 10% test."""
    generator = torch.Generator().manual_seed(random_state)
    n = len(img_paths)
    indices = torch.randperm(n, generator=generator).tolist()
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    return (
        [img_paths[i] for i in indices[:n_train]],
        [img_paths[i] for i in indices[n_train:n_train + n_val]],
        [img_paths[i] for i in indices[n_train + n_val:]]  # test = оставшиеся ~10%
    )

def main():
    dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset')
    runs_dir = os.path.join(os.path.dirname(__file__), 'runs')
    random_state = int(os.environ.get('DOCTYPE_RANDOM_STATE', 42))
    epochs = int(os.environ.get('DOCTYPE_EPOCHS', 50))
    batch_size = int(os.environ.get('DOCTYPE_BATCH', 8))
    lr = float(os.environ.get('DOCTYPE_LR', 1e-3))
    patience = int(os.environ.get('DOCTYPE_PATIENCE', 5))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Находим все папки cropped
    cropped_folders = find_cropped_folders(dataset_root)
    print(f'Found {len(cropped_folders)} cropped folders:')
    for folder in cropped_folders:
        print(f'  {folder}')
    
    # Собираем изображения и определяем классы
    from classes_config import cropped_path_to_class, CLASS_NAMES, N_CLASSES

    # (img_path, cls_id, cropped_path) — cropped_path для отчёта по подпапкам
    all_images = []
    for cropped_path in cropped_folders:
        cls_id = cropped_path_to_class(cropped_path)
        if cls_id is None:
            continue
        images = collect_images_from_cropped(cropped_path)
        for img_path in images:
            all_images.append((img_path, cls_id, cropped_path))

    class_to_items = {}  # cls_id -> [(img_path, cropped_path), ...]
    for img_path, cls_id, cropped_path in all_images:
        class_to_items.setdefault(cls_id, []).append((img_path, cropped_path))

    max_per_class = int(os.environ.get('DOCTYPE_MAX_PER_CLASS', 0))
    if max_per_class > 0:
        rng = np.random.default_rng(random_state)
        for cls_id in range(N_CLASSES):
            items = class_to_items.get(cls_id, [])
            if len(items) <= max_per_class:
                continue
            reg_items = [(p, cp) for p, cp in items if 'registration_page' in cp]
            other_items = [(p, cp) for p, cp in items if 'registration_page' not in cp]
            n_reg = len(reg_items)
            n_slots = max(0, max_per_class - n_reg)
            if n_slots >= len(other_items):
                sampled_other = other_items
            else:
                idx = rng.choice(len(other_items), n_slots, replace=False)
                sampled_other = [other_items[i] for i in sorted(idx)]
            class_to_items[cls_id] = reg_items + sampled_other
        print(f'\n[Ограничение] max_per_class={max_per_class}, registration_page всегда все')

    print('\nФайлов из подпапок:')
    for cls_id in range(N_CLASSES):
        items = class_to_items.get(cls_id, [])
        folders = {}
        for img_path, cropped_path in items:
            sub = os.path.basename(os.path.dirname(cropped_path))
            folders[sub] = folders.get(sub, 0) + 1
        total = len(items)
        parts = [f'{k}: {v}' for k, v in sorted(folders.items())]
        parts_str = ', '.join(parts)
        print(f'  {CLASS_NAMES[cls_id]}: {total} total ({parts_str})')

    class_to_images = {cls_id: [p for p, _ in items] for cls_id, items in class_to_items.items()}

    if len(all_images) == 0:
        print('Error: No images found!')
        return

    # Разделяем каждый класс на train/val/test
    train_images, train_labels = [], []
    val_images, val_labels = [], []
    for cls_id in range(N_CLASSES):
        imgs = class_to_images.get(cls_id, [])
        if not imgs:
            continue
        tr, v, _ = split_dataset_by_folder(imgs, random_state=random_state)
        train_images.extend(tr)
        train_labels.extend([cls_id] * len(tr))
        val_images.extend(v)
        val_labels.extend([cls_id] * len(v))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = DocTypeDataset(train_images, train_labels, transform)
    val_dataset = DocTypeDataset(val_images, val_labels, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Модель
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    last_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    in_features = last_linear.in_features
    model.classifier[-1] = torch.nn.Linear(in_features, N_CLASSES)
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_wts = None
    epochs_without_improve = 0
    
    print(f'\nTraining parameters:')
    print(f'  Epochs: {epochs}')
    print(f'  Batch size: {batch_size}')
    print(f'  Learning rate: {lr}')
    print(f'  Device: {device}')
    print(f'  Train samples: {len(train_dataset)}')
    print(f'  Val samples: {len(val_dataset)}')
    print(f'  Early stopping patience: {patience}')
    
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

        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = model.state_dict().copy()
            os.makedirs(runs_dir, exist_ok=True)
            torch.save(best_model_wts, os.path.join(runs_dir, 'best_mobilenetv3_doctype.pth'))
            print(f'  -> Best model saved!')
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f'\nEarly stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).')
                break
    
    # Сохраняем последнюю модель
    torch.save(model.state_dict(), os.path.join(runs_dir, 'last_mobilenetv3_doctype.pth'))
    
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
    plt.savefig(os.path.join(runs_dir, 'train_curves.png'), dpi=150)
    plt.close()
    
    # Сохраняем имена классов для export и pipeline
    import json
    names_path = os.path.join(runs_dir, 'class_names.json')
    with open(names_path, 'w', encoding='utf-8') as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False, indent=2)
    print(f'Class names saved to: {names_path}')

    print(f'\nBest model saved to: {os.path.join(runs_dir, "best_mobilenetv3_doctype.pth")}')
    print(f'Training curves saved to: {os.path.join(runs_dir, "train_curves.png")}')

if __name__ == '__main__':
    main()




