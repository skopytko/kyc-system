import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import MobileNet_V3_Small_Weights

# Параметры
DATA_DIR = 'dataset/dataset_for_doc_type/main/train'
VAL_DIR = 'dataset/dataset_for_doc_type/main/val'
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAGE_CLASSES = ['passport_centerfold', 'passport_pages']  # Классы страниц паспорта
NUM_CLASSES = len(PAGE_CLASSES)

# Собираем список файлов и меток для train
all_imgs = glob(os.path.join(DATA_DIR, '*.jpg'))
# Добавляем изображения из passport/
passport_dir = 'dataset/dataset_for_doc_type/passport'
all_imgs += glob(os.path.join(passport_dir, '*.jpg'))
all_imgs += glob(os.path.join(passport_dir, '*.png'))
X = []
y = []
for img_path in all_imgs:
    basename = os.path.basename(img_path).lower()
    # Если имя только число (например, 91.jpg), это passport_centerfold
    if basename.split('.')[0].isdigit():
        X.append(img_path)
        y.append(0)
        continue
    if 'passport_centerfold' in basename:
        X.append(img_path)
        y.append(0)
        continue
    if 'passport_pages' in basename or 'page_5-12' in basename:
        X.append(img_path)
        y.append(1)
        continue
# Собираем список файлов и меток для val
val_imgs = glob(os.path.join(VAL_DIR, '*.jpg'))
X_val = []
y_val = []
for img_path in val_imgs:
    basename = os.path.basename(img_path).lower()
    if basename.split('.')[0].isdigit():
        X_val.append(img_path)
        y_val.append(0)
        continue
    for idx, page in enumerate(PAGE_CLASSES):
        if page in basename:
            X_val.append(img_path)
            y_val.append(idx)
            break

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class PassportPageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    def __len__(self) -> int:
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

if __name__ == "__main__":
    train_ds = PassportPageDataset(X, y, transform)
    val_ds = PassportPageDataset(X_val, y_val, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Модель
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    # Корректно ищем Linear слой
    last_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    in_features = last_linear.in_features
    model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)  # многоклассовая классификация
    model = model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_model_wts = None

    # Обучение
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - train'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
            for imgs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} - val'):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
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
            best_model_wts = model.state_dict()
            torch.save(model.state_dict(), 'models/model_passport_page_type/runs/best_mobilenetv3_passport_page.pth')

    # Сохраняем финальную модель
            torch.save(model.state_dict(), 'models/model_passport_page_type/runs/last_mobilenetv3_passport_page.pth')
            print('Финальная модель сохранена в models/model_passport_page_type/runs/last_mobilenetv3_passport_page.pth')
        print('Лучшая модель сохранена в models/model_passport_page_type/runs/best_mobilenetv3_passport_page.pth')

    # Сохраняем историю
    import pickle
            with open('models/model_passport_page_type/runs/train_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Графики
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train acc')
    plt.plot(history['val_acc'], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.tight_layout()
            plt.savefig('models/model_passport_page_type/runs/train_curves.png')
