import os
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import numpy as np
import csv
import time

# Параметры
VAL_DIR = 'dataset/dataset_for_doc_type/main/val'
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PAGE_CLASSES = ['passport_centerfold', 'passport_pages']
NUM_CLASSES = len(PAGE_CLASSES)
MODEL_PATH = 'models/model_passport_page_type/runs/best_mobilenetv3_passport_page.pth'

# Собираем список файлов и меток
val_imgs = glob(os.path.join(VAL_DIR, '*.jpg'))
X_val, y_val = [], []
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

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Evaluating'):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds.cpu() == labels.cpu()).sum().item()
            total += imgs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = correct / total if total > 0 else 0
    return acc, np.array(all_preds), np.array(all_labels)

if __name__ == "__main__":
    val_ds = PassportPageDataset(X_val, y_val, transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = models.mobilenet_v3_small(pretrained=False)
    # Корректно ищем Linear слой
    last_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    if last_linear is None:
        print('Не найден Linear слой в classifier:', model.classifier)
        raise RuntimeError('Последний слой classifier не является torch.nn.Linear, проверьте структуру модели!')
    in_features = last_linear.in_features
    model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_files = []
    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc='Evaluating'):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(dim=1)
            all_probs.extend(max_probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    end_time = time.time()
    # Собираем список файлов в том же порядке, что и X_val
    all_files = [os.path.basename(f) for f in X_val]

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Accuracy: {acc:.4f}')

    # Скорость обработки
    if len(val_ds) > 0:
        avg_time = (end_time - start_time) / len(val_ds) * 1000  # ms/img
        print(f'Average speed: {avg_time:.2f} ms/img')

    # Сохраняем результаты в CSV
            with open('models/model_passport_page_type/results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file', 'label', 'pred', 'prob'])
        for fname, true, pred, prob in zip(all_files, all_labels, all_preds, all_probs):
            writer.writerow([fname, true, pred, float(prob)]) 