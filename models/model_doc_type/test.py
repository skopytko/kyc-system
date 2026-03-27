import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
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
        return img, label, img_path

def find_cropped_folders(dataset_root):
    """Находит все папки cropped в dataset."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
    dataset_root = os.path.join(os.path.dirname(__file__), '../../dataset')
    model_path = os.path.join(os.path.dirname(__file__), 'runs', 'best_mobilenetv3_doctype.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random_state = int(os.environ.get('DOCTYPE_RANDOM_STATE', 42))
    
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from classes_config import cropped_path_to_class, CLASS_NAMES, N_CLASSES

    cropped_folders = find_cropped_folders(dataset_root)
    all_images = []
    for cropped_path in cropped_folders:
        cls_id = cropped_path_to_class(cropped_path)
        if cls_id is None:
            continue
        images = collect_images_from_cropped(cropped_path)
        for img_path in images:
            all_images.append((img_path, cls_id))

    class_to_images = {i: [] for i in range(N_CLASSES)}
    for img_path, cls_id in all_images:
        class_to_images[cls_id].append(img_path)

    test_images, test_labels = [], []
    for cls_id in range(N_CLASSES):
        imgs = class_to_images.get(cls_id, [])
        if not imgs:
            continue
        _, _, te = split_dataset_by_folder(imgs, random_state=random_state)
        test_images.extend(te)
        test_labels.extend([cls_id] * len(te))

    print(f'Found {len(test_images)} test images across {N_CLASSES} classes')
    
    test_dataset = DocTypeDataset(test_images, test_labels, transform)
    batch_size = int(os.environ.get('DOCTYPE_BATCH', 8))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Загружаем модель
    if not os.path.exists(model_path):
        print(f'Model not found: {model_path}')
        print('Please train the model first using train.py')
        return
    
    model = models.mobilenet_v3_small(weights=None)
    # Заменяем классификатор на 2 класса
    last_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    in_features = last_linear.in_features
    model.classifier[-1] = torch.nn.Linear(in_features, N_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels, all_probs, all_paths = [], [], [], []
    
    print(f'Testing on {len(test_dataset)} samples...')
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            imgs, labels, paths = batch
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(list(paths))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    cm = confusion_matrix(all_labels, all_preds)
    class_names = CLASS_NAMES
    
    results = []
    results.append('=' * 60)
    results.append('MODEL EVALUATION RESULTS')
    results.append('=' * 60)
    results.append(f'\nOverall Metrics:')
    results.append(f'  Accuracy: {acc:.4f}')
    results.append(f'  Precision (macro): {precision_macro:.4f}')
    results.append(f'  Recall (macro): {recall_macro:.4f}')
    results.append(f'  F1-score (macro): {f1_macro:.4f}')
    
    results.append(f'\nPer-Class Metrics:')
    for i, name in enumerate(class_names):
        results.append(f'  {name}:')
        results.append(f'    Precision: {precision[i]:.4f}')
        results.append(f'    Recall: {recall[i]:.4f}')
        results.append(f'    F1-score: {f1[i]:.4f}')
        results.append(f'    Support: {support[i]}')
    
    results.append(f'\nConfusion Matrix:')
    results.append(str(cm))
    
    results.append(f'\nClassification Report:')
    results.append(classification_report(all_labels, all_preds, target_names=class_names))
    
    results.append(f'\nConfusion Matrix (Normalized):')
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    results.append(str(cm_norm))
    
    output = '\n'.join(results)
    print(output)
    
    output_file = os.path.join(os.path.dirname(__file__), 'runs', 'test_results.txt')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(output)
    print(f'\nResults saved to: {output_file}')
    
    # Визуализация confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    short_names = [n.replace('belarus_', 'b_').replace('passport_', 'p').replace('usa_', '') for n in class_names]
    im1 = axes[0].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(short_names, rotation=45, ha='right')
    axes[0].set_yticklabels(short_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color='black' if cm[i, j] < cm.max()/2 else 'white')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(cm_norm, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(short_names, rotation=45, ha='right')
    axes[1].set_yticklabels(short_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center', color='black' if cm_norm[i, j] < 0.5 else 'white')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'runs', 'confusion_matrix.png'), dpi=150)
    plt.close()
    print('Confusion matrix saved to: runs/confusion_matrix.png')
    
    # Находим неправильно предсказанные файлы
    wrong_predictions = []
    
    for i, (true_label, pred_label, path, prob) in enumerate(zip(all_labels, all_preds, all_paths, all_probs)):
        if true_label != pred_label:
            wrong_predictions.append({
                'path': path,
                'true_class': class_names[true_label],
                'pred_class': class_names[pred_label],
                'confidence': prob[pred_label],
                'true_confidence': prob[true_label]
            })
    
    if wrong_predictions:
        results.append(f'\n\nIncorrect Predictions ({len(wrong_predictions)}/{len(all_labels)}):')
        results.append('=' * 60)
        for i, item in enumerate(wrong_predictions, 1):
            results.append(f'\n{i}. {os.path.basename(item["path"])}')
            results.append(f'   True: {item["true_class"]} | Predicted: {item["pred_class"]} | Confidence: {item["confidence"]:.4f} | True conf: {item["true_confidence"]:.4f}')
            results.append(f'   Path: {item["path"]}')
        
        output = '\n'.join(results)
        print('\n' + '=' * 60)
        print(f'INCORRECT PREDICTIONS ({len(wrong_predictions)}/{len(all_labels)}):')
        print('=' * 60)
        for i, item in enumerate(wrong_predictions[:20], 1):  # Показываем первые 20
            print(f'{i}. {os.path.basename(item["path"])}')
            print(f'   True: {item["true_class"]} | Pred: {item["pred_class"]} | Conf: {item["confidence"]:.4f}')
        if len(wrong_predictions) > 20:
            print(f'\n... and {len(wrong_predictions) - 20} more (see test_results.txt for full list)')
        
        # Сохраняем обновленные результаты
        with open(output_file, 'w') as f:
            f.write(output)
        print(f'\nFull list of incorrect predictions saved to: {output_file}')
    else:
        print('\nAll predictions are correct!')

if __name__ == '__main__':
    main()

