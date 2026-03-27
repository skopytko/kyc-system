import torch
import numpy as np
from torchvision import models, transforms
from typing import Union
from pathlib import Path
from PIL import Image

PAGE_CLASSES = ['passport_centerfold', 'passport_pages']
NUM_CLASSES = len(PAGE_CLASSES)

class PassportPageType:
    """Detects passport page type from image (centerfold/page) using PyTorch MobileNetV3."""
    def __init__(self, model_format: str = 'PT', model_path: str = None, device: str = 'cpu', verbose: bool = False):
        self.model_name = 'PassportPageType'
        self.device = torch.device(device)
        self.verbose = verbose
        # model_format игнорируется, только для совместимости
        if model_path is None:
            model_path = 'kyc_document/document_processing/models/PassportPageType/best.pt'
        self.model = models.mobilenet_v3_small(weights=None)
        # Replace classifier for 2 classes
        last_linear = None
        for m in reversed(self.model.classifier):
            if isinstance(m, torch.nn.Linear):
                last_linear = m
                break
        in_features = last_linear.in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def preprocess(self, img: Union[str, Path, np.ndarray, Image.Image]):
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pass
        else:
            raise Exception('Unsupported image type for PassportPageType')
        return self.transform(img)

    def predict(self, img: Union[str, Path, np.ndarray, Image.Image]) -> str:
        x = self.preprocess(img)
        x = x.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            top1 = int(np.argmax(probs))
            page_type = PAGE_CLASSES[top1]
        return page_type 