import torch
import numpy as np
from torchvision import models, transforms
from typing import Union
from pathlib import Path
from PIL import Image

from kyc_document.document_processing.processing.torch_device import torch_device

PAGE_CLASSES = ['passport_centerfold', 'passport_pages']
NUM_CLASSES = len(PAGE_CLASSES)

_COUNTRY_WEIGHTS = {
    'russia': 'kyc_document/document_processing/models/PassportPageType/russia/best.pt',
    'belarus': 'kyc_document/document_processing/models/PassportPageType/belarus/best.pt',
}

class PassportPageType:
    """Detects passport page type from image (centerfold/page) using PyTorch MobileNetV3.
    
    Supports per-country weights: pass ``country='russia'`` or ``country='belarus'``.
    """
    def __init__(self, model_format: str = 'PT', model_path: str = None,
                 device: str = 'cpu', verbose: bool = False,
                 country: str = 'russia'):
        self.model_name = 'PassportPageType'
        self.device = torch_device(device)
        self.verbose = verbose
        self.country = country.lower()
        if model_path is None:
            model_path = _COUNTRY_WEIGHTS.get(self.country)
            if model_path is None:
                raise ValueError(f"No PassportPageType weights for country={self.country}")
        self.model = models.mobilenet_v3_small(weights=None)
        last_linear = None
        for m in reversed(self.model.classifier):
            if isinstance(m, torch.nn.Linear):
                last_linear = m
                break
        in_features = last_linear.in_features
        self.model.classifier[-1] = torch.nn.Linear(in_features, NUM_CLASSES)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
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