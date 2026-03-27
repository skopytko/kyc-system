from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np

class DocType(BaseModule):
    """Detects document country (belarus/russia) from image using ONNX MobileNetV3."""

    def __init__(self, model_format: str = 'ONNX', device: str = 'cpu', verbose: bool = False):
        """Initializes document type detection model."""
        self.model_name = 'DocType'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Предсказывает страну документа (belarus/russia) и уверенность.

        Args:
            img: Входное изображение документа

        Returns:
            Dict с типом документа (belarus/russia) и уверенностью
        """
        img = self.load_img(img)
        tensor = self.model.preprocessing(img)
        tensor = tensor.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        logits = self.model.inference_model.predict(tensor)[0]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        doc_type, confidence = self.model.postprocessing(probs)
        
        meta = {
            self.model_name: {
                'doc_type': doc_type,
                'confidence': confidence
            }
        }
        return meta

