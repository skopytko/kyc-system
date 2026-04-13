from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import os

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

        # Классификатор обучен только на известных классах и может быть сильно "переуверенным"
        # даже на нерелевантных изображениях. Добавляем простой rejection:
        # если max(prob) ниже порога или разница top1-top2 мала — считаем тип 'NONE'.
        min_conf = float(os.getenv("KYC_DOCTYPE_MIN_CONF", os.getenv("RDOCR_DOCTYPE_MIN_CONF", "0.0")))
        min_margin = float(os.getenv("KYC_DOCTYPE_MIN_MARGIN", os.getenv("RDOCR_DOCTYPE_MIN_MARGIN", "0.0")))
        try:
            p = probs.reshape(-1).astype(float)
            if p.size >= 2:
                top2 = np.sort(p)[-2:]
                margin = float(top2[1] - top2[0])
            else:
                margin = 1.0
            if float(confidence) < min_conf or margin < min_margin:
                doc_type = "NONE"
        except Exception:
            pass
        
        meta = {
            self.model_name: {
                'doc_type': doc_type,
                'confidence': confidence
            }
        }
        return meta

