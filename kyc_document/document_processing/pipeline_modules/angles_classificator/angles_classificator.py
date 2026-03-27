from ..base_module import BaseModule
from typing import Union
from pathlib import Path
import numpy as np
import cv2

class Angle90(BaseModule):
    """Detects document rotation angle and corrects it.

    Detects if a document image needs to be rotated 0, 90, 180
    or 270 degrees. Can return just the rotation angle prediction
    or rotate the image in-place.
    """

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the angle detection model."""
        self.model_name = 'Angle90'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    def _get_angle(self, img):
        """Helper method to get angle and confidence."""
        tensor = self.model.preprocessing(img)
        tensor = tensor.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
        logits = self.model.inference_model.predict(tensor)[0]
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        return self.model_info['Labels'][int(probs.argmax())], float(probs.max())

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts rotation angle and confidence."""
        angle, conf = self._get_angle(self.load_img(img))
        return {self.model_name: {'angle': angle, 'confidence': conf}}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Predicts and applies document rotation."""
        img = self.load_img(img)
        angle, conf = self._get_angle(img)
        for _ in range(angle // 90):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return {self.model_name: {'angle': angle, 'confidence': conf, 'warped_img': img}}
