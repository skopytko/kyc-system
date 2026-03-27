from ....base_module import BaseModule
from typing import List, Union
from pathlib import Path
import numpy as np

class TextFieldsDetector(BaseModule):
    """Detects text field regions in document images.

    Identifies areas like names, numbers, dates etc and
    returns bounding boxes and image patches.

    """
    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        """Initializes the text field detection model for Russia."""
        self.model_name = 'TextFieldsDetectorRussia'
        super().__init__(self.model_name, model_format=model_format, device=device, verbose=verbose)

    @staticmethod
    def _safe_crop(img: np.ndarray, box: list) -> tuple:
        """Вырезка по bbox с привязкой к границам кадра; пустые/инвертированные боксы отбрасываются."""
        h, w = img.shape[:2]
        try:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        except (TypeError, ValueError, IndexError):
            return None, False
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return None, False
        crop = img[y1:y2, x1:x2]
        if crop is None or crop.size == 0 or crop.shape[0] < 1 or crop.shape[1] < 1:
            return None, False
        return crop, True

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects text fields, returns bounding boxes.

        Args:
            img: Input document image

        Returns:
            List of detected text field bounding boxes
        """
        self.load_img(img)
        bbox = self.model.predict(img)
        meta = {
            self.model_name:
                {
                    'bbox': bbox,

                }
        }
        return meta

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects fields and extracts image patches.

        Args:
            img: Input document image

        Returns:
            Bounding boxes, List of extracted image patches
        """
        img = self.load_img(img)
        bbox = self.model.predict(img)
        img_patches: List[np.ndarray] = []
        valid_bbox: list = []
        for box in bbox:
            crop, ok = self._safe_crop(img, box)
            if not ok:
                continue
            valid_bbox.append(box)
            img_patches.append(crop)
        meta = {
            self.model_name:
                {
                    'bbox': valid_bbox,
                    'warped_img': img_patches,
                }
        }
        return meta
