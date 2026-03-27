from typing import Union
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

class TextFieldsDetector:
    """Detects text field regions in document images for Belarus passports.

    Uses ultralytics YOLO directly for better compatibility.
    """

    def __init__(self, model_format: str = 'PT', device='cpu', verbose: bool = False):
        """Initializes the text field detection model for Belarus."""
        self.model_name = 'TextFieldsDetectorBelarus'
        self.verbose = verbose
        # Путь к модели в папке models/TextFields/belarus/passport/PT/
        current_file = Path(__file__).resolve()
        # Поднимаемся от textfields_detector.py до models/TextFields/belarus/passport/PT/
        model_path = current_file.parent.parent.parent.parent.parent / 'models' / 'TextFields' / 'belarus' / 'passport' / 'PT' / 'best.pt'
        if not model_path.exists():
            raise Exception(f"Model not found: {model_path}")
        self.model = YOLO(str(model_path.resolve()))

    def load_img(self, img_path: Union[str, Path, np.ndarray]):
        """Loads image and converts it to RGB color mode"""
        if isinstance(img_path, Path):
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise Exception("Unsupported input type as img")
        return img

    @staticmethod
    def _parse_and_filter(results, labels):
        """Parse YOLO results and keep only the top-confidence bbox per class."""
        all_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            for box in boxes_data:
                x1, y1, x2, y2 = box[:4].astype(int)
                conf = float(box[4])
                cls = int(box[5])
                label = labels[cls] if cls < len(labels) else f'class_{cls}'
                all_boxes.append([x1, y1, x2, y2, conf, cls, label])
        best = {}
        for b in all_boxes:
            cls = b[5]
            if cls not in best or b[4] > best[cls][4]:
                best[cls] = b
        return list(best.values())

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Detects text fields, returns bounding boxes.

        Args:
            img: Input document image

        Returns:
            List of detected text field bounding boxes
        """
        img = self.load_img(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, conf=0.04, verbose=self.verbose)
        
        labels = ['authority', 'authority2', 'code_of_issuing', 'date_of_birth',
                  'date_of_expiry', 'date_of_issue', 'identification_no', 'names',
                  'nationality', 'passport_no', 'photo', 'place_of_birth',
                  'sex', 'signature', 'signature2', 'surname', 'type']
        
        bbox = self._parse_and_filter(results, labels)
        
        meta = {
            self.model_name: {
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
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, conf=0.04, verbose=self.verbose)
        
        labels = ['authority', 'authority2', 'code_of_issuing', 'date_of_birth',
                  'date_of_expiry', 'date_of_issue', 'identification_no', 'names',
                  'nationality', 'passport_no', 'photo', 'place_of_birth',
                  'sex', 'signature', 'signature2', 'surname', 'type']
        
        bbox = self._parse_and_filter(results, labels)
        img_patches = []
        for x1, y1, x2, y2, conf, cls, label in bbox:
            img_patches.append(img[y1:y2, x1:x2])
        
        meta = {
            self.model_name: {
                'bbox': bbox,
                'warped_img': img_patches,
            }
        }
        return meta

