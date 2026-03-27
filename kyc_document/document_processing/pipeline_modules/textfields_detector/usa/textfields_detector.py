from typing import Union
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO


USA_STATES = [
    'alabama', 'alaska', 'arizona', 'arkansas', 'idaho',
    'iowa', 'vermont', 'virginia', 'washington', 'wisconsin', 'wyoming',
]

# Индексы как в docs_generator/usa/generator.py: IMAGE_FIELDS (0–2) + TEXT_FIELDS без последнего
# элемента dob_short (в YOLO 20 классов: photo, mini_photo, handwritten_signature, class…dob).
USA_FIELD_LABELS = [f'field_{i}' for i in range(20)]
# address в генераторе — поле address (class id 8 = field_8): несколько строк → несколько bbox.
USA_ADDRESS_CLASS_ID = 8
USA_ADDRESS_CONF_THRESHOLD = 0.5


class TextFieldsDetectorUSA:
    """Detects text field regions in USA driver's license images.

    Uses per-state ultralytics YOLO models. Each state has its own
    model trained on that state's DL layout.
    """

    def __init__(self, state: str, model_format: str = 'PT', device='cpu', verbose: bool = False):
        self.model_name = 'TextFieldsDetectorUSA'
        self.state = state.lower()
        self.verbose = verbose

        if self.state not in USA_STATES:
            raise ValueError(f"Unsupported USA state: {self.state}. Supported: {USA_STATES}")

        current_file = Path(__file__).resolve()
        model_path = (
            current_file.parent.parent.parent.parent
            / 'models' / 'TextFields' / 'usa' / self.state / 'PT' / 'best.pt'
        )
        if not model_path.exists():
            raise FileNotFoundError(f"USA TextFields model not found: {model_path}")
        self.model = YOLO(str(model_path))

    @staticmethod
    def load_img(img_path: Union[str, Path, np.ndarray]):
        if isinstance(img_path, Path):
            img = cv2.imread(img_path.as_posix())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, str):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_path, np.ndarray):
            img = img_path
        else:
            raise TypeError("Unsupported input type as img")
        return img

    @staticmethod
    def _parse_and_filter(results, labels):
        """Parse YOLO results: address (class 8) — все bbox с conf >= 0.7; остальные — один bbox на класс."""
        all_boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes_data = results[0].boxes.data.cpu().numpy()
            for box in boxes_data:
                x1, y1, x2, y2 = box[:4].astype(int)
                conf = float(box[4])
                cls = int(box[5])
                label = labels[cls] if cls < len(labels) else f'class_{cls}'
                all_boxes.append([x1, y1, x2, y2, conf, cls, label])

        address = [
            b for b in all_boxes
            if b[5] == USA_ADDRESS_CLASS_ID and b[4] >= USA_ADDRESS_CONF_THRESHOLD
        ]
        address.sort(key=lambda b: (b[1], b[0]))

        best_other = {}
        for b in all_boxes:
            if b[5] == USA_ADDRESS_CLASS_ID:
                continue
            cls = b[5]
            if cls not in best_other or b[4] > best_other[cls][4]:
                best_other[cls] = b

        return address + list(best_other.values())

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        img = self.load_img(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, conf=0.04, verbose=self.verbose)
        bbox = self._parse_and_filter(results, USA_FIELD_LABELS)
        return {self.model_name: {'bbox': bbox}}

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        img = self.load_img(img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        results = self.model(img_bgr, conf=0.04, verbose=self.verbose)
        bbox = self._parse_and_filter(results, USA_FIELD_LABELS)
        img_patches = []
        for x1, y1, x2, y2, conf, cls, label in bbox:
            img_patches.append(img[y1:y2, x1:x2])

        return {
            self.model_name: {
                'bbox': bbox,
                'warped_img': img_patches,
            }
        }
