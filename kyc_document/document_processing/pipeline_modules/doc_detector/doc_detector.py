from .image_transformation import fix_perspective
from typing import Union
from pathlib import Path
import numpy as np
import cv2

class DocDetector:
    """Detects document and fixes perspective issues. Использует PT (YOLOv8) только для Borders, иначе ONNX/BaseModule."""
    # Минимальная доля площади кадра, чтобы считать детекцию документа валидной.
    # Если маска меньше — скорее всего это артефакт/ложная детекция.
    MIN_MASK_AREA_FRAC = 0.08

    def __init__(self, model_format: str = 'ONNX', device='cpu', verbose: bool = False):
        self.model_name = 'DocDetector'
        self.model_format = model_format
        self.device = device
        self.verbose = verbose
        if model_format in ['PT', 'BORDERS_PT']:
            from ultralytics import YOLO
            model_path = 'kyc_document/document_processing/models/Borders/best.pt'
            self.model = YOLO(model_path)
            self.pt_mode = True
        else:
            from ..base_module import BaseModule
            super().__init__()
            self.base = BaseModule(self.model_name, model_format=model_format, device=device, verbose=verbose)
            self.pt_mode = False

    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        if self.pt_mode:
            is_bgr = False
            if isinstance(img, (str, Path)):
                img = cv2.imread(str(img))
                if img is None:
                    raise Exception(f"Not an image {img}")
                is_bgr = True
            elif isinstance(img, np.ndarray):
                pass
            else:
                raise Exception("Unsupported Image Type")
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                is_bgr = True
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            img_bgr = img if is_bgr else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Ultralytics по умолчанию использует conf≈0.25; для рамок документа это бывает излишне строго.
            # Мы компенсируем возможный шум фильтрацией по площади маски.
            results = self.model(img_bgr, verbose=self.verbose, conf=0.1)[0]
            if hasattr(results, 'masks') and results.masks is not None:
                confs = results.boxes.conf.cpu().numpy()
                boxes_all = results.boxes.data.cpu().numpy()
                masks_all = results.masks.data.cpu().numpy()

                h, w = img.shape[:2]
                img_area = float(h * w) if h and w else 1.0

                candidates = []
                # Считаем площадь по бинаризованной маске в исходном размере.
                for i in range(len(confs)):
                    mask_resized = cv2.resize(
                        masks_all[i],
                        (w, h),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    mask_bin = (mask_resized > 0.5).astype(np.uint8)
                    area_frac = float(mask_bin.sum()) / img_area
                    # Скоринг: приоритет площади (чтобы не выбирать «крошечные» детекции),
                    # а confidence — как tie-breaker.
                    score = (area_frac, float(confs[i]))
                    candidates.append((score, i, mask_resized, mask_bin))

                # Сортируем по площади, затем по уверенности.
                candidates.sort(key=lambda x: x[0], reverse=True)

                chosen = []
                for score, i, mask_resized, mask_bin in candidates:
                    area_frac = score[0]
                    if area_frac < self.MIN_MASK_AREA_FRAC:
                        continue
                    chosen.append((i, mask_resized, mask_bin))
                    if len(chosen) >= 2:
                        break

                # Если все маски маленькие — считаем, что документа нет (лучше без perspective-fix,
                # чем искажать картинку по артефакту).
                if not chosen:
                    bboxes, masks, segm = [], [], []
                else:
                    idxs = np.array([i for i, _, _ in chosen], dtype=int)
                    bboxes = boxes_all[idxs]
                    masks = masks_all[idxs]
                    segm = []
                    for _, mask_resized, mask_bin in chosen:
                        mask_u8 = (mask_bin * 255).astype(np.uint8)
                        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Берём самый большой контур, а не первый попавшийся.
                            cnt = max(contours, key=cv2.contourArea)
                            segm.append(cnt.reshape(-1, 2))
                        else:
                            segm.append(np.zeros((0, 2)))
            else:
                bboxes, masks, segm = [], [], []
            meta = {
                self.model_name: {
                    'bbox': bboxes,
                    'mask': masks,
                    'segm': segm,
                }
            }
            return meta
        else:
            # ONNX/OV/TF etc через BaseModule
            return self.base.predict(img)

    def predict_transform(self, img: Union[str, Path, np.ndarray]) -> dict:
        if isinstance(img, (str, Path)):
            img = cv2.imread(str(img))
            if img is None:
                raise Exception(f"Not an image {img}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        meta = self.predict(img)
        doc = meta[self.model_name]
        bbox, mask, segm = doc['bbox'], doc['mask'], doc['segm']
        if len(segm) > 0:
            try:
                result_img, borders_img = fix_perspective(img=img, segments=segm)
            except Exception as e:
                print('[!] Failed to fix perspective')
                result_img = borders_img = img
        else:
            result_img = borders_img = img
        meta[self.model_name]['border_img'] = borders_img
        meta[self.model_name]['warped_img'] = result_img
        return meta
