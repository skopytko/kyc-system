from kyc_document.document_processing.pipeline_modules.passport_seal_detector.seal_processing import process_seals, select_last_seal
from typing import Union, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

class PassportSealDetector:
    """Detects passport seals and selects the most recent one based on passport page layout."""
    
    def __init__(self, model_format: str = 'PT', device='cpu', verbose: bool = False):
        self.model_name = 'PassportSealDetector'
        self.model_format = model_format
        self.device = device
        self.verbose = verbose
        
        if model_format == 'PT':
            from ultralytics import YOLO
            model_path = 'kyc_document/document_processing/models/PassportSeal/best.pt'
            self.model = YOLO(model_path)
            self.pt_mode = True
        else:
            # Для других форматов можно добавить поддержку позже
            raise NotImplementedError(f"Model format {model_format} not supported yet")
    
    def predict(self, img: Union[str, Path, np.ndarray]) -> dict:
        """Обнаруживает печати на изображении паспорта.
        
        Args:
            img: Входное изображение
            
        Returns:
            Словарь с результатами обнаружения
        """
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
        results = self.model(img_bgr, verbose=False)[0]
        
        seals = []
        if hasattr(results, 'masks') and results.masks is not None:
            confs = results.boxes.conf.cpu().numpy()
            bboxes = results.boxes.data.cpu().numpy()
            masks = results.masks.data.cpu().numpy()
            
            # Обрабатываем каждую обнаруженную печать
            for i, (conf, bbox, mask) in enumerate(zip(confs, bboxes, masks)):
                # Масштабируем маску до размера изображения
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
                mask_bin = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Аппроксимируем до 4-точечного контура
                quad = self._approximate_seal_to_quad(mask_bin)
                
                if quad is not None:
                    # Вычисляем центр печати
                    center = quad.mean(axis=0)
                    
                    seals.append({
                        'id': i,
                        'confidence': float(conf),
                        'bbox': bbox.tolist(),
                        'mask': mask_bin,
                        'quad': quad.tolist(),
                        'center': center.tolist(),
                        'side': self._determine_page_side(center, img.shape)
                    })
        
        # Обрабатываем печати и выбираем последнюю
        processed_seals = process_seals(seals, img.shape)
        last_seal = select_last_seal(processed_seals)
        
        # Возвращаем результаты с контурами печатей
        meta = {
            self.model_name: {
                'total_seals': len(processed_seals),
                'has_seals': len(processed_seals) > 0,
                'seals': processed_seals,   # список печатей с quad/center/side/confidence
                'last_seal': last_seal
            }
        }
        
        return meta
    
    def _fix_seal_perspective(self, img: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Исправляет перспективу печати и возвращает обрезанное изображение.
        
        Args:
            img: Исходное изображение
            quad: 4 точки контура печати
            
        Returns:
            Обрезанное и выровненное изображение печати
        """
        if quad is None or len(quad) != 4:
            return img
        
        # Упорядочиваем точки по часовой стрелке
        quad = self._order_seal_points(quad)
        
        # Вычисляем размеры выходного изображения
        widthA = np.linalg.norm(quad[1] - quad[0])  # верхняя сторона
        widthB = np.linalg.norm(quad[2] - quad[3])  # нижняя сторона
        maxWidth = int(max(widthA, widthB))
        
        heightA = np.linalg.norm(quad[3] - quad[0])  # левая сторона
        heightB = np.linalg.norm(quad[2] - quad[1])  # правая сторона
        maxHeight = int(max(heightA, heightB))
        
        # Создаем матрицу преобразования
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")
        
        # Применяем перспективное преобразование
        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        
        return warped
    
    def _order_seal_points(self, pts: np.ndarray) -> np.ndarray:
        """Упорядочивает точки контура печати по часовой стрелке.
        
        Args:
            pts: 4 точки контура
            
        Returns:
            Упорядоченные точки
        """
        if pts is None or len(pts) != 4:
            return pts
        
        # Находим центр
        center = pts.mean(axis=0)
        
        # Вычисляем углы от центра
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        
        # Сортируем по углам
        indices = np.argsort(angles)
        
        return pts[indices]
    
    def predict_transform(self, img: Union[str, Path, np.ndarray], include_annotated_img: bool = False) -> dict:
        """Обнаруживает печати и возвращает изображения с контурами.
        
        Args:
            img: Входное изображение
            include_annotated_img: Включать ли изображение с аннотациями в результат
            
        Returns:
            Словарь с результатами и изображениями
        """
        meta = self.predict(img)
        
        # Создаем изображение с контурами печатей
        if isinstance(img, np.ndarray):
            img_copy = img.copy()
        else:
            img_copy = cv2.imread(str(img))
            if img_copy is None:
                img_copy = img
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        
        # Получаем информацию о печатях
        seals_info = meta[self.model_name]
        
        # Создаем изображение с контурами печатей
        seals_img = img_copy.copy()
        fixed_seal_img = None
        
        # Рисуем квадраты печатей, если они есть
        if seals_info.get('has_seals', False):
            for i, seal in enumerate(seals_info.get('seals', [])):
                quad = seal.get('quad')
                conf = seal.get('confidence', 0.0)
                if quad is None:
                    continue
                pts = np.array(quad, dtype=np.int32).reshape(-1, 1, 2)

                # Полупрозрачная заливка полигона для лучшей видимости
                overlay = seals_img.copy()
                cv2.fillPoly(overlay, [pts], color=(0, 0, 255))  # красный
                alpha = 0.18
                seals_img = cv2.addWeighted(overlay, alpha, seals_img, 1 - alpha, 0)

                # Рамка печати поверх заливки
                cv2.polylines(seals_img, [pts], True, (0, 0, 255), 3)

                # Подпись в левом верхнем углу рамки на черном фоне
                left_top = pts[0][0]
                label = f"Seal {i+1} ({conf:.2f})"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                pad = 4
                x1 = max(left_top[0], 0)
                y1 = max(left_top[1] - th - 2 * pad, 0)
                x2 = min(x1 + tw + 2 * pad, seals_img.shape[1] - 1)
                y2 = min(left_top[1], seals_img.shape[0] - 1)
                cv2.rectangle(seals_img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                cv2.putText(seals_img, label, (x1 + pad, y2 - pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Создаем обрезанное изображение основной печати
            last_seal = seals_info.get('last_seal')
            if last_seal and last_seal.get('quad') is not None:
                quad = np.array(last_seal['quad'])
                fixed_seal_img = self._fix_seal_perspective(img_copy, quad)
        
        # Добавляем изображения в результат
        meta[self.model_name]['seals_img'] = seals_img  # Изображение с контурами печатей
        meta[self.model_name]['warped_img'] = img_copy  # Исходное изображение
        if fixed_seal_img is not None:
            meta[self.model_name]['fixed_seal_img'] = fixed_seal_img  # Обрезанная печать
        
        return meta
    
    def _determine_page_side(self, seal_center: np.ndarray, img_shape: Tuple[int, int]) -> str:
        """Определяет, на какой стороне страницы находится печать.
        
        Args:
            seal_center: Центр печати (x, y)
            img_shape: Размеры изображения (height, width)
            
        Returns:
            'left' или 'right' в зависимости от расположения
        """
        h, w = img_shape[:2]
        center_x = seal_center[0]
        
        # Если центр печати находится в левой половине изображения
        if center_x < w / 2:
            return 'left'
        else:
            return 'right'
    
    def _approximate_seal_to_quad(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Аппроксимирует маску печати до 4-точечного контура.
        
        Args:
            mask: Бинарная маска печати
            
        Returns:
            4 точки контура или None если не удалось аппроксимировать
        """
        # Находим контуры в маске
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Берем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Аппроксимируем до многоугольника
        peri = cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
        
        # Если получилось 4 точки, возвращаем их
        if len(approx) == 4:
            return approx.reshape(-1, 2)
        
        # Если точек больше или меньше, пытаемся получить 4 точки
        if len(approx) > 4:
            # Берем 4 точки с максимальным расстоянием между ними
            points = approx.reshape(-1, 2)
            # Простая эвристика: берем угловые точки
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            return box.astype(np.int32)
        else:
            # Если точек меньше 4, расширяем контур
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            return box.astype(np.int32)
