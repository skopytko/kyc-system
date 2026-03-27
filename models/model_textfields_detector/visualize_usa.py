"""
Визуализация предсказаний TextFields Detector для прав штатов USA.

Использует обученную модель из runs/usa_<state>*/weights/best.pt и рисует bbox
на произвольном изображении.

Пример запуска:
    python visualize_usa.py idaho /path/to/image.png
"""

import os
import sys
import yaml
import cv2
from ultralytics import YOLO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def find_usa_model(state):
    """Ищет best.pt для штата: runs/usa_<state>*/weights/best.pt."""
    prefix = f"usa_{state}"
    if not os.path.exists(RUNS_DIR):
        return None
    for name in sorted(os.listdir(RUNS_DIR)):
        if name == prefix or (name.startswith(prefix) and len(name) > len(prefix)):
            cand = os.path.join(RUNS_DIR, name, "weights", "best.pt")
            if os.path.exists(cand):
                return cand
    return None


def load_class_names(state: str) -> list[str]:
    """Берёт имена классов из data.yaml для соответствующего штата."""
    data_yaml = os.path.join(
        RUNS_DIR, "dataset_usa", state, "data.yaml"
    )
    if os.path.exists(data_yaml):
        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        names = data.get("names")
        if isinstance(names, list) and names:
            return [str(n) for n in names]
    #Fallback: 30 абстрактных полей
    return [f"field_{i}" for i in range(30)]


def draw_boxes(img, results, class_names, conf_threshold: float = 0.25):
    """Рисует bbox на изображении и возвращает результат."""
    result = img.copy()
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return result

    boxes_data = results[0].boxes.data.cpu().numpy()
    colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
        (200, 150, 100),
        (100, 200, 150),
        (150, 100, 200),
        (200, 100, 150),
        (100, 150, 200),
        (200, 200, 100),
        (100, 200, 200),
        (200, 100, 200),
        (150, 200, 100),
        (100, 150, 200),
        (180, 120, 80),
        (80, 180, 120),
        (120, 80, 180),
        (180, 80, 120),
        (80, 120, 180),
    ]

    for box in boxes_data:
        x1, y1, x2, y2 = box[:4].astype(int)
        conf = float(box[4]) if len(box) > 4 else 1.0
        cls = int(box[5]) if len(box) > 5 else 0

        if conf < conf_threshold:
            continue

        color = colors[cls % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        name = class_names[cls] if cls < len(class_names) else f"cls_{cls}"
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            result,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python visualize_usa.py <state> <image_path>")
        sys.exit(1)

    state = sys.argv[1].strip().lower()
    image_path = sys.argv[2]
    image_path = (
        image_path
        if os.path.isabs(image_path)
        else os.path.join(REPO_ROOT, image_path)
    )

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model_path = find_usa_model(state)
    if not model_path:
        print(f"Model not found for state {state} in {RUNS_DIR}")
        sys.exit(1)

    class_names = load_class_names(state)
    conf_threshold = float(os.environ.get("TEXTFIELDS_CONF", 0.25))

    print(f"State: {state}")
    print(f"Model: {model_path}")
    print(f"Image: {image_path}")
    print(f"Conf threshold: {conf_threshold}")

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image with OpenCV.")
        sys.exit(1)

    model = YOLO(model_path)
    results = model(img, conf=conf_threshold, verbose=False)
    result_img = draw_boxes(img, results, class_names, conf_threshold)

    base, ext = os.path.splitext(image_path)
    out_path = f"{base}_pred_{state}{ext}"
    cv2.imwrite(out_path, result_img)
    print(f"Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()

