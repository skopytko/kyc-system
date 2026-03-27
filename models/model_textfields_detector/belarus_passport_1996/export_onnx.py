"""
Экспорт TextFields Detector Belarus passport_1996 в ONNX.
"""

import os
import onnxruntime as ort
from ultralytics import YOLO


def main():
    model_path = os.path.join(os.path.dirname(__file__), 'runs', 'train', 'weights', 'best.pt')
    onnx_path = os.path.join(os.path.dirname(__file__), 'runs', 'train', 'weights', 'best.onnx')

    if not os.path.exists(model_path):
        print(f'Error: Model not found at {model_path}')
        print('Please train the model first: python train.py')
        return

    print(f'Loading YOLO model from: {model_path}')
    model = YOLO(model_path)

    print('Exporting to ONNX...')
    try:
        exported_path = model.export(
            format='onnx',
            imgsz=640,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False
        )
        if exported_path != onnx_path:
            import shutil
            os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
            shutil.move(exported_path, onnx_path)
        print(f'Model exported to: {onnx_path}')
    except Exception as e:
        print(f'Error during export: {e}')
        import traceback
        traceback.print_exc()
        return

    print('\nVerifying ONNX model...')
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_info = session.get_inputs()[0]
        print(f'Input: {input_info.name}, shape: {input_info.shape}')
        for out in session.get_outputs():
            print(f'Output: {out.name}, shape: {out.shape}')
        print('ONNX verification OK')
    except Exception as e:
        print(f'Verification warning: {e}')


if __name__ == '__main__':
    main()
