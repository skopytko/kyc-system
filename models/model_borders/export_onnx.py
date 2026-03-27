import os
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np

def main():
    model_path = Path(__file__).parent / 'runs' / 'weights' / 'best.pt'
    onnx_path = Path(__file__).parent / 'runs' / 'yolo11m_seg_borders.onnx'
    
    print(f'Загрузка модели: {model_path}')
    model = YOLO(str(model_path))
    
    print('Экспорт в ONNX...')
    model.export(
        format='onnx',
        imgsz=640,
        simplify=True,
        opset=11
    )
    
    # YOLO сохраняет в ту же директорию, что и модель
    exported_path = model_path.parent / f"{model_path.stem}.onnx"
    if exported_path.exists():
        # Перемещаем в нужное место
        exported_path.rename(onnx_path)
        print(f'Модель экспортирована в: {onnx_path}')
    else:
        print(f'Ошибка: файл не найден в {exported_path}')
        return
    
    # Проверяем что ONNX модель работает
    print('Проверка ONNX модели...')
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # Получаем информацию о входе и выходе
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f'Вход: {input_name}, форма: {input_shape}')
    
    # Тестируем на dummy input
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outputs = session.run(None, {input_name: dummy_input})
    
    print(f'Количество выходов: {len(outputs)}')
    for i, output in enumerate(outputs):
        print(f'Выход {i}: форма {output.shape}')
    
    print('✓ ONNX модель успешно проверена!')

if __name__ == '__main__':
    main()





