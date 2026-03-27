import os
import sys
import torch
import numpy as np
import onnxruntime as ort
from torchvision import models

def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from classes_config import N_CLASSES, CLASS_NAMES

    model_path = os.path.join(os.path.dirname(__file__), 'runs', 'best_mobilenetv3_doctype.pth')
    onnx_path = os.path.join(os.path.dirname(__file__), 'runs', 'mobilenetv3_doctype.onnx')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Loading model from: {model_path}')
    model = models.mobilenet_v3_small(weights=None)
    last_linear = None
    for m in reversed(model.classifier):
        if isinstance(m, torch.nn.Linear):
            last_linear = m
            break
    in_features = last_linear.in_features
    model.classifier[-1] = torch.nn.Linear(in_features, N_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Создаем dummy input для экспорта
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print('Exporting to ONNX...')
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Model exported to: {onnx_path}')
    
    # Проверяем что ONNX модель работает
    print('Verifying ONNX model...')
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    # Получаем информацию о входе и выходе
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    print(f'Input name: {input_name}, shape: {input_shape}')
    print(f'Output name: {output_name}')
    
    # Тестируем на dummy input
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run([output_name], {input_name: dummy_np})
    
    print(f'Output shape: {outputs[0].shape}')
    print(f'Output sample: {outputs[0][0]}')
    print('ONNX model verification successful!')
    
    # Сравниваем с PyTorch моделью
    print('\nComparing PyTorch and ONNX outputs...')
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(dummy_np).to(device)).cpu().numpy()
    
    diff = np.abs(pytorch_output - outputs[0]).max()
    print(f'Max difference: {diff:.6f}')
    if diff < 1e-5:
        print('✓ Outputs match!')
    else:
        print(f'⚠ Warning: outputs differ by {diff:.6f}')

if __name__ == '__main__':
    main()

