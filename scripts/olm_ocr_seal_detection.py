#!/usr/bin/env python3
"""
Скрипт для распознавания текста с изображения fixed_seal.jpg через модель OLM OCR (Qwen2VL-OCR-2B)
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Добавляем путь к модулям проекта
sys.path.append('../russian_docs_ocr')

# Проверка доступности GPU
print(f"CUDA доступен: {torch.cuda.is_available()}")
print(f"Устройство модели: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

# Загрузка модели с явным указанием устройства
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Загружает модель OLM OCR"""
    try:
        print("🔄 Загрузка модели Qwen2VL-OCR-2B...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            torch_dtype=torch.float32
        )
        print("✓ Модель успешно загружена!")
        return model
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None

def load_processor():
    """Загружает процессор"""
    try:
        print("🔄 Загрузка процессора...")
        processor = AutoProcessor.from_pretrained(
            "prithivMLmods/Qwen2-VL-OCR-2B-Instruct",
            use_fast=True
        )
        print("✓ Процессор успешно загружен!")
        return processor
    except Exception as e:
        print(f"❌ Ошибка загрузки процессора: {e}")
        return None

def get_ocr_text(image_path, model, processor, lang="Russian"):
    """Функция для получения текста из изображения"""
    if not Path(image_path).exists():
        print(f"❌ Файл {image_path} не найден!")
        return ""
    
    try:
        prompt = f"Give me text from image, written in {lang} language, nothing else."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Подготовка входных данных
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Загружаем изображение
        image = Image.open(image_path)
        
        inputs = processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Генерация текста
        print("🔄 Выполняется распознавание текста...")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]  # Обрезаем промпт
        result = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        result = result.replace("<|im_end|>", "").strip()
        print("✓ OCR завершен успешно")
        return result
    
    except Exception as e:
        print(f"❌ Ошибка обработки {image_path}: {e}")
        return ""

def process_seal_image(image_path, output_file=None, language="Russian"):
    """
    Основная функция для обработки изображения fixed_seal.jpg
    """
    print(f"🔍 Обработка изображения: {image_path}")
    
    # Загружаем модель и процессор
    model = load_model()
    if model is None:
        return
    
    processor = load_processor()
    if processor is None:
        return
    
    # Выполняем OCR
    ocr_result = get_ocr_text(image_path, model, processor, language)
    if not ocr_result:
        print("❌ Не удалось получить результат OCR")
        return
    
    # Выводим результаты
    print("\n" + "="*50)
    print("📝 РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ ТЕКСТА:")
    print("="*50)
    print(f"📄 Распознанный текст:\n{ocr_result}")
    
    # Сохраняем результат в файл если указан
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(ocr_result)
            print(f"\n💾 Результат сохранен в файл: {output_file}")
        except Exception as e:
            print(f"❌ Ошибка при сохранении в файл: {e}")
    
    print("\n" + "="*50)

def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(
        description='Распознавание текста с изображения fixed_seal.jpg через OLM OCR (Qwen2VL-OCR-2B)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python olm_ocr_seal_detection.py
  python olm_ocr_seal_detection.py -i path/to/image.jpg
  python olm_ocr_seal_detection.py -l "Russian" -o result.txt
        """
    )
    
    parser.add_argument(
        '-i', '--image',
        type=str,
        default='../intermediate_results/fixed_seal.jpg',
        help='Путь к изображению для OCR (по умолчанию: ../intermediate_results/fixed_seal.jpg)'
    )
    
    parser.add_argument(
        '-l', '--language',
        type=str,
        default='Russian',
        help='Язык для распознавания (по умолчанию: Russian)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Файл для сохранения результата OCR'
    )
    
    args = parser.parse_args()
    
    # Проверяем существование изображения
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Изображение не найдено: {image_path}")
        print(f"💡 Проверьте путь или используйте аргумент -i для указания правильного пути")
        return
    
    # Обрабатываем изображение
    process_seal_image(
        image_path=image_path,
        output_file=args.output,
        language=args.language
    )

if __name__ == '__main__':
    main()
