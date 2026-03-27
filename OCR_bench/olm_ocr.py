import torch
import base64
import urllib.request
import math
import json

from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig

# from olmocr.data.renderpdf import render_pdf_to_base64png
# from olmocr.prompts import build_finetuning_prompt
# from olmocr.prompts.anchor import get_anchor_text

import cv2
import base64
import glob
import os
from pathlib import Path

# Initialize the model with memory optimization
print("Инициализация модели с оптимизацией памяти...")

# Проверяем доступность CUDA
if torch.cuda.is_available():
    print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
    print(f"GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Очищаем GPU память
    torch.cuda.empty_cache()
    
    try:
        # Пробуем загрузить с квантизацией
        print("Пробуем загрузить с 4-bit квантизацией...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview", 
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map="auto"
        ).eval()
        print("✅ Модель загружена с 4-bit квантизацией на GPU")
        
    except Exception as e:
        print(f"4-bit квантизация не удалась: {e}")
        print("Загружаем модель без квантизации с оптимизацией памяти...")
        
        # Загружаем без квантизации, но с оптимизацией
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "allenai/olmOCR-7B-0225-preview", 
            torch_dtype=torch.float16,  # Используем float16 для экономии памяти
            device_map="auto",
            low_cpu_mem_usage=True
        ).eval()
        print("✅ Модель загружена без квантизации на GPU")
    
else:
    print("CUDA недоступна, загружаем на CPU...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", 
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True
    ).eval()
    print("✅ Модель загружена на CPU")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Определяем устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Модель уже загружена на нужное устройство благодаря device_map="auto"

def get_ocr_text(path, lang):
    # Grab image
    img = cv2.imread(path)
    height, width, channels = img.shape
    new_height = height * 0.75
    new_height_text = str(float(new_height))
    new_height_text = new_height_text[0:new_height_text.find(".")+2]
    new_width = width * 0.75
    new_width_text = str(float(new_width))
    new_width_text = new_width_text[0:new_width_text.find(".")+2]
    jpg_img = cv2.imencode('.png', img)
    b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')


    # Render page 1 to an image
    image_base64 = b64_string

    prompt = "Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.\nDo not hallucinate.\n"
    prompt += f"RAW_TEXT_START\nPage dimensions: {new_width_text}x{new_height_text}\n[Image 0x0 to {str(math.ceil(new_width))}x{str(math.ceil(new_height))}]\n\nRAW_TEXT_END"

    #print("prompt: " + prompt)
    # Build the full prompt
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

    # Apply the chat template and processor
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}

    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
            )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    res = ""
    try:
        data = json.loads(text_output[0])
        res = data["natural_text"]
    except:
        print("Error with parsing json")
        print(text_output)
    if res == None:
        res = ""
    return res
    




def test(langs, unOCRed_image_paths):
    result = []
    for lang in langs:
        print(f"Checking {lang}")
        # Изменяем путь для работы с паспортными печатями
        images = glob.glob("dataset_passport_seal_ocr/*.jpg")
        for path in images:
            if not(path in unOCRed_image_paths):
                continue
            file_name = Path(path).name  # Используем .name вместо .stem для полного имени файла
            predicted_text = get_ocr_text(path, lang)
            print(predicted_text)

            data = (file_name, predicted_text, lang)
            result.append(data)
    return result

# Тест для паспортных печатей
def test_passport_seals():
    """Тестирует OLM OCR на паспортных печатях"""
    print("Запуск тестирования OLM OCR на паспортных печатях...")
    
    # Загружаем правильные ответы из Excel
    try:
        import pandas as pd
        df = pd.read_excel('db_seal_ocr.xlsx')
        answers_dict = {}
        for _, row in df.iterrows():
            file_name = row['name of the file']
            answer_text = row['text']
            if pd.notna(answer_text):
                answers_dict[file_name] = str(answer_text).strip()
        print(f"Загружено {len(answers_dict)} правильных ответов из Excel")
    except Exception as e:
        print(f"Не удалось загрузить Excel файл: {e}")
        answers_dict = {}
    
    # Получаем список всех изображений
    print("Поиск изображений...")
    
    # Пробуем разные пути
    possible_paths = [
        './dataset_passport_seal_ocr/*.jpg',
        'dataset_passport_seal_ocr/*.jpg',
        '../dataset_passport_seal_ocr/*.jpg',
        os.path.join(os.getcwd(), 'dataset_passport_seal_ocr', '*.jpg')
    ]
    
    image_paths = []
    for path_pattern in possible_paths:
        found = glob.glob(path_pattern)
        if found:
            image_paths = found
            print(f"✅ Найдено {len(image_paths)} изображений по пути: {path_pattern}")
            break
    
    if not image_paths:
        print("❌ Изображения не найдены!")
        print(f"Текущая директория: {os.getcwd()}")
        print(f"Проверяемые пути:")
        for path_pattern in possible_paths:
            print(f"  - {path_pattern}")
        return []
    
    print(f"📁 Первые 3 файла: {[os.path.basename(p) for p in image_paths[:3]]}")
    
    # Тестируем первые 3 изображения для проверки
    test_images = image_paths[:3]
    print(f"Тестируем первые {len(test_images)} изображения...")
    
    results = []
    total_similarity = 0
    valid_comparisons = 0
    
    for i, image_path in enumerate(test_images):
        file_name = Path(image_path).name
        print(f"\n[{i+1}/{len(test_images)}] Обрабатываем: {file_name}")
        
        try:
            # Получаем предсказание от OLM OCR
            predicted_text = get_ocr_text(image_path, "ru")
            print(f"Предсказание: {predicted_text}")
            
            # Получаем правильный ответ
            correct_answer = answers_dict.get(file_name, "")
            print(f"Правильный ответ: {correct_answer}")
            
            # Вычисляем сходство
            similarity = 0
            if correct_answer and predicted_text:
                try:
                    import jellyfish
                    similarity = jellyfish.jaro_similarity(predicted_text, correct_answer)
                    total_similarity += similarity
                    valid_comparisons += 1
                    print(f"Сходство: {similarity:.3f}")
                except ImportError:
                    print("jellyfish не установлен, сходство не вычисляется")
            else:
                print("Сходство: 0.000 (нет данных для сравнения)")
            
            results.append({
                'file_name': file_name,
                'predicted': predicted_text,
                'correct': correct_answer,
                'similarity': similarity
            })
            
        except Exception as e:
            print(f"Ошибка при обработке {file_name}: {e}")
            results.append({
                'file_name': file_name,
                'predicted': f"ОШИБКА: {e}",
                'correct': answers_dict.get(file_name, ""),
                'similarity': 0
            })
    
    # Выводим итоговые результаты
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    
    if valid_comparisons > 0:
        avg_similarity = total_similarity / valid_comparisons
        print(f"Среднее сходство: {avg_similarity:.3f}")
        print(f"Общее сходство: {total_similarity:.3f}")
        print(f"Количество сравнений: {valid_comparisons}")
    else:
        print("Нет данных для сравнения")
    
    # Сортируем по сходству
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nРезультаты по файлам:")
    for result in results:
        print(f"\nФайл: {result['file_name']}")
        print(f"Предсказание: {result['predicted']}")
        if result['correct']:
            print(f"Правильный ответ: {result['correct']}")
        print(f"Сходство: {result['similarity']:.3f}")
        print("-" * 40)
    
    return results

# Запускаем тест если файл запущен напрямую
if __name__ == "__main__":
    test_passport_seals()