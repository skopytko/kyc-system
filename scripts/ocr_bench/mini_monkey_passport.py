from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import glob
import os
from pathlib import Path
import pandas as pd
import jellyfish

# Используем CPU для локального запуска
device = "cpu"
print(f"Используем устройство: {device}")

langs_dict = {
    "ja": "Japanese",
    "ru": "Russian",
    "en": "English",
    "cn": "Chinese",
    "ko": "Korean"
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio

def dynamic_preprocess2(image, min_num=1, max_num=12, prior_aspect_ratio=None, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    new_target_ratios = []
    for i in target_ratios:
        if prior_aspect_ratio[0]%i[0] or prior_aspect_ratio[1]%i[1]:
            new_target_ratios.append(i)
        else:
            continue
    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, new_target_ratios, orig_width, orig_height, image_size)
    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, min_num=1, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_aspect_ratio = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_aspect_ratio

def load_image2(image_file, input_size=448, min_num=1, max_num=12, target_aspect_ratio=None):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess2(image, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num, prior_aspect_ratio=target_aspect_ratio)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_ocr_text(image_path, lang):
    try:
        # Проверяем, загружена ли модель
        if 'model' not in globals() or model is None:
            return "ОШИБКА: Модель не загружена"
        
        # set the max number of tiles in `max_num`
        pixel_values, target_aspect_ratio = load_image(image_path, min_num=4, max_num=12)
        pixel_values = pixel_values.to(torch.float16).to(device)  # Используем float16
        pixel_values2 = load_image2(image_path, min_num=3, max_num=7, target_aspect_ratio=target_aspect_ratio)
        pixel_values2 = pixel_values2.to(torch.float16).to(device)  # Используем float16
        pixel_values = torch.cat([pixel_values2[:-1], pixel_values[:-1], pixel_values2[-1:]], 0)

        generation_config = dict(do_sample=False, max_new_tokens=512)

        question = f"Give me text from image, writen in {langs_dict[lang]} language, nothing else."
        response, history = model.chat(tokenizer, pixel_values, target_aspect_ratio, question, generation_config, history=None, return_history=True)
        print(f'User: {question} Assistant: {response}')
        return response
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return f"ОШИБКА: {e}"

def test_passport_seals():
    """Тестирует MiniMonkey на паспортных печатях"""
    print("🎯 Запуск тестирования MiniMonkey на паспортных печатях...")
    
    # Загружаем правильные ответы из Excel
    try:
        # Пробуем разные пути для Excel файла
        excel_paths = [
            'db_seal_ocr.xlsx',  # Локальный путь
            './db_seal_ocr.xlsx',
            '/content/dataset_passport_seal_ocr/db_seal_ocr.xlsx',
            '/content/drive/MyDrive/db_seal_ocr.xlsx'
        ]
        
        excel_loaded = False
        for excel_path in excel_paths:
            if os.path.exists(excel_path):
                print(f"📁 Найден Excel файл: {excel_path}")
                df = pd.read_excel(excel_path)
                answers_dict = {}
                for _, row in df.iterrows():
                    file_name = row['name of the file']
                    answer_text = row['text']
                    if pd.notna(answer_text):
                        answers_dict[file_name] = str(answer_text).strip()
                print(f"✅ Загружено {len(answers_dict)} правильных ответов из Excel")
                excel_loaded = True
                break
        
        if not excel_loaded:
            print("❌ Excel файл не найден ни по одному из путей")
            answers_dict = {}
            
    except Exception as e:
        print(f"❌ Не удалось загрузить Excel файл: {e}")
        answers_dict = {}
    
    # Получаем список всех изображений
    print("🔍 Поиск изображений...")
    
    # Пробуем разные пути для Colab
    possible_paths = [
        '/content/dataset_passport_seal_ocr/*.jpg',
        '/content/drive/MyDrive/dataset_passport_seal_ocr/*.jpg',
        './dataset_passport_seal_ocr/*.jpg'
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
        return []
    
    print(f"📁 Первые 3 файла: {[os.path.basename(p) for p in image_paths[:3]]}")
    
    # Тестируем все изображения
    print(f"🚀 Тестируем все {len(image_paths)} изображений...")
    
    results = []
    total_similarity = 0
    valid_comparisons = 0
    
    for i, image_path in enumerate(image_paths):
        file_name = os.path.basename(image_path)
        print(f"\n[{i+1}/{len(image_paths)}] Обрабатываем: {file_name}")
        
        try:
            # Получаем предсказание от MiniMonkey
            print("🔄 Распознаем текст...")
            predicted_text = get_ocr_text(image_path, "ru")
            print(f"📝 Предсказание: {predicted_text}")
            
            # Получаем правильный ответ
            correct_answer = answers_dict.get(file_name, "")
            if correct_answer:
                print(f"✅ Правильный ответ: {correct_answer}")
            else:
                print(f"⚠️ Правильный ответ не найден для {file_name}")
            
            # Вычисляем сходство
            similarity = 0
            if correct_answer and predicted_text:
                try:
                    similarity = jellyfish.jaro_similarity(predicted_text, correct_answer)
                    total_similarity += similarity
                    valid_comparisons += 1
                    print(f"🎯 Сходство: {similarity:.3f}")
                except Exception as e:
                    print(f"⚠️ Ошибка вычисления сходства: {e}")
            else:
                print("❌ Сходство: 0.000 (нет данных для сравнения)")
            
            results.append({
                'file_name': file_name,
                'predicted': predicted_text,
                'correct': correct_answer,
                'similarity': similarity
            })
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {file_name}: {e}")
            results.append({
                'file_name': file_name,
                'predicted': f"ОШИБКА: {e}",
                'correct': answers_dict.get(file_name, ""),
                'similarity': 0
            })
    
    # Выводим итоговые результаты
    print("\n" + "="*60)
    print("🏆 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    
    if valid_comparisons > 0:
        avg_similarity = total_similarity / valid_comparisons
        print(f"📊 Среднее сходство: {avg_similarity:.3f}")
        print(f"📈 Общее сходство: {total_similarity:.3f}")
        print(f"🔢 Количество сравнений: {valid_comparisons}")
    else:
        print("❌ Нет данных для сравнения")
    
    # Сортируем по сходству
    results.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\n📋 Топ-5 лучших результатов:")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. {result['file_name']}: {result['similarity']:.3f}")
    
    print("\n📋 Топ-5 худших результатов:")
    for i, result in enumerate(results[-5:]):
        print(f"{i+1}. {result['file_name']}: {result['similarity']:.3f}")
    
    # Сохраняем результаты в файл
    with open('minimonkey_passport_results.txt', 'w', encoding='utf-8') as f:
        f.write("Результаты тестирования MiniMonkey на паспортных печатях\n")
        f.write("="*60 + "\n\n")
        if valid_comparisons > 0:
            f.write(f"Среднее сходство: {avg_similarity:.3f}\n")
            f.write(f"Общее сходство: {total_similarity:.3f}\n")
            f.write(f"Количество сравнений: {valid_comparisons}\n\n")
        
        f.write("Детальные результаты:\n")
        for result in results:
            f.write(f"\nФайл: {result['file_name']}\n")
            f.write(f"Предсказание: {result['predicted']}\n")
            if result['correct']:
                f.write(f"Правильный ответ: {result['correct']}\n")
            f.write(f"Сходство: {result['similarity']:.3f}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\n📄 Результаты сохранены в файл: minimonkey_passport_results.txt")
    return results

# Загружаем модель на CPU
print("🔄 Загружаем модель MiniMonkey на CPU...")
try:
    path = 'mx262/MiniMonkey'
    
    # Пробуем загрузить с игнорированием ошибок конфигурации
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cpu",
        ignore_mismatched_sizes=True,
        local_files_only=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("✅ Модель загружена!")
    
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    print("🔄 Пробуем загрузить с минимальными параметрами...")
    
    try:
        # Минимальная загрузка
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            local_files_only=False
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        print("✅ Модель загружена с минимальными параметрами!")
        
    except Exception as e2:
        print(f"❌ Критическая ошибка: {e2}")
        print("🔄 Пробуем загрузить только токенизатор...")
        
        # Загружаем только токенизатор для тестирования
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        print("✅ Токенизатор загружен, но модель не загрузилась!")
        print("⚠️ Тест будет работать только с токенизатором")
        
        # Создаем заглушку для модели
        model = None

# Запускаем тест если файл запущен напрямую
if __name__ == "__main__":
    test_passport_seals()
