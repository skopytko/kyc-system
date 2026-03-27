import sqlite3
import os.path
import glob
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import jellyfish
import pandas as pd

class PassportSealOCR_test:
    connection = None
    cursor = None
    table_name = "passport_seal_ocr"
    db_path = "passport_seal_ocr.db"
    excel_path = "db_seal_ocr.xlsx"
    image_folder = "dataset_passport_seal_ocr"
    font_name = "Geoform.ttf"
    text_size = 16
    colors = ["AntiqueWhite", "Aqua", "Aquamarine", "Azure", "DeepPink", "DeepSkyBlue", "BlueViolet", "Brown", "Cornsilk", "Chartreuse", "Chocolate", "Coral", "CornflowerBlue", "Crimson", "Cyan", "Gold", "DarkCyan", "DarkGoldenRod", "Fuchsia", "GreenYellow", "DarkKhaki", "HotPink", "GhostWhite", "LawnGreen", "LemonChiffon", "FloralWhite", "DarkSeaGreen", "DarkSlateBlue"]
    not_model_names = ["id", "file_name", "answer"]

    def __init__(self):
        # Загружаем правильные ответы из Excel файла
        self.load_answers_from_excel()

    def load_answers_from_excel(self):
        """Загружает правильные ответы из Excel файла"""
        try:
            df = pd.read_excel(self.excel_path)
            self.answers_dict = {}
            for _, row in df.iterrows():
                file_name = row['name of the file']
                answer_text = row['text']
                if pd.notna(answer_text):  # Проверяем, что ответ не NaN
                    self.answers_dict[file_name] = str(answer_text).strip()
            print(f"Загружено {len(self.answers_dict)} правильных ответов из Excel файла")
        except Exception as e:
            print(f"Ошибка при загрузке Excel файла: {e}")
            self.answers_dict = {}

    def make_db(self, model_names):
        """Создает базу данных с таблицей для паспортных печатей"""
        print("Создание базы данных для паспортных печатей")
        if os.path.isfile(self.db_path):
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            print("База данных уже существует, пропускаем создание")
            return
        
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()
        
        # Создаем таблицу с колонками для каждой модели
        sub_text = ""
        for i, model_name in enumerate(model_names):
            name = model_name.replace("-", "_")
            sub_text += f"{name} TEXT"
            if i != len(model_names) - 1:
                sub_text += ", "
        
        create_table_sql = f"""
        CREATE TABLE {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT NOT NULL UNIQUE,
            answer TEXT,
            {sub_text}
        )
        """
        
        self.cursor.execute(create_table_sql)
        self.connection.commit()
        print("База данных создана")

    def setup_db(self, model_names):
        """Настраивает базу данных и добавляет новые модели"""
        # Проверяем существующие колонки
        command = f"PRAGMA table_info({self.table_name})"
        self.cursor.execute(command)
        rows = self.cursor.fetchall()
        cur_model_names = [row[1] for row in rows]
        
        # Добавляем новые модели
        for m_name in model_names:
            if m_name not in cur_model_names:
                command = f"ALTER TABLE {self.table_name} ADD COLUMN {m_name} TEXT"
                print(f"Добавляем колонку: {command}")
                self.cursor.execute(command)
        
        self.connection.commit()
        
        # Добавляем записи для всех изображений с правильными ответами
        image_paths = glob.glob(f"{self.image_folder}/*.jpg")
        for path in image_paths:
            file_name = Path(path).name
            answer = self.answers_dict.get(file_name, "")
            
            command = f"INSERT OR IGNORE INTO {self.table_name} (file_name, answer) VALUES (?, ?)"
            self.cursor.execute(command, (file_name, answer))
        
        self.connection.commit()
        print(f"Добавлено {len(image_paths)} записей в базу данных")

    def find_unOCRed_image_paths(self, model_name):
        """Находит изображения, которые еще не были обработаны конкретной моделью"""
        result = []
        image_paths = glob.glob(f"{self.image_folder}/*.jpg")
        
        for path in image_paths:
            file_name = Path(path).name
            command = f"SELECT {model_name} FROM {self.table_name} WHERE file_name = ?"
            self.cursor.execute(command, (file_name,))
            rows = self.cursor.fetchall()
            
            if not rows or rows[0][0] is None or rows[0][0] == "":
                result.append(path)
        
        print(f"Найдено {len(result)} необработанных изображений для модели {model_name}")
        return result

    def update_db(self, model_name, predictions):
        """Обновляет базу данных результатами OCR"""
        for prediction in predictions:
            file_name, pred_text = prediction
            pred_text = pred_text.replace("'", "''")
            
            command = f"UPDATE {self.table_name} SET {model_name} = ? WHERE file_name = ?"
            self.cursor.execute(command, (pred_text, file_name))
        
        self.connection.commit()
        print(f"Обновлена база данных для модели {model_name}")

    def test_qwen2_vl_ocr(self):
        """Тестирует модель Qwen2-VL-OCR-2B-Instruct"""
        model_name = "qwen2_vl_ocr_2B_instruct"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Qwen2-VL-OCR")
            return
        
        try:
            import qwen2_vl_ocr_2B_instruct
            predictions = qwen2_vl_ocr_2B_instruct.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Qwen2-VL-OCR: {e}")

    def test_tesseract_ocr(self):
        """Тестирует модель Tesseract OCR"""
        model_name = "tesseract_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Tesseract")
            return
        
        try:
            import tesseract_ocr
            predictions = tesseract_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Tesseract: {e}")

    def test_easy_ocr(self):
        """Тестирует модель Easy OCR"""
        model_name = "easy_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Easy OCR")
            return
        
        try:
            import easy_ocr
            predictions = easy_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Easy OCR: {e}")

    def test_paddle_ocr(self):
        """Тестирует модель Paddle OCR"""
        model_name = "paddle_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Paddle OCR")
            return
        
        try:
            import paddle_ocr
            predictions = paddle_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Paddle OCR: {e}")

    def test_callisto_ocr(self):
        """Тестирует модель Callisto OCR"""
        model_name = "callisto_ocr3_2b_instruct"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Callisto OCR")
            return
        
        try:
            import callisto_ocr3_2b_instruct
            predictions = callisto_ocr3_2b_instruct.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Callisto OCR: {e}")

    def test_dddd_ocr(self):
        """Тестирует модель DDDD OCR"""
        model_name = "dddd_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью DDDD OCR")
            return
        
        try:
            import dddd_ocr
            predictions = dddd_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании DDDD OCR: {e}")

    def test_doctr_ocr(self):
        """Тестирует модель DocTR OCR"""
        model_name = "doctr_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью DocTR OCR")
            return
        
        try:
            import doctr_ocr
            predictions = doctr_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании DocTR OCR: {e}")

    def test_got_ocr(self):
        """Тестирует модель GOT OCR"""
        model_name = "got_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью GOT OCR")
            return
        
        try:
            import got_ocr
            predictions = got_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании GOT OCR: {e}")

    def test_granite_ocr(self):
        """Тестирует модель Granite OCR"""
        model_name = "granite_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Granite OCR")
            return
        
        try:
            import granite_ocr
            predictions = granite_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Granite OCR: {e}")

    def test_internVL3_2B(self):
        """Тестирует модель InternVL3 2B"""
        model_name = "internVL3_2B"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью InternVL3 2B")
            return
        
        try:
            import internVL3_2B
            predictions = internVL3_2B.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании InternVL3 2B: {e}")

    def test_internVL3_8B(self):
        """Тестирует модель InternVL3 8B"""
        model_name = "internVL3_8B"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью InternVL3 8B")
            return
        
        try:
            import internVL3_8B
            predictions = internVL3_8B.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании InternVL3 8B: {e}")

    def test_manga_ocr(self):
        """Тестирует модель Manga OCR"""
        model_name = "manga_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Manga OCR")
            return
        
        try:
            import manga_ocr1
            predictions = manga_ocr1.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Manga OCR: {e}")

    def test_olm_ocr(self):
        """Тестирует модель OLM OCR"""
        model_name = "olm_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью OLM OCR")
            return
        
        try:
            import olm_ocr
            predictions = olm_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании OLM OCR: {e}")

    def test_paligemma_ocr(self):
        """Тестирует модель Paligemma OCR"""
        model_name = "paligemma_3b_gt_ocrvqa_448"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Paligemma OCR")
            return
        
        try:
            import paligemma_3b_gt_ocrvqa_448
            predictions = paligemma_3b_gt_ocrvqa_448.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Paligemma OCR: {e}")

    def test_phi4_ocr(self):
        """Тестирует модель Phi4 OCR"""
        model_name = "phi4_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Phi4 OCR")
            return
        
        try:
            import phi4_ocr
            predictions = phi4_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Phi4 OCR: {e}")

    def test_rapid_ocr(self):
        """Тестирует модель Rapid OCR"""
        model_name = "rapid_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Rapid OCR")
            return
        
        try:
            import rapid_ocr
            predictions = rapid_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Rapid OCR: {e}")

    def test_tr_ocr(self):
        """Тестирует модель TR OCR"""
        model_name = "tr_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью TR OCR")
            return
        
        try:
            import tr_ocr
            predictions = tr_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании TR OCR: {e}")

    def test_windows_ocr(self):
        """Тестирует модель Windows OCR"""
        model_name = "windows_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Windows OCR")
            return
        
        try:
            import windows_ocr
            predictions = windows_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Windows OCR: {e}")

    def test_surya_ocr(self):
        """Тестирует модель Surya OCR"""
        model_name = "surya_ocr"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Surya OCR")
            return
        
        try:
            import surya_ocr
            predictions = surya_ocr.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Surya OCR: {e}")

    def test_miniCPM_ocr(self):
        """Тестирует модель MiniCPM OCR"""
        model_name = "miniCPM_o_2_6"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью MiniCPM OCR")
            return
        
        try:
            import miniCPM_o_2_6
            predictions = miniCPM_o_2_6.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании MiniCPM OCR: {e}")

    def test_mini_monkey(self):
        """Тестирует модель Mini Monkey"""
        model_name = "mini_monkey"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью Mini Monkey")
            return
        
        try:
            import mini_monkey
            predictions = mini_monkey.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании Mini Monkey: {e}")

    def test_h2ovl_ocr(self):
        """Тестирует модель H2OVL Mississippi 2B"""
        model_name = "h2ovl_mississippi_2b"
        unOCRed_image_paths = self.find_unOCRed_image_paths(model_name)
        
        if len(unOCRed_image_paths) == 0:
            print("Все изображения уже обработаны моделью H2OVL Mississippi 2B")
            return
        
        try:
            import h2ovl_mississippi_2b
            predictions = h2ovl_mississippi_2b.test(["ru"], unOCRed_image_paths)
            self.update_db(model_name, predictions)
        except Exception as e:
            print(f"Ошибка при тестировании H2OVL Mississippi 2B: {e}")

    def gen_comparison_images(self):
        """Генерирует сравнительные изображения с результатами всех моделей"""
        print("Генерация сравнительных изображений...")
        
        # Получаем список всех моделей
        command = f"PRAGMA table_info({self.table_name})"
        self.cursor.execute(command)
        rows = self.cursor.fetchall()
        model_names = [row[1] for row in rows if row[1] not in self.not_model_names]
        
        # Получаем все данные
        command = f"SELECT * FROM {self.table_name}"
        self.cursor.execute(command)
        rows = self.cursor.fetchall()
        
        # Создаем папку для результатов
        output_folder = "passport_seal_ocr_reports"
        os.makedirs(output_folder, exist_ok=True)
        
        # Словарь для подсчета очков
        model_scores = {model: 0.0 for model in model_names}
        
        for row in rows:
            file_name = row[1]
            answer = row[2]
            
            # Создаем изображение-сравнение
            width, height = 1920, 1080
            result_img = Image.new('RGBA', (width, height), 'black')
            
            # Загружаем оригинальное изображение
            image_path = f"{self.image_folder}/{file_name}"
            if os.path.exists(image_path):
                img = Image.open(image_path).convert("RGBA")
                img_width, img_height = img.size
                result_img.paste(img, (0, 0), img)
                
                # Добавляем текст с результатами
                idraw = ImageDraw.Draw(result_img)
                try:
                    font = ImageFont.truetype(self.font_name, size=self.text_size)
                except:
                    font = ImageFont.load_default()
                
                cur_y = img_height + 10
                cur_x = 10
                
                # Добавляем правильный ответ
                answer_text = f"ПРАВИЛЬНЫЙ ОТВЕТ: {answer}"
                idraw.text((cur_x, cur_y), answer_text, font=font, fill="White")
                cur_y += self.text_size + 5
                
                # Добавляем результаты моделей
                for i, model_name in enumerate(model_names):
                    if i < len(row) - 3:  # -3 для id, file_name, answer
                        model_prediction = str(row[i + 3])
                        model_prediction = model_prediction.replace("\n", " ")
                        
                        if model_prediction and model_prediction != "None":
                            # Вычисляем сходство
                            similarity = jellyfish.jaro_similarity(model_prediction, answer)
                            model_scores[model_name] += similarity
                            similarity_rounded = round(similarity, 3)
                            
                            text = f"{model_name}: ({similarity_rounded}) {model_prediction}"
                            color = self.colors[i % len(self.colors)]
                            idraw.text((cur_x, cur_y), text, font=font, fill=color)
                            cur_y += self.text_size
                
                # Сохраняем изображение
                output_path = f"{output_folder}/{file_name.replace('.jpg', '_comparison.png')}"
                result_img.save(output_path)
        
        # Создаем итоговый рейтинг
        print("\nИтоговый рейтинг моделей:")
        sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        for model, score in sorted_scores:
            print(f"{model}: {score:.3f}")
        
        # Сохраняем рейтинг в файл
        with open(f"{output_folder}/final_ranking.txt", "w", encoding="utf-8") as f:
            f.write("Итоговый рейтинг моделей OCR для паспортных печатей:\n")
            f.write("=" * 60 + "\n\n")
            for model, score in sorted_scores:
                f.write(f"{model}: {score:.3f}\n")
        
        print(f"\nРезультаты сохранены в папку: {output_folder}")

    def start(self):
        """Основной метод запуска"""
        # Список всех доступных моделей
        model_names = [
            "callisto_ocr3_2b_instruct", "dddd_ocr", "doctr_ocr", "easy_ocr", 
            "got_ocr", "granite_ocr", "internVL3_2B", "internVL3_8B", "manga_ocr", 
            "olm_ocr", "paddle_ocr", "paligemma_3b_gt_ocrvqa_448", "phi4_ocr", 
            "qwen2_vl_ocr_2B_instruct", "rapid_ocr", "tesseract_ocr", "tr_ocr", 
            "windows_ocr", "surya_ocr", "miniCPM_o_2_6", "mini_monkey", 
            "h2ovl_mississippi_2b"
        ]
        
        print("Запуск тестирования OCR для паспортных печатей")
        print(f"Папка с изображениями: {self.image_folder}")
        print(f"Excel файл с ответами: {self.excel_path}")
        
        # Создаем и настраиваем базу данных
        self.make_db(model_names)
        self.setup_db(model_names)
        
        # Запускаем все тесты
        print("Запуск тестирования всех моделей...")
        
        # Основные модели
        self.test_qwen2_vl_ocr()
        self.test_tesseract_ocr()
        self.test_easy_ocr()
        self.test_paddle_ocr()
        
        # Дополнительные модели
        self.test_callisto_ocr()
        self.test_dddd_ocr()
        self.test_doctr_ocr()
        self.test_got_ocr()
        self.test_granite_ocr()
        self.test_internVL3_2B()
        self.test_internVL3_8B()
        self.test_manga_ocr()
        self.test_olm_ocr()
        self.test_paligemma_ocr()
        self.test_phi4_ocr()
        self.test_rapid_ocr()
        self.test_tr_ocr()
        self.test_windows_ocr()
        self.test_surya_ocr()
        self.test_miniCPM_ocr()
        self.test_mini_monkey()
        self.test_h2ovl_ocr()
        
        # Генерируем сравнительные отчеты
        self.gen_comparison_images()
        
        print("Тестирование завершено!")

if __name__ == "__main__":
    PassportSealOCR_test().start()
