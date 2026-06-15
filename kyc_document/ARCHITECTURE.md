# Архитектура модуля `kyc_document`

> **Версия пакета:** 1.0.1  
> **Публичный API:** `from kyc_document.document_processing import Pipeline`

---

## 1. Общая структура файлов

```
kyc_document/
├── ARCHITECTURE.md                          # Этот документ
├── document_processing/
│   ├── __init__.py                          # Экспорт Pipeline, __version__
│   ├── config/
│   │   ├── __init__.py                      # Загрузка DEFAULT_CFG из YAML
│   │   └── models_path.yaml                 # Пути к весам моделей
│   ├── pipeline/
│   │   └── pipeline.py                      # Pipeline, PipelineResults, OCROptions*
│   ├── processing/                          # Фреймворк загрузки и инференса моделей
│   │   ├── __init__.py                      # re-export ModelLoader
│   │   ├── inference.py                     # ModelInference (ONNX/TF/OV/CoreML/TFLite)
│   │   ├── models.py                        # ModelLoader + Model / ClassificationModel /
│   │   │                                    #   OCRModel / YoloDetectorModel / YoloSegmentorModel
│   │   ├── preprocessing.py                 # Base / Classification / Yolo / OCR Preprocessing
│   │   └── postprocessing.py                # Base / BinaryClass / OCR / Metric /
│   │                                        #   MultiClass / YoloDetector / YoloSegmentor Postprocessing
│   ├── pipeline_modules/                    # Бизнес-модули пайплайна
│   │   ├── __init__.py                      # Экспорт всех модулей
│   │   ├── base_module.py                   # BaseModule — базовый класс
│   │   ├── angles_classificator/
│   │   │   └── angles_classificator.py      # Angle90
│   │   ├── doc_detector/
│   │   │   ├── doc_detector.py              # DocDetector (YOLO-seg PT / ONNX)
│   │   │   └── image_transformation.py      # fix_perspective, order_points, _quad_from_hull
│   │   ├── doctype_classificator/
│   │   │   └── doc_type.py                  # DocType (MobileNetV3)
│   │   ├── passport_page_type/
│   │   │   └── passport_page_type.py        # PassportPageType (MobileNetV3-Small PT)
│   │   ├── passport_seal_detector/
│   │   │   ├── passport_seal_detector.py    # PassportSealDetector (YOLO-seg PT)
│   │   │   └── seal_processing.py           # process_seals, select_last_seal и утилиты
│   │   ├── textfields_detector/
│   │   │   ├── __init__.py                  # Экспорт TextFieldsDetector, *Belarus, *USA
│   │   │   ├── russia/passport/
│   │   │   │   └── textfields_detector.py   # TextFieldsDetector (ONNX через BaseModule)
│   │   │   ├── belarus/passport/
│   │   │   │   └── textfields_detector.py   # TextFieldsDetectorBelarus (YOLO PT)
│   │   │   └── usa/
│   │   │       └── textfields_detector.py   # TextFieldsDetectorUSA (YOLO PT per-state)
│   │   ├── words_detector/
│   │   │   └── words_detector.py            # WordsDetector (YOLO ONNX)
│   │   ├── ocr_rus/
│   │   │   └── ocr_rus.py                   # OCRRus (CRNN ONNX)
│   │   ├── ocr_engnums/
│   │   │   └── ocr_engnums.py               # OCREngNums (CRNN ONNX)
│   │   ├── blur_detector/
│   │   │   ├── blur.py                      # Blur
│   │   │   └── quality.py                   # QualityChecker (разбивка на патчи)
│   │   ├── glare_detector/
│   │   │   ├── glare.py                     # Glare
│   │   │   └── quality.py                   # QualityChecker (аналогичный blur)
│   │   ├── lcd_spoofing_detector/
│   │   │   └── lcdspoofing.py               # LCDSpoofing
│   │   └── print_spoofing_detector/
│   │       └── printspoofing.py             # PrintSpoofing
│   └── models/                              # Веса и конфиги моделей ML
│       ├── Angles90/{ONNX,TFlite,...}/model.json + веса
│       ├── Borders/best.pt                  # DocDetector PT
│       ├── DocType/{ONNX,...}/model.json
│       ├── OCR/eng+nums/{ONNX,...}/model.json
│       ├── OCR/rus/{ONNX,...}/model.json
│       ├── Words/{ONNX,...}/model.json
│       ├── Blur/{ONNX,...}/model.json
│       ├── Glare/{ONNX,...}/model.json
│       ├── LCDSpoofing/{ONNX,...}/model.json
│       ├── PrintSpoofing/{ONNX,...}/model.json
│       ├── TextFields/
│       │   ├── russia/passport/{ONNX,...}/model.json
│       │   ├── belarus/passport/PT/best.pt
│       │   └── usa/{state}/PT/best.pt       # Отдельная модель на штат
│       ├── PassportPageType/
│       │   ├── russia/best.pt
│       │   └── belarus/best.pt
│       └── PassportSeal/
│           ├── russia/best.pt
│           └── belarus/best.pt
└── scripts/                                 # CLI-утилиты
    ├── process_img.py                       # Обработка одного изображения
    ├── process_video.py                     # Захват с веб-камеры / видео
    ├── benchmark.py                         # Массовый прогон + CSV-отчёт
    ├── ocr_patches.py                       # Экспорт патчей слов для дообучения
    ├── test_bbox_detection.py               # Оценка IoU/P/R/F1 TextFieldsDetector
    ├── test_ocr_accuracy.py                 # Сравнение OCR с разметкой
    ├── test_doc_borders_on_dataset2.py      # Оценка IoU DocDetector vs VIA
    └── augment_dataset.py                   # Аугментации для VIA-датасета
```

---

## 2. Конфигурация и загрузка путей к моделям

### 2.1. `config/models_path.yaml`

Содержит маппинг «логическое имя модуля → относительный путь к каталогу с весами»:

| Ключ                      | Путь                                  |
|---------------------------|---------------------------------------|
| `Angle90`                 | `models\Angles90`                     |
| `DocDetector`             | `models\Borders`                      |
| `TextFieldsDetectorRussia`| `models\TextFields\russia\passport`  |
| `TextFieldsDetectorBelarus`| `models\TextFields\belarus\passport`|
| `DocType`                 | `models\DocType`                      |
| `Blur`                    | `models\Blur`                         |
| `Glare`                   | `models\Glare`                        |
| `LCDSpoofing`            | `models\LCDSpoofing`                 |
| `PrintSpoofing`          | `models\PrintSpoofing`               |
| `WordsDetector`           | `models\Words`                        |
| `OCREngNums`              | `models\OCR\eng+nums`                |
| `OCRRus`                  | `models\OCR\rus`                      |

### 2.2. `config/__init__.py`

При импорте:
1. Читает `models_path.yaml` через `yaml.safe_load`.
2. Для каждого значения: если путь не абсолютный, разрешает его относительно `document_processing/` (корень пакета).
3. Экспортирует словарь `DEFAULT_CFG`, который используется в `BaseModule` для нахождения `model.json`.

---

## 3. Фреймворк инференса (`processing/`)

### 3.1. `ModelInference` — универсальный инференс

Класс принимает путь к файлу модели и по расширению файла выбирает бэкенд:

| Расширение      | Бэкенд             | Метод predict          |
|-----------------|---------------------|------------------------|
| `.h5`           | TensorFlow Keras    | `__predict_saved_model`|
| каталог         | TF SavedModel       | `__predict_saved_model`|
| `.pb`           | TF Frozen Graph     | `__predict_frozen`     |
| `.tflite`       | TF Lite             | `__predict_tflite`     |
| `.onnx`         | ONNX Runtime        | `__predict_onnx`       |
| `.ir`           | OpenVINO            | `__predict_openvino`   |
| `.mlmodel`      | CoreML (macOS only) | `__predict_coreml`     |

Зависимости (`tensorflow`, `openvino`, `coremltools`) импортируются лениво через `importlib.import_module`, что позволяет работать только с установленным бэкендом.

Особенности ONNX-инференса: автоматическая транспозиция NHWC → NCHW, если модель ожидает каналы вторым измерением. При инициализации выполняется прогрев (dummy predict), чтобы первый реальный вызов не был медленным. Поддержка GPU через `CUDAExecutionProvider` с fallback на CPU.

### 3.2. `ModelLoader` — фабрика моделей

Читает JSON-конфиг (`model.json`) и по полю `Type` создаёт соответствующий объект:

| Значение `Type`            | Создаваемый класс     | Preprocessing              | Postprocessing               |
|----------------------------|------------------------|----------------------------|------------------------------|
| `Metric`                   | `ClassificationModel`  | `ClassificationPreprocessing` | `MetricPostprocessing`       |
| `BinaryClassification`     | `ClassificationModel`  | `ClassificationPreprocessing` | `BinaryClassPostprocessing`  |
| `MultiLabelClassification` | `ClassificationModel`  | `ClassificationPreprocessing` | `MultiClassPostprocessing`   |
| `Classifier`               | `ClassificationModel`  | `ClassificationPreprocessing` | `MultiClassPostprocessing`   |
| `YoloDetector`             | `YoloDetectorModel`    | `YoloPreprocessing`           | `YoloDetectorPostprocessing` |
| `YoloSegmentor`            | `YoloSegmentorModel`   | `YoloPreprocessing`           | `[YoloDetectorPP, YoloSegmentorPP]` |
| `OCR`                      | `OCRModel`             | `OCRPreprocessing`            | `OCRPostprocessing`          |

### 3.3. Иерархия Model-классов

```
Model (абстрактный)
├── ClassificationModel      # predict: preprocessing → inference → postprocessing
├── OCRModel                 # predict: preprocessing → expand_dims → inference → postprocessing
├── YoloDetectorModel        # predict: preprocessing → inference → NMS + координаты
└── YoloSegmentorModel       # predict: preprocessing → inference → NMS + маски + сегменты
```

Каждый реализует:
- `predict(img)` — полный пайплайн (preprocessing → inference → postprocessing).
- `predict_fv(img)` — пайплайн без постобработки (feature vector / raw output).

### 3.4. Preprocessing

| Класс                       | Вход           | Что делает                                                                                  |
|-----------------------------|----------------|---------------------------------------------------------------------------------------------|
| `BasePreprocessing`         | Path / ndarray | Загрузка BGR→RGB, padding, нормализация                                                    |
| `ClassificationPreprocessing` | —            | + resize к `image_size`, + batch dim                                                        |
| `YoloPreprocessing`         | —              | + letterbox-resize с сохранением пропорций и stride-выравниванием (32px), возвращает метаданные паддинга |
| `OCRPreprocessing`          | —              | Конвертация в grayscale, resize к 31×200 с сохранением пропорций, padding до фиксированного размера |

### 3.5. Postprocessing

| Класс                        | Назначение                                                                                      |
|------------------------------|------------------------------------------------------------------------------------------------|
| `BinaryClassPostprocessing`  | Порог → метка из двух классов + confidence                                                     |
| `MultiClassPostprocessing`   | `argmax` → метка класса + max probability                                                      |
| `OCRPostprocessing`          | CTC-декодирование: индексы → символы по алфавиту (`eng`: цифры+A-Z+спецсимволы; `rus`: А-Я+`.-`) |
| `MetricPostprocessing`       | Pickle-центроиды + sklearn NearestNeighbors, cosine/euclidean, radius-based rejection → метка/NONE |
| `YoloDetectorPostprocessing` | Фильтрация по confidence, xywh→xyxy, NMS по IoU, обратное масштабирование координат, добавление текстовых меток |
| `YoloSegmentorPostprocessing`| Matmul proto-masks × mask-коэффициенты, sigmoid, resize, clip по bbox, бинаризация, contour → segments |

---

## 4. Базовый класс модулей: `BaseModule`

Каждый доменный модуль (Angle90, DocType, Blur и т.д.) наследуется от `BaseModule`, который:

1. По имени модуля ищет путь в `DEFAULT_CFG`.
2. Формирует путь к `model.json`: `{путь}/{model_format}/model.json`.
3. Вызывает `ModelLoader` для создания объекта модели.
4. Предоставляет:
   - `model` — загруженная модель.
   - `model_info` — содержимое `model.json` (словарь).
   - `model_name` — строковое имя модуля.
   - `load_img(img)` — загрузка и конвертация в RGB.
   - `predict(img)` / `predict_transform(img)` — виртуальные методы.

---

## 5. Модули пайплайна (pipeline_modules)

### 5.1. Angle90 — определение угла поворота (произвольный угол, zero-shot)

Гибридный детектор: алгоритмический skew (свободный угол ±30°) + квадрант (0/90/180/270).

- **Шаг 1: Skew detection (произвольный угол).** Классический алгоритм **projection profile** без обучения:
  1. Изображение в grayscale, downscale до 700px, Otsu-бинаризация, morphology close (15×3) для «соединения» текстовых строк.
  2. **Грубый поиск:** перебор углов в диапазоне ±30° с шагом 1°, поворот бинарной картинки, для каждой считается дисперсия `np.diff` горизонтальной проекции. Чем строки текста точнее параллельны горизонтали — тем выше score.
  3. **Уточнение:** в окне ±1.5° вокруг лучшего грубого угла с шагом 0.1°.
  4. Дополнительная sanity-проверка через **HoughLinesP** (медиана углов с MAD-фильтрацией). Если совпадает с projection profile в пределах 2° — confidence повышается до 1.0.
  5. Возвращает correction angle: `cv2.warpAffine` с расширением канвы (без обрезки документа).
- **Шаг 2: Квадрант (0/90/180/270).** Применяется к уже выровненному изображению:
  - **Первый источник: pytesseract OSD** (zero-shot, без обучения) — если установлен `tesseract` и `pip install pytesseract`.
  - **Fallback: ONNX-классификатор EfficientNet-B3** (старый Angle90, остаётся в репозитории).
  - Применяется только если `confidence ≥ 0.6` (порог `quadrant_conf_threshold`) — защита от ложных переворотов на 180° при низкой уверенности.
- **`predict(img)`** → `{angle, confidence, skew_deg, quadrant_deg, quadrant_source}`.
- **`predict_transform(img)`** → дополнительно `warped_img` (повёрнутое изображение).
- **Точность на синтетике:** ±0.2° для произвольных углов в диапазоне ±15°.
- **Скорость:** ~0.8–1.5 с на изображение на CPU (без квадранта через ONNX); ~1.5–2 с с ONNX-классификатором.

### 5.2. DocDetector — детекция границ документа

- **Основной режим (PT):** Ultralytics YOLO сегментационная модель (`models/Borders/best.pt`), `conf=0.1`.
- **Альтернативный режим (ONNX и др.):** через `BaseModule` (ONNX-сегментатор из model.json).
- **Логика PT-режима:**
  1. Запускает `self.model(img_bgr)`, получает маски и bbox.
  2. Для каждой маски: resize до размера изображения, бинаризация (порог 0.5), вычисление доли площади `area_frac`.
  3. Фильтрация: `area_frac < MIN_MASK_AREA_FRAC (0.08)` → отбрасывается.
  4. Сортировка кандидатов по `(area_frac, confidence)`, берутся до 2 лучших.
  5. Из бинарных масок извлекаются контуры (`cv2.findContours`), выбирается максимальный по площади.
- **`predict_transform(img)`:** вызывает `fix_perspective(img, segments)` из `image_transformation.py`.

### 5.3. `image_transformation.py` — исправление перспективы

- **`_quad_from_hull(hull)`:** аппроксимация выпуклой оболочки контура до 4 точек. Итерирует epsilon от 0.005 до 0.35 для `cv2.approxPolyDP`, проверяет что площадь четырёхугольника ≥ 75% от hull. Fallback на `cv2.minAreaRect`.
- **`order_points(pts)`:** упорядочивание 4 точек: top-left (min sum), top-right (min diff), bottom-right (max sum), bottom-left (max diff).
- **`fix_perspective(img, segments)`:**
  1. Для каждого сегмента: clip к границам изображения, рисование маски, `convexHull` → `_quad_from_hull` → `order_points`.
  2. Вычисление целевого размера по максимальной ширине/высоте сторон четырёхугольника.
  3. `cv2.getPerspectiveTransform` + `cv2.warpPerspective`.
  4. Если найдено 2 области: определяется направление расположения (вертикально / горизонтально по разнице центров Y/X) и склеивается через `np.vstack` или `np.hstack`.
  5. Возвращает `(warped_image, annotated_image_with_contours)`.

### 5.4. DocType — классификация типа документа

- **Архитектура:** MobileNetV3-Small (через ONNX), 14 классов.
- **Препроцессинг:** транспозиция NHWC→NCHW, `/255`, ImageNet mean/std, softmax вручную.
- **Классы (из model.json):** `belarus_passport_1996`, `belarus_passport_2019_2021`, `russia_passport`, `usa_alabama`, `usa_alaska`, `usa_arizona`, `usa_arkansas`, `usa_idaho`, `usa_iowa`, `usa_vermont`, `usa_virginia`, `usa_washington`, `usa_wisconsin`, `usa_wyoming`.
- **Rejection-логика:**
  - env `KYC_DOCTYPE_MIN_CONF` / `KYC_DOCTYPE_MIN_MARGIN` (legacy `RDOCR_*`).
  - Если `max(prob) < min_conf` или `top1 - top2 < min_margin` → `doc_type = "NONE"`.

### 5.5. PassportPageType — тип страницы паспорта

- **Архитектура:** MobileNetV3-Small (PyTorch), 2 класса: `['passport_centerfold', 'passport_pages']`.
- **Веса per-country:** `russia/best.pt`, `belarus/best.pt`.
- **Препроцессинг:** `torchvision.transforms`: Resize(224), ToTensor, Normalize(ImageNet).
- **`predict(img)`** → строка: `'passport_centerfold'` или `'passport_pages'`.
- **Кеширование:** в Pipeline модели кэшируются в `_passport_page_type_models[country]`.

### 5.6. PassportSealDetector — детекция печатей

- **Архитектура:** YOLOv8 сегментационная модель (PT), класс `seal`.
- **Веса per-country:** `russia/best.pt`, `belarus/best.pt`.
- **`predict(img)`:**
  1. YOLO-инференс → bboxes + masks + confidences.
  2. Для каждой маски: resize, бинаризация, аппроксимация до quad (`_approximate_seal_to_quad`).
  3. Вычисление центра и стороны страницы (`left`/`right`).
  4. Вызов `process_seals()` → вычисление приоритетов (правая + ниже = последнее), сортировка.
  5. `select_last_seal()` → печать с максимальным приоритетом.
- **`predict_transform(img)`:** + рисование полупрозрачных полигонов с подписями, вырезка последней печати с исправлением перспективы.

### 5.7. `seal_processing.py` — утилиты обработки печатей

| Функция                   | Назначение                                                                          |
|---------------------------|-------------------------------------------------------------------------------------|
| `determine_page_side`     | `center_x < w/2` → `'left'`, иначе `'right'`                                       |
| `calculate_seal_priority` | Левая: `norm_y`; правая: `norm_y + 1.0` (правая всегда «позже» левой)               |
| `process_seals`           | Добавляет `priority`, сортирует descending (последняя печать первая)                  |
| `select_last_seal`        | Первый элемент отсортированного списка                                               |
| `approximate_seal_to_quad`| Контуры → approxPolyDP → 4 точки или minAreaRect                                    |
| `validate_seal_quad`      | Проверка: 4 уникальные точки, площадь > 0                                           |
| `order_seal_points`       | Сортировка по углам от центра (`arctan2`)                                            |

### 5.8. TextFieldsDetector — детекция текстовых полей

#### 5.8.1. Россия (`TextFieldsDetector` / `TextFieldsDetectorRussia`)

- Наследуется от `BaseModule`, использует ONNX YOLO-детектор (`models/TextFields/russia/passport`).
- **`_safe_crop(img, box)`:** безопасная вырезка с привязкой к границам кадра, отбрасывание пустых/инвертированных боксов.
- **`predict(img)`** → список bbox.
- **`predict_transform(img)`** → bbox + список патчей (вырезки полей).

#### 5.8.2. Беларусь (`TextFieldsDetectorBelarus`)

- Использует Ultralytics YOLO напрямую (`models/TextFields/belarus/passport/PT/best.pt`), `conf=0.01`.
- **17 классов полей:** `authority`, `authority2`, `code_of_issuing`, `date_of_birth`, `date_of_expiry`, `date_of_issue`, `identification_no`, `names`, `nationality`, `passport_no`, `photo`, `place_of_birth`, `sex`, `signature`, `signature2`, `surname`, `type`.
- **`_parse_and_filter(results, labels)`:** для каждого класса сохраняется только bbox с максимальной confidence (top-1 per class).

#### 5.8.3. USA (`TextFieldsDetectorUSA`)

- Per-state YOLO модели: `models/TextFields/usa/{state}/PT/best.pt`, `conf=0.04`.
- **11 поддерживаемых штатов:** Alabama, Alaska, Arizona, Arkansas, Idaho, Iowa, Vermont, Virginia, Washington, Wisconsin, Wyoming.
- **20 классов:** 3 image-поля (`photo`, `mini_photo`, `handwritten_signature`) + 17 текстовых (`class`, `end`, `rest`, `firstname`, `lastname`, `address`, `sex`, `hgt`, `wgt`, `eyes`, `hair`, `dd`, `dln`, `iss`, `iss_duplicate`, `exp`, `dob`).
- **Особая логика для `address` (class_id=8):** все bbox с `conf ≥ 0.5` сохраняются (многострочный адрес), сортируются по Y затем X. Остальные классы — top-1 per class.

### 5.9. WordsDetector — детекция слов

- ONNX YOLO-детектор (`models/Words`).
- **`predict(img)`** → список bbox слов.
- **`predict_transform(img)`** → bbox отсортированные по X (слева направо) + патчи (вырезки отдельных слов).

### 5.10. OCRRus — распознавание русского текста

- CRNN модель через ONNX (`models/OCR/rus`).
- **Алфавит:** `АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ.-` (35 символов).
- **`predict(img)`** → распознанная строка.
- **`fix_errors(field_type, text)`:**
  - Имена / места / отчества: `lstrip('.')`.
  - Пол (`Sex_ru`): нормализация к `М` или `Ж`.

### 5.11. OCREngNums — распознавание латиницы и цифр

- CRNN модель через ONNX (`models/OCR/eng+nums`).
- **Алфавит:** `0-9A-Z%'(),-./:*#` (47 символов).
- При формате `OpenVINO` автоматически подменяется на ONNX (OCR-модели не конвертированы в OV).
- **`fix_errors(field_type, text)`:**
  - Даты: `O→0`, `-→.`, 8 цифр → `dd.mm.yyyy`.
  - Пол (`Sex_en`): нормализация к `M` или `F`.
  - Класс водительского удостоверения: фильтрация допустимых символов (`A`, `B`, `C`, `D`, `E`, `M`, `1`).

### 5.12. Модули проверки качества

#### Blur — детекция размытия

- Разбивает изображение на сетку 7×4 патчей по 128 px.
- Каждый патч классифицируется: `Blur0`/`NonBlur` (0.0), `Blur05`/`Blur5` (0.5), `Blur10` (1.0). Патчи `Background`, `Faces`, `Fingers`, `Glare` игнорируются.
- Итоговый скор: `1 - mean(scores)`. Порог: `> 0.9` → `'good'`, иначе `'bad'`.

#### Glare — детекция бликов

- Аналогичная сетка 7×4 × 128 px с тем же `QualityChecker`.
- Логика инвертирована: `quality > 0` → `'bad'` (есть блики), `0` → `'good'`.

#### LCDSpoofing — экранная подделка

- `BinaryClassification` или `Metric` модель.
- Возвращает `(label, confidence)`. Используется как есть — без дополнительных порогов.

#### PrintSpoofing — печатная подделка

- Бинарная модель с жёстким порогом `0.9`: если `conf < 0.9` → `'FAKE'`.

---

## 6. Главный пайплайн (`Pipeline`)

### 6.1. Инициализация

```python
Pipeline(model_format='ONNX', device='cpu', verbose=False)
```

При создании загружаются **базовые модели** (нужны для любого документа):

| Модуль         | Формат                                         |
|----------------|-------------------------------------------------|
| `angle90`      | Выбранный `model_format`                        |
| `doctype`      | Выбранный `model_format`                        |
| `doc_detector` | **Всегда PT** (Ultralytics YOLO)                |
| `words_detector` | Выбранный `model_format`                      |
| `ocr_ru`       | ONNX (если `model_format='OpenVINO'` → ONNX)   |
| `ocr_en`       | ONNX (если `model_format='OpenVINO'` → ONNX)   |
| `lcd_spoofing` | Выбранный `model_format`                        |
| `print_spoofing` | Выбранный `model_format`                      |
| `glare`        | Выбранный `model_format`                        |
| `blur`         | Выбранный `model_format`                        |

**Ленивая загрузка** (после определения типа документа):
- `text_fields` — `TextFieldsDetector` / `TextFieldsDetectorBelarus` / `TextFieldsDetectorUSA`.
- `_passport_page_type_models` — `PassportPageType` (кеш по стране).
- `_seal_detector_models` — `PassportSealDetector` (кеш по стране).

### 6.2. Вызов пайплайна

```python
result = pipeline(
    img_path,               # Path | str | np.ndarray
    ocr=True,               # Выполнять ли OCR
    get_doc_borders=True,   # Детектировать ли рамки документа
    find_text_fields=True,  # Детектировать ли текстовые поля
    check_quality=True,     # Проверять ли качество
    low_quality=True,       # Обрабатывать ли изображения плохого качества
    docconf=0.5,            # Мин. порог уверенности DocType
    img_size=1500,          # Макс. размер длинной стороны
)
```

### 6.3. Последовательность этапов

```
┌──────────────────────────────────────────────────────────────────┐
│  1. _prepare_image                                               │
│     Загрузка → RGB → масштабирование (длинная сторона ≤ img_size)│
├──────────────────────────────────────────────────────────────────┤
│  2. _angle (Angle90)                                             │
│     Определение угла → поворот изображения                       │
├──────────────────────────────────────────────────────────────────┤
│  3. _doc_detector (DocDetector)  [если get_doc_borders=True]     │
│     Детекция границ → исправление перспективы                    │
├──────────────────────────────────────────────────────────────────┤
│  4. _doctype (DocType)                                           │
│     Классификация типа документа                                 │
│     + Проверка площади bbox (KYC_DOC_BBOX_MIN_FRAC ≥ 0.12)      │
│     Если "NONE" → ранний возврат                                 │
├──────────────────────────────────────────────────────────────────┤
│  5. _parse_doc_type + OCROptions.make_options                    │
│     Разбор строки doc_type → (country, state)                    │
│     Создание OCROptions для страны                               │
├──────────────────────────────────────────────────────────────────┤
│  6. _init_text_fields                                            │
│     Ленивая инициализация детектора полей для страны/штата        │
├──────────────────────────────────────────────────────────────────┤
│  7. PassportPageType  [только russia/belarus]                    │
│     Определение типа страницы: centerfold / pages                │
├──────────────────────────────────────────────────────────────────┤
│  8. Проверка качества  [если check_quality=True]                 │
│     _glare → _blur → _print_spoofing → _lcd_spoofing             │
├──────────────────────────────────────────────────────────────────┤
│  9. Ранний выход по качеству  [если low_quality=False]           │
│     Glare='bad' ИЛИ Blur='bad' ИЛИ DocConf < docconf → возврат  │
├──────────────────────────────────────────────────────────────────┤
│  10. Ветвление по типу страницы                                  │
│      ┌─────────────────────────────────────────────┐             │
│      │ passport_pages (russia/belarus)              │             │
│      │ → PassportSealDetector                       │             │
│      │ → Возвращает информацию о печатях            │             │
│      ├─────────────────────────────────────────────┤             │
│      │ passport_centerfold ИЛИ usa                  │             │
│      │ → _fields_detector (TextFieldsDetector*)     │             │
│      │ → _split_words (WordsDetector)               │             │
│      │ → _ocr (OCRRus / OCREngNums)                 │             │
│      └─────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

### 6.4. Парсинг типа документа

`_parse_doc_type(full_doc_type)` → `(country, state)`:

| Вход                        | country      | state       |
|-----------------------------|-------------|-------------|
| `'usa_alabama'`             | `'usa'`      | `'alabama'` |
| `'russia_passport'`         | `'russia'`   | `None`      |
| `'belarus_passport_1996'`   | `'belarus'`  | `None`      |
| `'belarus_passport_2019_2021'` | `'belarus'` | `None`    |

### 6.5. Дополнительная логика DocType

В `_doctype()` после получения результата от модели выполняется проверка площади рамки документа:
- Берутся bbox из результата `DocDetector`.
- Вычисляется `max_frac = max_bbox_area / img_area`.
- Если `max_frac < KYC_DOC_BBOX_MIN_FRAC` (default 0.12) → `doc_type = "NONE"` (артефакт/нерелевантное изображение).

### 6.6. Разделение слов (`_split_words`)

1. Для каждого bbox поля проверяет, есть ли оно в `ocr_options.en_fields` или `ru_fields`.
2. Если поле в `needed_split` → запускает `WordsDetector.predict_transform()` для разбиения на отдельные слова.
3. Если WordsDetector не нашёл слов → fallback: используется целый патч.
4. `_pick_single_licence_number_bbox` — при нескольких bbox `Licence_number` оставляет только один с максимальной confidence.

### 6.7. OCR (`_ocr`)

Для каждого поля из `words_dict`:

1. Определяет модель OCR: если поле в `ru_fields` → `ocr_ru`, в `en_fields` → `ocr_en`. Особый случай для SNILS: каждый второй патч даты → `ocr_ru`.
2. Вызывает `fix_errors()` для постобработки.
3. Постобработка результатов по типам полей:
   - **`Licence_number`:** `O→0`, извлечение только цифр, форматирование `XX XX XXXXXX` (2+2+6).
   - **Даты (SNILS):** через пробел.
   - **Даты (Belarus):** формат `DD MM YYYY` (пробелы). Если пришло 8 цифр → разбиение `DD MM YYYY`.
   - **Даты (прочие):** через точку.
   - **Остальные:** через пробел с удалением двойных пробелов.
4. **Belarus-специфика:** `authority` и `authority2` объединяются в одно поле `authority`.

---

## 7. OCR-опции по типам документов

### 7.1. Иерархия классов

```
OCROptionsClass (базовый, пустые списки)
├── OCROptionsINTPassport      # Внутренний паспорт РФ
├── OCROptionsEXTPassport      # Заграничный паспорт РФ
├── OCROptionsDL               # Водительское удостоверение РФ
├── OCROptionsSNILS            # СНИЛС
├── OCROptionsBelarus          # Паспорт Беларуси
└── OCROptionsUSA              # Водительское удостоверение США
```

Фабрика `OCROptionsClass.make_options(country)`:
- `'belarus'` → `OCROptionsBelarus`
- `'russia'` → `OCROptionsEXTPassport`
- `'usa'` → `OCROptionsUSA`
- прочее → базовый `OCROptionsClass`

### 7.2. Свойства опций

| Свойство              | Назначение                                                        |
|-----------------------|-------------------------------------------------------------------|
| `needed_split`        | Поля, которые нужно разбивать через WordsDetector                 |
| `en_fields`           | Поля для англо-цифрового OCR                                     |
| `ru_fields`           | Поля для русского OCR                                             |
| `needs_licence_rotation` | Поворачивать ли патч серии/номера на 90° (вертикальная серия)  |

### 7.3. Список полей по документам

#### Заграничный паспорт РФ (`OCROptionsEXTPassport`)
- **Английские:** Last_name_en, First_name_en, Middle_name_en, Licence_number, Issue_date, Expiration_date, Birth_date, Birth_place_en, Issue_organization_en, Living_region_en, Sex_en, Issue_organisation_code
- **Русские:** Last_name_ru, First_name_ru, Middle_name_ru, Birth_place_ru, Issue_organization_ru, Living_region_ru, Sex_ru
- **Нужен split:** Licence_number, Birth_place_ru/en, Issue_organization_ru/en, Living_region_ru/en
- **Поворот серии:** Да

#### Паспорт Беларуси (`OCROptionsBelarus`)
- **Все поля английские:** passport_no, surname, names, nationality, date_of_birth, identification_no, sex, place_of_birth, date_of_issue, date_of_expiry, authority, authority2, type, code_of_issuing
- **Русские:** нет
- **Нужен split:** place_of_birth, authority, authority2, nationality

#### ВУ США (`OCROptionsUSA`)
- **Все поля английские:** class, end, rest, firstname, lastname, address, sex, hgt, wgt, eyes, hair, dd, dln, iss, iss_duplicate, exp, dob
- **Русские:** нет
- **Split не нужен:** нет полей в needed_split

---

## 8. `PipelineResults` — результаты пайплайна

### 8.1. Хранилище

- `meta_results` — словарь результатов всех этапов: `{Quality: {}, Angle90: {...}, DocDetector: {...}, ...}`.
- `_timings` — словарь `{method_name: seconds}`.

### 8.2. Свойства-аксессоры

| Свойство                   | Тип возврата        | Описание                                              |
|----------------------------|---------------------|-------------------------------------------------------|
| `ocr`                      | `Dict | None`       | Словарь `{field_name: recognized_text}`                |
| `doctype`                  | `str | None`        | Строка типа документа (`russia_passport` и т.д.)       |
| `quality`                  | `dict`              | `{Glare, Blur, PrintSpoofing, LCDSpoofing, DocConf}`  |
| `rotated_image`            | `np.ndarray`        | Изображение после коррекции угла                       |
| `img_with_fixed_perspective` | `list | None`     | Изображение после исправления перспективы               |
| `text_fields`              | `(bboxes, patches) | None` | Координаты + вырезки текстовых полей            |
| `text_fields_meta`         | `Dict | None`       | Полные метаданные детектора полей                       |
| `words_patches`            | `Dict | None`       | `{field_name: {patches: [...], ocr: [...]}}`           |
| `passport_page_type`       | `str | None`        | `'passport_centerfold'` / `'passport_pages'`           |
| `seal_info`                | `Dict | None`       | `{total_seals, has_seals, last_seal, seals}` (без масок)|
| `full_report`              | `dict`              | Сводный отчёт: DocType + PageType + SealInfo + OCR + Quality + Timings |
| `timings`                  | `dict`              | Тайминги по этапам + `total`                           |

---

## 9. Формат `model.json`

Каждая модель описывается JSON-файлом с типизированной схемой:

```json
{
  "Type": "Classifier | MultiLabelClassification | BinaryClassification | Metric | YoloDetector | YoloSegmentor | OCR",
  "File": "model.onnx",
  "Input": [{
    "Shape": [224, 224, 3],
    "PaddingSize": [0, 0],
    "PaddingColor": [0, 0, 0],
    "Normalization": [0, 1]
  }],
  "Output": [{
    "Classes": ["class_a", "class_b"]
  }],
  "Labels": ["label_0", "label_1"],
  "IOU": 0.2,
  "CLS": 0.5,
  "MaskFilter": 0.8,
  "Lang": "eng|rus",
  "Centers": "centers.pkl"
}
```

Поля зависят от `Type`:
- **Classifier / Multi/BinaryClassification:** `Input`, `Output.Classes` / `Labels`.
- **YoloDetector:** `Input`, `IOU`, `CLS`, `Labels`.
- **YoloSegmentor:** `Input`, `IOU`, `CLS`, `Labels`, `MaskFilter`.
- **OCR:** `Input`, `Lang`.
- **Metric:** `Input`, `Output.Metric`, `Centers`.

---

## 10. Переменные окружения

| Переменная                  | Default | Назначение                                                           |
|-----------------------------|---------|----------------------------------------------------------------------|
| `KYC_DOC_BBOX_MIN_FRAC`    | `0.12`  | Мин. доля площади bbox от изображения для валидной детекции документа |
| `KYC_DOCTYPE_MIN_CONF`     | `0.0`   | Мин. confidence DocType для rejection (ниже → `NONE`)                |
| `KYC_DOCTYPE_MIN_MARGIN`   | `0.0`   | Мин. разница top1-top2 для rejection (ниже → `NONE`)                 |
| `RDOCR_DOCTYPE_MIN_CONF`   | `0.0`   | Legacy-алиас для `KYC_DOCTYPE_MIN_CONF`                              |
| `RDOCR_DOCTYPE_MIN_MARGIN` | `0.0`   | Legacy-алиас для `KYC_DOCTYPE_MIN_MARGIN`                            |

---

## 11. Тайминги

Каждый вызов модуля оборачивается в `_model_call(func, *args)`, который замеряет время выполнения и записывает в `results.timings`:

```python
{
    "_angle": 0.0312,
    "_doc_detector": 0.1245,
    "_doctype": 0.0198,
    "_glare": 0.0876,
    "_blur": 0.0943,
    "_print_spoofing": 0.0156,
    "_lcd_spoofing": 0.0201,
    "_fields_detector": 0.0534,
    "_split_words": 0.0423,
    "_ocr": 0.0867,
    "total": 0.5755
}
```

---

## 12. Используемые модели по модулям

| Модуль               | Архитектура        | Формат по умолчанию | Вход        | Выход                         |
|----------------------|--------------------|----------------------|-------------|-------------------------------|
| Angle90              | EfficientNet-B3    | ONNX                 | 224×224×3   | 4 класса (0°, 90°, 180°, 270°) |
| DocDetector          | YOLOv8-seg         | PT (Ultralytics)     | динамический| 1 класс `border` + маски       |
| DocType              | MobileNetV3-Small  | ONNX                 | 224×224×3   | 14 классов (country+subtype)   |
| PassportPageType     | MobileNetV3-Small  | PT (PyTorch)         | 224×224×3   | 2 класса (centerfold/pages)    |
| PassportSealDetector | YOLOv8-seg         | PT (Ultralytics)     | динамический| 1 класс `seal` + маски         |
| TextFieldsDetector RU| YOLOv8/YOLO11n     | ONNX                 | 640×640×3   | N классов полей документа      |
| TextFieldsDetector BY| YOLO               | PT (Ultralytics)     | динамический| 17 классов полей               |
| TextFieldsDetector USA| YOLO per-state    | PT (Ultralytics)     | динамический| 20 классов полей               |
| WordsDetector        | YOLO               | ONNX                 | 640×640×3   | 1 класс `word`                 |
| OCRRus               | CRNN               | ONNX                 | 31×200×1    | Последовательность символов    |
| OCREngNums           | CRNN               | ONNX                 | 31×200×1    | Последовательность символов    |
| Blur                 | CNN бинарный       | ONNX                 | 128×128     | Класс размытия                 |
| Glare                | CNN бинарный       | ONNX                 | 128×128     | Класс блика                    |
| LCDSpoofing          | CNN/Metric         | ONNX                 | 224×224×3   | Real/Fake + confidence         |
| PrintSpoofing        | CNN бинарный       | ONNX                 | 224×224×3   | Real/Fake + confidence         |

---

## 13. Зависимости

**Основные (всегда нужны):**
- `opencv-python` (`cv2`) — загрузка/обработка изображений, геометрические преобразования
- `numpy` — тензорные операции, массивы
- `onnxruntime` — ONNX-инференс
- `PyYAML` — загрузка конфигурации
- `torch` + `torchvision` — PassportPageType (MobileNetV3)
- `ultralytics` — DocDetector, PassportSealDetector, TextFields Belarus/USA
- `Pillow` — преобразования для torchvision transforms

**Опциональные (по бэкенду):**
- `tensorflow` — H5/SavedModel/PB/TFLite инференс
- `openvino` — OpenVINO инференс
- `coremltools` — CoreML (только macOS)

**Для скриптов:**
- `pandas` — benchmark, MetricPostprocessing
- `scikit-learn` — MetricPostprocessing (NearestNeighbors)
- `py-cpuinfo` — benchmark
- `tqdm` — benchmark
- `argparse` — CLI

---

## 14. Скрипты (`scripts/`)

| Скрипт                         | Назначение                                                                                         |
|--------------------------------|----------------------------------------------------------------------------------------------------|
| `process_img.py`               | CLI: `-i path -f ONNX -d cpu --check_quality --img_size 1500 --show_intermediate`. Создаёт Pipeline, обрабатывает изображение, печатает `full_report`. При `--show_intermediate` сохраняет промежуточные jpg (original, rotated, doc_detection, fixed_perspective, seal_detection, text_fields, word patches). |
| `process_video.py`             | Захват с веб-камеры или URL, покадровая обработка через Pipeline, вывод OCR-результатов.            |
| `benchmark.py`                 | Массовый прогон по директориям с документами, замер среднего времени, сохранение в CSV с метаданными об оборудовании. |
| `ocr_patches.py`               | Экспорт вырезанных патчей слов по полям для дообучения OCR-моделей.                                |
| `test_bbox_detection.py`       | Расчёт метрик IoU / Precision / Recall / F1 для TextFieldsDetector (RU) по JSON-разметке.           |
| `test_ocr_accuracy.py`         | Посимвольное и пословное сравнение OCR-результатов с JSON ground truth.                             |
| `test_doc_borders_on_dataset2.py` | Оценка IoU контуров DocDetector vs VIA-разметка.                                                 |
| `augment_dataset.py`           | Генерация аугментаций (повороты, шум, перспектива) для обучающего датасета.                         |

---

## 15. Поток данных (Data Flow)

```
Входное изображение (Path | str | np.ndarray)
        │
        ▼
  ┌─────────────┐
  │ _prepare_image │──→ RGB ndarray (≤ img_size px по длинной стороне)
  └──────┬──────┘
         ▼
  ┌──────────┐
  │ Angle90  │──→ results.meta_results['Angle90'] = {angle, confidence, warped_img}
  └──────┬───┘
         ▼ (повёрнутое изображение)
  ┌──────────────┐
  │ DocDetector  │──→ results.meta_results['DocDetector'] = {bbox, mask, segm, border_img, warped_img}
  └──────┬───────┘
         ▼ (изображение с исправленной перспективой)
  ┌──────────┐
  │ DocType  │──→ results.meta_results['DocType'] = 'russia_passport'
  └──────┬───┘    results.meta_results['Quality']['DocConf'] = 0.97
         │
         ├──→ _parse_doc_type → (country='russia', state=None)
         ├──→ OCROptions.make_options('russia') → OCROptionsEXTPassport
         ├──→ _init_text_fields('russia') → TextFieldsDetector (ONNX)
         │
         ▼ [только russia/belarus]
  ┌──────────────────┐
  │ PassportPageType  │──→ results.meta_results['PassportPageType'] = {page_type: 'passport_centerfold'}
  └──────┬───────────┘
         │
         ▼ [если check_quality]
  ┌────────────────────────────────────────────┐
  │ Glare → Blur → PrintSpoofing → LCDSpoofing │
  └──────┬─────────────────────────────────────┘
         │  results.meta_results['Quality'] = {
         │      Glare: ('good', 0.0),
         │      Blur: ('good', 0.95),
         │      PrintSpoofing: ('REAL', 0.98),
         │      LCDSpoofing: ('REAL', 0.99),
         │      DocConf: 0.97
         │  }
         │
         ▼ [ветвление]
    ┌────┴────────────────────────────┐
    │ passport_pages                   │ passport_centerfold / usa
    │                                  │
    ▼                                  ▼
  ┌──────────────────┐          ┌──────────────────┐
  │PassportSealDetect│          │ TextFieldsDetector│
  └──────┬───────────┘          └──────┬───────────┘
         │                             │
         │ results['PassportSeal-      │ results['TextFieldsDetector*'] =
         │  Detector'] = {             │   {bbox: [...], warped_img: [...]}
         │  total_seals, has_seals,    │
         │  seals: [...],              ▼
         │  last_seal: {...}           ┌──────────────┐
         │ }                           │ WordsDetector │──→ {field: {patches, ocr}}
         │                             └──────┬───────┘
         │                                    │
         │                                    ▼
         │                             ┌────────────────┐
         │                             │ OCRRus/EngNums │──→ results['OCR'] = {field: text}
         │                             └────────────────┘
         │                                    │
         ▼────────────────────────────────────▼
                      PipelineResults
```

---

## 16. Расширяемость

### Добавление нового типа документа

1. Обучить модель DocType с новым классом (добавить в `model.json` → `Output.Classes`).
2. Создать `TextFieldsDetector{Country}` по аналогии с Belarus/USA в `pipeline_modules/textfields_detector/`.
3. Добавить `OCROptions{Country}` с перечнем полей в `pipeline.py`.
4. Дополнить `_parse_doc_type()` и `_init_text_fields()`.
5. Если документ многостраничный — обучить и подключить `PassportPageType` с весами для новой страны.

### Добавление нового штата США

1. Обучить YOLO-модель на разметке DL штата.
2. Положить веса в `models/TextFields/usa/{state}/PT/best.pt`.
3. Добавить строку штата в `USA_STATES` в `textfields_detector/usa/textfields_detector.py`.
4. Добавить класс DocType (например, `usa_newstate`) в `model.json` классификатора.

### Добавление нового бэкенда инференса

1. Реализовать `__load_xxx` и `__predict_xxx` в `ModelInference`.
2. Привязать по расширению файла в `__init__` конструкторе.

### Добавление нового модуля качества

1. Создать подпакет в `pipeline_modules/` с классом, наследующим `BaseModule`.
2. Реализовать `predict()`, возвращающий `{model_name: (label, confidence)}`.
3. Добавить вызов в `Pipeline.__call__()` и сохранение в `results.meta_results['Quality']`.
