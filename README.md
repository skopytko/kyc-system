# KYC System (OCR-Project)

Сервис для автоматизированного извлечения данных из изображений российских документов (паспорт, водительское удостоверение, СНИЛС и др.) с помощью современных моделей компьютерного зрения и OCR.

**Репозиторий на GitHub:** [github.com/skopytko/kyc-system](https://github.com/skopytko/kyc-system)

---

## 📚 Документация

- [Техническая документация (Google Docs)](https://docs.google.com/document/d/1vIBiNq87qu7HIV8m8zlX_oxiyYIdH3MQVp7AYY8dlDU/edit?usp=sharing)

---

## 🛠️ Установка и подготовка окружения

### 1. Клонирование репозитория

```bash
git clone https://github.com/skopytko/kyc-system.git
cd kyc-system
```

Ранее проект вёлся на GitLab: `https://gitlab.com/triumph3712147/ocr-project.git` (при необходимости укажите его вторым remote: `git remote add gitlab …`).

### 2. Создание и активация conda-окружения

```bash
conda env create -f environment.yml
conda activate kyc-system
```

---

## 🚀 Запуск пайплайна (CLI)

Из **корня репозитория** (чтобы импортировался пакет `kyc_document`):

```bash
PYTHONPATH=. python kyc_document/scripts/process_img.py -i <путь_к_изображению> [доп. параметры]
```

**Основные параметры:**

- `-i`, `--img_path` — путь к изображению для инференса
- `-f`, `--format` — формат моделей: `TFlite`, `ONNX`, `OpenVINO` (по умолчанию: `OpenVINO`)
- `-d`, `--device` — устройство: `cpu` или `gpu` (по умолчанию: `cpu`)
- `--check_quality` — включить проверку качества документа
- `--img_size` — максимальный размер изображения (по умолчанию: 1500)
- `--show_intermediate` — сохранять промежуточные результаты обработки (в папку `intermediate_results/`)

**Пример:**

```bash
PYTHONPATH=. python kyc_document/scripts/process_img.py -i data_samples/russia/passport_centerfold_045.jpg -f ONNX -d cpu --show_intermediate
```

Для веб-приложения удобно задать те же `KYC_MODEL_FORMAT` / `KYC_DEVICE`, что и здесь (`ONNX` / `cpu` и т.д.).

---

## 🌐 Веб‑сервис (FastAPI) — запуск пошагово

1) Активируйте окружение

```bash
conda activate kyc-system
```

2) (Рекомендуется) Подготовьте самоподписанный сертификат для HTTPS

```bash
mkdir -p webapp/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout webapp/certs/server.key -out webapp/certs/server.crt \
  -subj "/C=RU/ST=Moscow/L=Moscow/O=OCR-Project/OU=Dev/CN=localhost"
```

3) Укажите параметры авторизации и запустите сервер

- Переменные окружения:
  - `AUTH_PASSWORD` — пароль для входа (обязательно в проде)
  - `AUTH_SALT` — соль для хэширования (по умолчанию `rdo-default-salt`)
  - `AUTH_SECRET_KEY` — ключ сессий FastAPI

Примечание:

- По умолчанию пароль — `admin123`, если `AUTH_PASSWORD` не задан.
- Пример запуска без HTTPS с переопределением пароля:

```bash
AUTH_PASSWORD='my_pass' uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8001
```

Запуск с HTTPS:

```bash
AUTH_PASSWORD='your_password' AUTH_SALT='your_salt' AUTH_SECRET_KEY='your_secret' \
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8443 \
  --ssl-keyfile webapp/certs/server.key \
  --ssl-certfile webapp/certs/server.crt
```

Альтернатива (без HTTPS, для локальной отладки):

```bash
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8001
```

4) Проверьте доступность

- Откройте `https://localhost:8443/login` или `http://127.0.0.1:8001/login` без HTTPS. Другой порт: флаг `--port` у uvicorn или переменные `KYC_PORT` / `PORT` при `python -m webapp.main`.

После входа загрузите изображение документа, при необходимости включите проверку качества и нажмите «Распознать».

---

## 🏗️ Структура проекта

```
ocr-project/
│
├── data_samples/                # Примеры документов по типам (russia, belarus)
├── dataset/                     # Датасеты проекта (см. примечание ниже)
│   ├── dataset_for_doc_detector/    # Датасеты для определения границ документа
│   │   ├── yolo_dataset/                # Основной датасет для обучения/тестирования модели 
│   │   ├── dataset_passport_centerfold/ # Центр. развороты паспортов с разметкой (COCO)
│   │   ├── dataset_passport_pages/      # Другие страницы паспорта с разметкой (COCO)
│   │   └── passport_midv-2020/          # MIDV-2020: фото паспортов с json-аннотациями
│   ├── dataset_for_doc_type/        # Датасеты для классификации типа документа
│   │   ├── main/                    # Основной датасет для классификации типа документа (train/val/test)
│   │   └── passport/                # Дополнительные изображения для расширения основного датасета
│   └── dataset_passport_page/       # Страницы паспортов (для генерации датасета классификации типа документа и страниц паспорта)
│
├── model_borders/               # Скрипты и модели для определения границ документа
│   ├── train.py                 # Скрипт для обучения модели сегментации границ (YOLOv8, ultralytics)
│   ├── scripts/                 # Вспомогательные скрипты для подготовки данных и тестирования
│   │   ├── test_borders.py                  # Тестирование и визуализация результатов сегментации на тестовом наборе
│   │   ├── convert_quads_to_coco.py         # Конвертация разметки (quads) в формат COCO
│   ├── runs/                    # Результаты обучения и валидации моделей
│
├── model_doc_type/               # Модель и скрипты для классификации типа документа
│   ├── train.py                  # Скрипт обучения классификатора типа документа
│   ├── test.py                   # Скрипт тестирования классификатора типа документа
│   ├── results.csv               # Результаты тестирования
│   └── runs/                     # Чекпоинты и графики обучения
│
├── model_passport_page_type/     # Модель и скрипты для классификации страниц паспорта
│   ├── train.py                  # Скрипт обучения классификатора страниц паспорта
│   ├── test.py                   # Скрипт тестирования классификатора страниц паспорта
│   ├── results.csv               # Результаты тестирования
│   └── runs/                     # Чекпоинты и графики обучения
│
├── kyc_document/                # Пакет: пайплайн OCR и обработка документов
│   ├── document_processing/
│   │   ├── processing/          # препроцессинг, инференс, постпроцессинг, обёртки моделей
│   │   ├── pipeline/            # сценарий Pipeline
│   │   ├── pipeline_modules/    # детекторы, классификаторы, OCR, качество
│   │   ├── config/              # models_path.yaml
│   │   └── models/              # веса ONNX/OpenVINO и др.
│   └── scripts/                 # process_img.py и утилиты
│
├── webapp/                      # FastAPI: авторизация, /api/ocr, шаблоны
│   ├── main.py
│   ├── capture_validation.py
│   ├── templates/
│   └── static/                  # превью загрузок
│
├── scripts/                     # Утилиты для подготовки датасетов
│   └── doc_type/                # Скрипты для генерации и подготовки датасета для классификации типа документа
│       ├── combine_page3_page4.py   # Склеивание изображений страниц (page3 и page4) в один центр. разворот
│       └── crop_for_doc_type.py     # Автоматическое обрезание документов по предсказанным границам (YOLO)
│
├── requirements.txt
└── README.md
```

> **Примечание:**
> Папка `dataset` физически расположена в сетевой папке сервера "Триумф" по пути: `\\fs\Папка Проектов Сетевая\2 Программисты\Копытько\dataset`.
