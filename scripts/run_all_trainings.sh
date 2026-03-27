#!/bin/bash
# Запуск всех обучений моделей (последовательно)
# Требуются: dataset с raw/cropped, зависимости (torch, ultralytics, timm, mlflow и т.д.)

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

run_train() {
    local name="$1"
    local cmd="$2"
    echo ""
    echo "=========================================="
    echo "=== $name ==="
    echo "=========================================="
    if eval "$cmd"; then
        echo "[OK] $name"
    else
        echo "[FAIL] $name"
    fi
}

# 1. DocType (14 классов: belarus, russia, usa)
run_train "model_doc_type" "DOCTYPE_MAX_PER_CLASS=150 DOCTYPE_EPOCHS=50 DOCTYPE_BATCH=8 DOCTYPE_PATIENCE=5 python $ROOT/models/model_doc_type/train.py"

# 2. Borders (YOLO-seg границ документа)
run_train "model_borders" "python $ROOT/models/model_borders/train.py"

# 3. Angles90 (повороты 0/90/180/270)
run_train "model_angles90" "python $ROOT/models/model_angles90/train.py"

# 4. TextFields Detector Belarus (YOLO)
run_train "model_textfields_detector_belarus" "python $ROOT/models/model_textfields_detector/belarus/train.py"

# 5. Passport Seal — пропуск, нужен dataset/dataset_for_seal_detector
if [ -d "$ROOT/dataset/dataset_for_seal_detector" ]; then
    run_train "model_passport_seal" "cd $ROOT/models/model_passport_seal && python train.py"
else
    echo ""
    echo "[SKIP] model_passport_seal — нужен dataset/dataset_for_seal_detector"
fi

# 6. Passport Page Type — пропуск, нужен dataset/dataset_for_doc_type
if [ -d "$ROOT/dataset/dataset_for_doc_type" ]; then
    run_train "model_passport_page_type" "cd $ROOT/models/model_passport_page_type && python train.py"
else
    echo ""
    echo "[SKIP] model_passport_page_type — нужен dataset/dataset_for_doc_type"
fi

echo ""
echo "=========================================="
echo "Все обучения завершены."
echo "=========================================="
