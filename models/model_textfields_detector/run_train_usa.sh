#!/usr/bin/env bash
# Поочерёдное обучение TextFields Detector для всех штатов USA.
# Датасеты: dataset/usa/<state>/cropped
# Результаты: runs/usa_<state>/train/weights/best.pt

set -e
cd "$(dirname "$0")/../.."
ROOT="$PWD"
SCRIPT="$ROOT/models/model_textfields_detector/train_usa.py"
USA_ROOT="$ROOT/dataset/usa"

if [[ ! -f "$SCRIPT" ]]; then
  echo "Not found: $SCRIPT"
  exit 1
fi

# Список штатов — все подпапки dataset/usa с cropped
STATES=()
for d in "$USA_ROOT"/*; do
  if [[ -d "$d" && -d "$d/cropped" && -d "$d/cropped/labels" ]]; then
    state=$(basename "$d")
    STATES+=("$state")
  fi
done

if [[ ${#STATES[@]} -eq 0 ]]; then
  echo "No states with dataset/usa/<state>/cropped and labels found."
  exit 1
fi

echo "States to train: ${STATES[*]}"
echo ""

for state in "${STATES[@]}"; do
  echo "=============================================="
  echo "Training USA state: $state"
  echo "=============================================="
  USA_STATE="$state" python "$SCRIPT" || {
    echo "Training failed for state: $state"
    exit 1
  }
  echo ""
done

echo "All USA states trained."
