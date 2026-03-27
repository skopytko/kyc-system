#!/bin/bash
# Генерация документов для всех штатов США
# Использует corner_crop + augment_cropped из конфигов

set -e
cd "$(dirname "$0")/../.."
CONFIG_DIR="dataset/generator_docs/usa/config"
GENERATOR="docs_generator/usa/generator.py"

# Штаты с конфигами (исключая config.json — общий шаблон)
STATES="alabama alaska arizona arkansas idaho iowa vermont virginia washington wisconsin wyoming"

for state in $STATES; do
    cfg="$CONFIG_DIR/$state.json"
    if [[ -f "$cfg" ]]; then
        echo "=== $state ==="
        python "$GENERATOR" --config "$cfg" --export-labels
        echo ""
    else
        echo "Пропуск $state: $cfg не найден"
    fi
done

echo "Готово: все штаты обработаны."
